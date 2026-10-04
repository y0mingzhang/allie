"""Search oracle for MoE checkpoints (Allie 2.0), with a tree key-value cache.

The model is built by the checkpoint's own training code (allie.eval.score.training_code: this package, or with
source= a pre-package checkpoint's frozen source). Only attention differs from training: roots prefill once into
a slot pool, and each search node appends one token that attends to its path (root prefix and ancestors).
"""

import os
import socket
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F

from allie.search.board import (
    advance_boards,
    advance_clocks,
    encode,
    predicted_seconds,
    root_other_previous,
)
from allie.eval.score import training_code

CONTEXT = 1025


def network(state, source=None):
    """The model module that trained this checkpoint."""
    return training_code(state, source)[1]


def load_model(checkpoint, source=None):
    """(model in eval mode on cuda:0, resolved checkpoint path)."""
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "directory" in state:
        checkpoint = checkpoint.parent / state["directory"] / "model.pt"
        state = torch.load(
            checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
    mm = network(state, source)
    torch.backends.cuda.matmul.allow_tf32 = True  # as in training

    if not dist.is_initialized():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        dist.init_process_group("gloo", rank=0, world_size=1)
    torch.cuda.set_device(0)
    model = mm.create_model(mm.Config(**state["config"]))
    # forward() restates GPT.forward without the header-only inputs of newer sources
    assert getattr(model, "tc_header", True) and not getattr(model, "header_feats", 0)
    floor = state["config"]["arch"].get("moe_gate_floor", 0.0)
    moes = [b.mlp for b in model.blocks if hasattr(b.mlp, "router")]
    assert moes and all(getattr(m, "gate_floor", 0.0) == floor for m in moes), floor
    model.scalars.data = state["model"]["scalars"].to("cuda", model.scalars.dtype)
    model.load_state_dict(state["model"])
    inf = state["inference"]
    model.split_embed = inf["split_embed"]
    for key in ("angular_freq", "cos", "sin"):
        getattr(model.yarn, key).copy_(inf["yarn"][key].to("cuda"))
    model.yarn.attn_scale = inf["yarn"]["attn_scale"]
    # every game fits both windows, so attention is plain causal within a game
    assert min(inf["ws_short"], inf["ws_long"]) * 128 >= CONTEXT
    for p in model.parameters():
        for key in ("fp32", "main_grad"):
            if hasattr(p, key):
                delattr(p, key)
    model.eval().requires_grad_(False)
    return model, checkpoint


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


@torch.no_grad()
def forward(m, ids, pos, feats, boards, previous, attend, last=None):
    """The training GPT.forward (eval, non-checkpointed path) on T flat tokens at arbitrary
    positions. previous(e): each token's predecessor embedding (zeros at a game start);
    attend(i, q, k, v): layer i attention. Returns (logits of rows `last`, embeddings e)."""
    n, s = m.num_layers, m.scalars
    x = m.embed(ids) if m.split_embed else F.embedding(ids, m.lm_head.weight)
    if m.use_feats:
        t = feats[:, : m.use_feats].float()
        v = torch.log1p(t.clamp(min=0))[..., None] / 10
        w = torch.pi * 2.0 ** torch.arange(8, device=t.device)
        f = torch.cat((torch.sin(v * w), torch.cos(v * w), v, torch.ones_like(v)), -1)
        f = (f * (t >= 0)[..., None]).flatten(1)
        f = F.pad(f, (0, 64 - f.shape[1]))
        x = x + f.type_as(x) @ m.feat_embed.weight.type_as(x)
    x = e = x + m.board(boards, x.dtype)
    x = x + s[3 * n] * torch.sigmoid(m.smear_gate(x[:, :16])) * previous(e)
    x = x0 = norm(x)
    x02 = norm(m.embed2(ids))
    ve = [embed(ids) for embed in m.value_embeds]
    ve = ve + [None] * (n - 2 * len(ve)) + ve
    skip_in = [i * n // 16 for i in (2, 4, 6)]
    skip_out = [9 * n // 16 + i for i in range(3)]
    lam, sa = s[:n], s[n : 3 * n].view(-1, 2)
    x0l = m.x0_lambdas.view(-1, 2)
    if not m.use_x0:
        x0l = torch.stack((x0l[:, 0] * 0, x0l[:, 1]), 1)
    cos, sin = m.yarn.cos[pos][:, None], m.yarn.sin[pos][:, None]

    def rotary(z):
        a, b = z.chunk(2, dim=-1)
        return torch.cat((a * cos + b * sin, a * (-sin) + b * cos), -1)

    skips, backout, j = [], None, 0
    for i, block in enumerate(m.blocks):
        if i in skip_out:
            gate = (
                torch.sigmoid(s[3 * n + 2 + j])
                * 2
                * torch.sigmoid(m.skip_gates[j](x0[..., :16]))
            )
            x = x + gate * skips.pop()
            j += 1
        if i == 0:
            x = (lam[0] + x0l[0, 0]) * x + x0l[0, 1] * x02
        else:
            x = lam[i] * x + x0l[i, 0] * x0 + x0l[i, 1] * x02
        a, h = block.attn, norm(x)
        d, heads, hd = a.dim, a.num_heads, a.head_dim
        qkv = F.linear(h, sa[i, 0] * a.qkvo_w[: 3 * d].type_as(h)).view(
            -1, 3 * heads, hd
        )
        q, k, v = qkv.chunk(3, dim=-2)
        q, k = rotary(norm(q)), rotary(norm(k))
        if ve[i] is not None:
            v = v + 2 * torch.sigmoid(a.value_embed_gate(h[..., :16])).view(
                -1, heads, 1
            ) * ve[i].view_as(v)
        y = attend(i, q, k, v) * torch.sigmoid(a.attn_gate(h[..., :16])).view(
            -1, heads, 1
        )
        x = x + F.linear(y.reshape(-1, d), sa[i, 1] * a.qkvo_w[3 * d :].type_as(y))
        x = x + block.mlp(norm(x))
        if i in skip_in:
            skips.append(x)
        if i == skip_out[-1]:
            backout = x
    if last is not None:
        x, backout = x[last], backout[last]
    x = norm(x - s[3 * n + 1] * backout)
    return 23 * torch.sigmoid((m.lm_head(x) + 5) / 7.5), e


class MoEOracle:
    """Owns the model and a slot pool of keys/values. One serial caller; reset() per root batch."""

    def __init__(
        self, checkpoint, source=None, slots=1 << 18, rows=1 << 17, gather=1 << 18
    ):
        self.model, self.checkpoint = load_model(checkpoint, source)
        m = self.model
        a = m.blocks[0].attn
        self.heads, self.head_dim, self.width = a.num_heads, a.head_dim, a.dim
        self.scale = m.yarn.attn_scale
        shape = (m.num_layers, slots, self.heads, self.head_dim)
        self.k = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
        self.v = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
        self.e = torch.empty(slots, self.width, dtype=torch.bfloat16, device="cuda")
        self.path = torch.zeros(rows, CONTEXT, dtype=torch.int64, device="cuda")
        self.origin = torch.zeros(rows, dtype=torch.int64, device="cuda")  # rotary offset
        self.slots, self.rows, self.gather = slots, rows, gather
        self.capacity = rows
        self.reset()

    def reset(self):
        self.next_slot = self.next_row = 0
        self.new_tokens = 0

    def handles(self, prefixes, features, clock_rule="predicted"):
        return MoEHandles(self, prefixes, features, clock_rule)

    def _take(self, n, rows):
        if self.next_slot + n > self.slots or self.next_row + rows > self.rows:
            raise RuntimeError("tree cache full; reset between root batches")
        s, r = self.next_slot, self.next_row
        self.next_slot += n
        self.next_row += rows
        return s, r

    def prefill(self, prefixes, features, budget=16384, origins=None):
        """Last-position logits of each root prefix; returns (logits [R, V], path rows [R]).
        origins: per-root rotary offsets (moe_parity only; search uses 0)."""
        origins = np.zeros(len(prefixes), np.int64) if origins is None else np.asarray(origins)
        out, rows = [], []
        lo = 0
        while lo < len(prefixes):
            hi, total = lo, 0
            while hi < len(prefixes) and (
                hi == lo or total + len(prefixes[hi]) <= budget
            ):
                total += len(prefixes[hi])
                hi += 1
            z, r = self._prefill(prefixes[lo:hi], features[lo:hi], origins[lo:hi])
            out.append(z)
            rows.extend(r)
            lo = hi
        return torch.cat(out).float().cpu().numpy(), np.array(rows)

    def _prefill(self, prefixes, features, origins):
        lengths = [len(p) for p in prefixes]
        assert all(11 <= n <= CONTEXT for n in lengths)
        total, count, width = sum(lengths), len(prefixes), max(lengths)
        slot, row = self._take(total, count)
        dev = "cuda"
        ids = torch.as_tensor(np.concatenate(prefixes), device=dev)
        pos = torch.as_tensor(
            np.concatenate([np.arange(n) + o for n, o in zip(lengths, origins)]), device=dev
        )
        self.origin[row : row + count] = torch.as_tensor(origins, device=dev)
        feats = torch.as_tensor(
            np.concatenate(features), dtype=torch.float32, device=dev
        )
        boards = torch.as_tensor(
            np.concatenate([encode(np.asarray(p)[None])[0] for p in prefixes]),
            device=dev,
        )
        starts = np.cumsum([0] + lengths[:-1])
        slots = torch.arange(slot, slot + total, device=dev)
        # padded [R, width] view of the flat tokens; pad entries point at an extra zero row
        grid = np.full((count, width), total)
        for r, (a, n) in enumerate(zip(starts, lengths)):
            grid[r, :n] = np.arange(a, a + n)
            self.path[row + r, :n] = slots[a : a + n]
        grid = torch.as_tensor(grid, device=dev)
        valid = grid < total
        ix = torch.arange(width, device=dev)
        mask = (ix[:, None] >= ix[None, :]) & valid[:, None, :]
        first = torch.as_tensor(starts, device=dev)

        def previous(e):
            p = torch.cat((e.new_zeros(1, e.shape[1]), e[:-1]))
            p[first] = 0
            self.e[slots] = e
            return p

        def attend(i, q, k, v):
            self.k[i, slots], self.v[i, slots] = k, v
            pad = lambda z: torch.cat((z, z.new_zeros(1, *z.shape[1:])))[
                grid
            ].transpose(1, 2)
            y = F.scaled_dot_product_attention(
                pad(q), pad(k), pad(v), attn_mask=mask[:, None], scale=self.scale
            )
            return y.transpose(1, 2)[valid]

        last = torch.as_tensor(starts + np.array(lengths) - 1, device=dev)
        z, _ = forward(self.model, ids, pos, feats, boards, previous, attend, last)
        self.new_tokens += total
        return z, list(range(row, row + count))

    def extend(self, parents, tokens, lengths, feats, boards):
        """Logits of children: parent path rows, appended token, child length (parent + 1)."""
        n = len(parents)
        slot, row = self._take(n, n)
        dev = "cuda"
        parents = torch.as_tensor(parents, device=dev, dtype=torch.long)
        lengths_t = torch.as_tensor(lengths, device=dev, dtype=torch.long)
        rows = torch.arange(row, row + n, device=dev)
        slots = torch.arange(slot, slot + n, device=dev)
        width = int(lengths.max())
        self.path[rows, :width] = self.path[parents, :width]
        self.path[rows, lengths_t - 1] = slots
        self.origin[rows] = self.origin[parents]
        pos = lengths_t - 1 + self.origin[rows]
        before = self.path[parents, lengths_t - 2]
        # attention chunks of similar lengths, each gathering at most `gather` keys
        order = np.argsort(lengths, kind="stable")
        chunks, lo = [], 0
        while lo < n:
            hi = lo + 1
            while hi < n and (hi + 1 - lo) * int(lengths[order[hi]]) <= self.gather:
                hi += 1
            w = int(lengths[order[hi - 1]])
            ix = torch.as_tensor(order[lo:hi], device=dev)
            keys = self.path[rows[ix], :w]
            live = torch.arange(w, device=dev)[None, :] < lengths_t[ix, None]
            chunks.append((ix, keys, live[:, None, None, :]))
            lo = hi

        def previous(e):
            self.e[slots] = e
            return self.e[before]

        def attend(i, q, k, v):
            self.k[i, slots], self.v[i, slots] = k, v
            y = torch.empty_like(q)
            for ix, keys, live in chunks:
                kk = self.k[i][keys].transpose(1, 2)
                vv = self.v[i][keys].transpose(1, 2)
                y[ix] = F.scaled_dot_product_attention(
                    q[ix][:, :, None], kk, vv, attn_mask=live, scale=self.scale
                )[:, :, 0]
            return y

        ids = torch.as_tensor(tokens, device=dev, dtype=torch.long)
        feats = torch.as_tensor(feats, device=dev, dtype=torch.float32)
        boards = torch.as_tensor(boards, device=dev)
        z, _ = forward(self.model, ids, pos, feats, boards, previous, attend)
        self.new_tokens += n
        return z.float().cpu().numpy(), np.arange(row, row + n)


class MoEHandles:
    """Evaluates search nodes: each node's causal board and clock features, then MoEOracle.extend."""

    def __init__(self, base, prefixes, features, clock_rule="predicted"):
        assert clock_rule in ("predicted", "zero")
        self.base, self.clock_rule = base, clock_rule
        prefixes = [np.asarray(p, np.int64) for p in prefixes]
        features = [np.asarray(f, np.float32) for f in features]
        self.root_logits, root_rows = base.prefill(prefixes, features)
        n, cap = len(prefixes), base.capacity
        self.rows = np.full(cap, -1, np.int64)
        self.lengths = np.zeros(cap, np.int64)
        self.boards = np.zeros((cap, 68), np.uint8)
        self.feats = np.full((cap, 3), -1.0, np.float32)
        self.other_previous = np.full(cap, -1.0, np.float32)
        self.inc = np.full(cap, -1.0, np.float32)
        self.elapsed = np.zeros(cap, np.float32)
        self.owners = np.full(cap, -1, np.int64)
        self.owners[:n] = np.arange(n)
        self.per_root_queries = np.zeros(n, np.int64)
        self.queries = 0
        for i, (p, f) in enumerate(zip(prefixes, features)):
            self.rows[i], self.lengths[i] = root_rows[i], len(p)
            self.boards[i] = encode(p[None])[0, -1]
            self.feats[i] = f[-1]
            self.inc[i] = p[2] - 10 if 10 <= p[2] < 191 else -1
            self.other_previous[i] = root_other_previous(p, f, self.inc[i])
        if clock_rule == "predicted":
            self.elapsed[:n] = predicted_seconds(self.root_logits)

    def __call__(self, handles):
        h = np.asarray(handles, np.int64)
        assert h.ndim == 2 and h.shape[1] == 4 and len(h)
        ids, parents, tokens, lengths = h.T
        cap = len(self.rows)
        assert ids.min() >= 0 and parents.min() >= 0 and max(ids.max(), parents.max()) < cap
        assert (self.rows[ids] < 0).all(), (
            "a search node must be evaluated exactly once"
        )
        assert (self.rows[parents] >= 0).all() and (
            lengths == self.lengths[parents] + 1
        ).all()
        assert tokens.min() >= 378 and tokens.max() < 2346 and lengths.max() <= CONTEXT
        self.owners[ids] = self.owners[parents]
        np.add.at(self.per_root_queries, self.owners[ids], 1)
        self.boards[ids] = advance_boards(self.boards[parents], tokens)
        self.feats[ids], self.other_previous[ids] = advance_clocks(
            self.feats[parents], self.other_previous[parents], self.lengths[parents],
            self.inc[parents], self.elapsed[parents],
        )  # fmt: skip
        self.inc[ids] = self.inc[parents]
        z, rows = self.base.extend(
            self.rows[parents], tokens, lengths, self.feats[ids], self.boards[ids]
        )
        self.rows[ids], self.lengths[ids] = rows, lengths
        if self.clock_rule == "predicted":
            self.elapsed[ids] = predicted_seconds(z)
        self.queries += len(h)
        return z
