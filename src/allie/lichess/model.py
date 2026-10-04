"""Allie 2.0 in plain PyTorch, CPU or GPU: no Triton kernels, flex attention or training code.

The forward restates the training GPT.forward (allie.model.nanogpt, eval path) over flat tokens,
with each game's keys and values in a Cache. It matches the trained model up to floating-point
summation order (analysis/lichess/parity.py). Weights come from export.py: BF16 in F.linear
layout with the attention lambdas folded in; router, balancing bias, centre and scalars FP32.
"""

import json
import warnings
from pathlib import Path

import torch
from torch.nn import functional as F

from .tokens import CONTEXT


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def swiglu(x):
    a, b = x.chunk(2, dim=-1)
    return F.silu(a) * b


MATRICES = ("qkv", "o", "fc", "proj", "up", "down", "shared_up", "shared_down")
FP32 = ("router", "moe_bias", "mu", "scalars", "x0_lambdas")


def read(path):
    """(name, CPU tensor) of a safetensors file, read in file order straight into fresh
    tensors: no file mapping and no second copy, so the resident size is the model's."""
    dtypes = dict(BF16=torch.bfloat16, F16=torch.float16, F32=torch.float32, I8=torch.int8)
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        for name, h in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"]):
            t = torch.empty(h["shape"], dtype=dtypes[h["dtype"]])
            lo, hi = h["data_offsets"]
            assert t.numel() * t.element_size() == hi - lo, name
            f.seek(8 + n + lo)
            f.readinto(t.view(-1).view(torch.uint8).numpy())
            yield name, t


def quantize(w):
    """Symmetric int8 per output row: (int8 weights, BF16 scales). In slices of the first
    dimension, so the FP32 copy is small."""
    q, s = torch.empty(w.shape, dtype=torch.int8), torch.empty(w.shape[:-1], dtype=torch.bfloat16)
    for a in range(0, len(w), 16):
        x = w[a : a + 16].float()
        r = x.abs().amax(-1).clamp_min(1e-12) / 127
        q[a : a + 16], s[a : a + 16] = (x / r[..., None]).round().clamp(-127, 127), r
    return q, s


class Model:
    """active_experts: route each token through only the best `active_experts` of its top-k,
    with the gates they have in the top-k (the speed knob; None = all). int8: the block
    matrices as int8 weights with per-row scales (CPU, BF16 activations): half the memory.
    backend: "fast" runs step() in fast.py: its C++ kernels on CPU (BF16 activations), CUDA
    graphs of forward() on GPU; "torch" in the PyTorch code below (the reference); None: fast
    where it applies (CPU with BF16, any GPU) and works."""

    def __init__(self, path, device="cpu", dtype=torch.bfloat16, active_experts=None, int8=False,
                 backend=None, threads=None):  # fmt: skip
        path = Path(path)
        self.config = c = json.loads((path / "config.json").read_text())
        self.device, self.dtype = torch.device(device), dtype
        self.layers, self.width = c["layers"], c["width"]
        self.heads, self.head_dim = c["heads"], c["head_dim"]
        self.topk, self.floor, self.scale = c["topk"], c["gate_floor"], c["attn_scale"]
        self.keep = active_experts or self.topk
        assert 1 <= self.keep <= self.topk
        assert not int8 or (self.device.type == "cpu" and dtype == torch.bfloat16)
        self.w, self.scales = {}, {}
        for k, v in read(path / "model.safetensors"):
            kind = k.split(".")[-1]
            if int8 and kind in MATRICES:
                self.w[k], self.scales[k] = quantize(v)
            else:
                self.w[k] = v.to(self.device, torch.float32 if kind in FP32 else dtype)
        for i in range(self.layers):  # the output gate and value-embedding gate read the same
            gates = [self.w.pop(f"{i}.attn_gate"), self.w.pop(f"{i}.ve_gate", None)]
            self.w[f"{i}.gates"] = torch.cat([g for g in gates if g is not None])
        self.moe_mode = None  # None: by device and tokens; or token / group / gather / dense
        self.views = {  # each routed expert's matrices (and int8 scales), indexed in Python
            k: [(v[e], self.scales[k][e] if k in self.scales else None) for e in range(len(v))]
            for k, v in self.w.items()
            if k.split(".")[-1] in ("up", "down")
        }
        s = self.w["scalars"]
        n = self.layers
        self.lam, self.x0l = s[:n].tolist(), self.w["x0_lambdas"].view(-1, 2).tolist()
        self.smear, self.backout = s[3 * n].item(), s[3 * n + 1].item()
        self.skip_lambdas = s[3 * n + 2 : 3 * n + 5]
        self.ve = c["value_embeds"]
        self.fast = self.graphs = None
        assert backend in (None, "fast", "torch"), backend
        if backend != "torch" and (backend or self.device.type == "cuda" or dtype == torch.bfloat16):
            try:
                from .fast import Fast, Graphs

                if self.device.type == "cuda":
                    self.graphs = Graphs(self, strict=backend == "fast")
                else:
                    self.fast = Fast(self, threads)
            except Exception as e:
                if backend == "fast":
                    raise
                warnings.warn(f"Allie's fast backend is unavailable, using PyTorch: {e}")

    def board(self, states):
        w, dt = self.w, self.dtype
        x = (
            F.one_hot(states[:, :64].long(), 13)
            .to(dt)
            .view(-1, 8, 8, 13)
            .permute(0, 3, 1, 2)
        )
        gelu = lambda z: F.gelu(z, approximate="tanh")
        x = gelu(F.conv2d(x, w["board.first"], padding=1))
        for i in range(2):
            x = x + gelu(F.conv2d(x, w[f"board.residual.{i}"], padding=1))
        x = F.conv2d(x, w["board.squeeze"]).flatten(1)
        side = F.one_hot(states[:, 64].long(), 2)
        rights = F.one_hot(states[:, 65].long(), 16)
        ep = F.one_hot(states[:, 66].long(), 9)
        meta = F.pad(torch.cat((side, rights, ep), -1), (0, 5)).to(dt)
        x = F.layer_norm(torch.cat((x, meta @ w["board.meta"]), -1), (544,))
        return (x @ w["board.output"]) * states[:, 67:68].to(dt)

    def clock(self, feats):
        t = feats.float()
        v = torch.log1p(t.clamp(min=0))[..., None] / 10
        f = torch.pi * 2.0 ** torch.arange(8, device=t.device)
        f = torch.cat((torch.sin(v * f), torch.cos(v * f), v, torch.ones_like(v)), -1)
        f = F.pad((f * (t >= 0)[..., None]).flatten(1), (0, 64 - 18 * t.shape[1]))
        return f.to(self.dtype) @ self.w["feat_embed"]

    def linear(self, x, key, e=None):
        if e is not None:  # a routed expert: its weights (and int8 scales) as prepared views
            w, s = self.views[key][e]
        else:
            w, s = self.w[key], self.scales.get(key)
        if s is None:
            return F.linear(x, w)
        return torch._weight_int8pack_mm(x.contiguous(), w, s)

    def ffn(self, x, up, down, e=None):
        return self.linear(swiglu(self.linear(x, up, e)), down, e)

    def mlp(self, i, h):
        if f"{i}.fc" in self.w:
            return self.ffn(h, f"{i}.fc", f"{i}.proj")
        w, k = self.w, self.topk
        s = torch.sigmoid(F.linear(h.float() - w[f"{i}.mu"], w[f"{i}.router"]))
        idx = torch.topk(s + w[f"{i}.moe_bias"], k, dim=-1).indices
        gate = s.gather(1, idx)
        gate = gate * (k**0.5 / gate.sum(-1, keepdim=True).clamp_min(self.floor))
        gate = gate[:, : self.keep].to(h.dtype).float()  # BF16 gates, FP32 sums (trainer)
        idx = idx[:, : self.keep]
        mode = self.moe_mode or (
            ("gather" if len(h) <= 4 else "dense" if len(h) <= 64 else "group")
            if h.is_cuda
            else ("token" if len(h) <= 2 else "group")
        )
        out = getattr(self, mode)(i, h, idx, gate)
        return out.to(h.dtype) + self.ffn(h, f"{i}.shared_up", f"{i}.shared_down")

    def token(self, i, h, idx, gate):
        """Each token's experts one by one: the fewest operations for one or two tokens."""
        up, down = f"{i}.up", f"{i}.down"
        ys = [self.ffn(h[t : t + 1], up, down, e) for t, es in enumerate(idx.tolist()) for e in es]
        y = torch.cat(ys).float().view(*gate.shape, -1)
        return (y * gate[..., None]).sum(1)

    def group(self, i, h, idx, gate):
        """Tokens grouped by expert: one matrix product per routed expert."""
        idx, gate = idx.flatten(), gate.flatten()
        order = idx.argsort(stable=True)
        experts, counts = torch.unique_consecutive(idx[order], return_counts=True)
        rows, gate = order // self.keep, gate[order]
        out = torch.zeros(h.shape, dtype=torch.float32, device=h.device)
        lo = 0
        for e, n in zip(experts.tolist(), counts.tolist()):
            r = rows[lo : lo + n]
            y = self.ffn(h[r], f"{i}.up", f"{i}.down", e)
            out.index_add_(0, r, y.float() * gate[lo : lo + n, None])
            lo += n
        return out

    def gather(self, i, h, idx, gate):
        """The selected experts' weights gathered into one batched product: no host syncs
        (GPU, few tokens)."""
        e = idx.flatten()
        x = h.repeat_interleave(self.keep, 0)[:, :, None]
        y = swiglu(torch.bmm(self.w[f"{i}.up"][e], x)[..., 0])
        y = torch.bmm(self.w[f"{i}.down"][e], y[..., None])[..., 0]
        return (y.float().view(len(h), self.keep, -1) * gate[..., None]).sum(1)

    def dense(self, i, h, idx, gate):
        """Every token through every expert, then the gated sum: reads each expert once and
        needs no host syncs (GPU, a batch of games)."""
        up, down = self.w[f"{i}.up"], self.w[f"{i}.down"]
        y = swiglu(F.linear(h, up.flatten(0, 1)).view(len(h), len(up), -1))  # [T, E, H]
        y = torch.bmm(y.transpose(0, 1), down.transpose(1, 2))  # [E, T, D]
        combine = torch.zeros(len(h), len(up), device=h.device).scatter_(1, idx, gate)
        return torch.einsum("te,etd->td", combine, y.float())

    @torch.inference_mode()
    def forward(self, ids, pos, feats, boards, previous, attend, last=None):
        """Logits [rows of `last`, 2432] of flat tokens at in-game positions pos.
        previous(e): each token's predecessor embedding (zeros at a game start);
        attend(i, q, k, v): layer i attention, [T, heads, head_dim] each."""
        w, n, dt = self.w, self.layers, self.dtype
        x = F.embedding(ids, w["embed"]) + self.clock(feats)
        x = e = x + self.board(boards)
        gate = torch.sigmoid(F.linear(x[:, :16], w["smear_gate"]))
        x = x + self.smear * gate * previous(e)
        x = x0 = norm(x)
        x02 = norm(F.embedding(ids, w["embed2"]))
        ve = [F.embedding(ids, w[f"value_embed.{j}"]) for j in range(self.ve)]
        ve = ve + [None] * (n - 2 * len(ve)) + ve
        skip_in = [i * n // 16 for i in (2, 4, 6)]
        skip_out = [9 * n // 16 + i for i in range(3)]
        # rotary as z * (cos, cos) + (b, a) * (sin, -sin): the training code's products and sums
        cos, sin = w["cos"][pos][:, None].to(dt), w["sin"][pos][:, None].to(dt)
        cos, sin = torch.cat((cos, cos), -1), torch.cat((sin, -sin), -1)

        def rotary(z):
            a, b = z.chunk(2, dim=-1)
            return z * cos + torch.cat((b, a), -1) * sin

        skips, backout, j = [], None, 0
        heads, hd = self.heads, self.head_dim
        for i in range(n):
            if i in skip_out:
                g = torch.sigmoid(self.skip_lambdas[j]) * 2
                g = g * torch.sigmoid(F.linear(x0[:, :16], w[f"skip_gate.{j}"]))
                x = x + g.to(dt) * skips.pop()
                j += 1
            if i == 0:
                x = (self.lam[0] + self.x0l[0][0]) * x + self.x0l[0][1] * x02
            else:
                x = torch.add(self.x0l[i][0] * x0 + self.x0l[i][1] * x02, x, alpha=self.lam[i])
            h = norm(x)
            qkv = self.linear(h, f"{i}.qkv").view(-1, 3 * heads, hd)
            qk, v = rotary(norm(qkv[:, : 2 * heads])), qkv[:, 2 * heads :]
            g = torch.sigmoid(F.linear(h[:, :16], w[f"{i}.gates"]))  # output, value
            if ve[i] is not None:
                v = v + 2 * g[:, heads:, None] * ve[i].view_as(v)
            y = attend(i, qk[:, :heads], qk[:, heads:], v) * g[:, :heads, None]
            x = x + self.linear(y.reshape(-1, self.width), f"{i}.o")
            x = x + self.mlp(i, norm(x))
            if i in skip_in:
                skips.append(x)
            if i == skip_out[-1]:
                backout = x
        if last is not None:
            x, backout = x[last], backout[last]
        x = norm(x - self.backout * backout)
        return 23 * torch.sigmoid((F.linear(x, w["lm_head"]).float() + 5) / 7.5)


class Cache:
    """One game's keys, values and token embeddings, grown on demand."""

    def __init__(self, model, capacity=128):
        self.model, self.n, self.capacity = model, 0, 0
        self.k = self.v = self.e = self.slot = None
        if not (model.graphs and model.graphs.attach(self)):  # a slot of the GPU pool, or our own
            self.reserve(capacity)

    def reserve(self, n):
        if n <= self.capacity:
            return
        assert n <= CONTEXT, n
        m = self.model
        cap = min(max(n, 2 * self.capacity), CONTEXT)
        kw = dict(dtype=m.dtype, device=m.device)
        k = torch.empty(m.layers, m.heads, cap, m.head_dim, **kw)
        v, e = torch.empty_like(k), torch.empty(cap, m.width, **kw)
        n = self.n
        if n:
            k[:, :, :n], v[:, :, :n], e[:n] = self.k[:, :, :n], self.v[:, :, :n], self.e[:n]
        self.k, self.v, self.e, self.capacity = k, v, e, cap

    def truncate(self, n):
        self.n = min(self.n, n)


@torch.inference_mode()
def step(model, items):
    """Append tokens to caches in one batched forward. items: (cache, ids, feats, boards) with
    ids [m] int64, feats [m, 3] and boards [m, 68] uint8 tensors on the model's device. Returns
    the logits [len(items), 2432] FP32 at each item's last new token."""
    if model.fast is not None:
        return model.fast.step(items)
    if model.graphs is not None and (z := model.graphs.step(items)) is not None:
        return z
    spans, lo = [], 0
    for cache, ids, *_ in items:
        cache.reserve(cache.n + len(ids))
        spans.append((cache, lo, lo + len(ids), cache.n))
        lo += len(ids)
    dev = model.device
    pos = torch.cat([torch.arange(n0, n0 + b - a) for _, a, b, n0 in spans]).to(dev)

    def previous(e):
        p = torch.empty_like(e)
        for c, a, b, n0 in spans:
            p[a] = c.e[n0 - 1] if n0 else 0
            p[a + 1 : b] = e[a : b - 1]
            c.e[n0 : n0 + b - a] = e[a:b]
        return p

    def attend(i, q, k, v):
        y = torch.empty_like(q)
        for c, a, b, n0 in spans:
            n1 = n0 + b - a
            c.k[i, :, n0:n1] = k[a:b].transpose(0, 1)
            c.v[i, :, n0:n1] = v[a:b].transpose(0, 1)
            mask = None
            if b - a > 1:
                mask = (
                    torch.arange(n1, device=dev)
                    <= torch.arange(n0, n1, device=dev)[:, None]
                )
            y[a:b] = F.scaled_dot_product_attention(
                q[a:b].transpose(0, 1),
                c.k[i, :, :n1],
                c.v[i, :, :n1],
                attn_mask=mask,
                scale=model.scale,
            ).transpose(0, 1)
        return y

    cat = lambda j: torch.cat([it[j] for it in items]).to(dev)
    last = torch.tensor([b - 1 for _, _, b, _ in spans], device=dev)
    logits = model.forward(cat(1), pos, cat(2), cat(3), previous, attend, last)
    for c, a, b, n0 in spans:
        c.n = n0 + b - a
    return logits
