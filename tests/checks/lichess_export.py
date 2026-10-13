"""export.convert of a small random model with Allie 2.2's arch (log gates, three dense blocks, router centres on the
first four MoE layers, XSA): the exported reference (allie.lichess.model, BF16 on the GPU, and its CUDA graphs) and
the GPU search oracle (allie.search.moe_oracle.forward) against the compiled training forward, game by game, within
BF16 rounding (3x the reference's gap to itself in FP32, in mean logit and CE differences); the reference without XSA
is outside it.

    torchrun --standalone --nproc_per_node=1 tests/checks/lichess_export.py
"""

import json
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

import chess
import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.nn import functional as F

from allie.lichess import export
from allie.lichess.model import Cache, Model, step
from allie.lichess.tokens import BOS, HEADER, MOVE_ID, START, advance, features, header
from allie.model.network import Config, core, create_model, make_context
from allie.search import moe_oracle

ARCH = dict(board="conv", mlp="swiglu", key_offset=False, moe=[256, 16], moe_seq=0.001, moe_init=0.006,
            moe_router_lr_mul=0.05, moe_gamma=0.01, moe_update="quantile", moe_kernel="scatter-dualgather",
            moe_shared_frac=0.25, moe_round=16, fp32_small_masters=True, moe_log_gates=True, moe_dense_first=3,
            moe_router_center=0.9, moe_router_center_first=4, xsa=True, moe_shard=True)  # fmt: skip
ROW, ROWS = 1024, 2


def game(seed):
    """A random game's tokens, clock features and board states."""
    rng, board = random.Random(seed), chess.Board()
    while not board.is_game_over() and len(board.move_stack) < 40 + seed % 50:
        board.push(rng.choice(list(board.legal_moves)))
    moves = [m.uci() for m in board.move_stack]
    base, inc = rng.choice([(60, 0), (180, 2), (600, 5)])
    tokens = header(base, inc, rng.randrange(800, 2800), rng.randrange(800, 2800)) + [
        MOVE_ID[m] for m in moves
    ]
    clocks = [base - k // 2 for k in range(len(moves))]
    feats = [[-1] * 3] * (HEADER - 1) + [
        features(k, base, inc, clocks) for k in range(len(moves) + 1)
    ]
    boards = [START] * HEADER
    for t in tokens[HEADER:]:
        boards.append(advance(boards[-1], t))
    boards = np.frombuffer(b"".join(boards), np.uint8).reshape(-1, 68)
    return (
        torch.tensor(tokens),
        torch.tensor(feats, dtype=torch.float32),
        torch.tensor(boards),
    )


@torch.no_grad()
def randomize(m, g):
    """Every weight random (the init zeroes many) at scales that keep activations O(1),
    the board CNN's as built."""
    r = lambda p, std: (
        torch.randn(p.shape, generator=g, device=p.device).to(p.dtype) * std
    )
    for name, p in m.named_parameters():
        if name == "xsa_alphas":
            p.copy_(r(p, 2))
        elif p.ndim < 2:
            p.add_(r(p, 0.1))
        elif not name.startswith("board."):
            embed = name in ("embed.weight", "embed2.weight") or "value_embeds" in name
            p.copy_(r(p, 1 if embed else p.shape[-1] ** -0.5))
    for name, b in m.named_buffers():
        if name.endswith("mlp.bias"):
            b.copy_(r(b, 0.1))
        elif name.endswith("mlp.mu"):
            b.copy_(r(b, 0.5))


def oracle(m, ids, feats, boards):
    """moe_oracle.forward on one game, prefilled at once."""
    scale, n = m.yarn.attn_scale, len(ids)

    def previous(e):
        return torch.cat((e.new_zeros(1, e.shape[1]), e[:-1]))

    def attend(i, q, k, v):
        q, k, v = (x.transpose(0, 1) for x in (q, k, v))
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scale
        ).transpose(0, 1)

    pos = torch.arange(n, device="cuda")
    return moe_oracle.forward(m, ids, pos, feats, boards, previous, attend)[0]


def ce(z, ids):
    """Each scored move's CE over the move ids, from the logits before it."""
    s = z[HEADER - 1 : -1, 378:2346].double()
    return s.logsumexp(-1) - s.gather(1, (ids[HEADER:, None] - 378)).squeeze(1)


def main():
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    cfg = Config(
        width=256,
        head_dim=64,
        layers=8,
        max_tokens=ROW * ROWS,
        scheduled_steps=8,
        feats=3,
        arch=ARCH,
    )
    m = create_model(cfg)
    randomize(m, torch.Generator(device="cuda").manual_seed(0))
    m.split_embed = True
    m.eval()

    # whole games packed into rows, the rest of a row one-token BOS games
    games, rows, spans, seed = [], [], [], 0
    for r in range(ROWS):
        ids, feats, boards, lo = [], [], [], 0
        while True:
            x = game(seed)
            if lo + len(x[0]) > ROW:
                break
            games.append(x)
            spans.append(r * ROW + lo)
            for a, b in zip((ids, feats, boards), x):
                a.append(b)
            lo, seed = lo + len(x[0]), seed + 1
        ids.append(torch.full((ROW - lo,), BOS))
        feats.append(torch.full((ROW - lo, 3), -1.0))
        rows.append((torch.cat(ids), torch.cat(feats)))
    x = torch.stack([r[0] for r in rows]).cuda()
    f = torch.cat([r[1] for r in rows]).cuda()
    net = torch.compile(m, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        ctx = make_context(x, 11 * 128, 23 * 128)
        train = net(
            x.flatten(), ctx, core.ForwardScheduleConfig(None, 11, 23), feat_seq=f
        )[0].float()

    inference = dict(split_embed=True, ws_short=11, ws_long=23,
                     yarn=dict(cos=m.yarn.cos.cpu(), sin=m.yarn.sin.cpu(), attn_scale=m.yarn.attn_scale))  # fmt: skip
    state = dict(
        config=asdict(cfg),
        model={k: v.detach().cpu() for k, v in m.state_dict().items()},
        inference=inference,
    )
    w, config = export.convert(state)
    assert (
        config["log_gates"]
        and "xsa" in w
        and all(f"{i}.fc" in w for i in range(3))
        and "3.up" in w
    )
    assert all(w[f"{i}.mu"].any() == (i < 7) for i in range(3, 8))
    with tempfile.TemporaryDirectory() as tmp:
        save_file(w, str(Path(tmp) / "model.safetensors"))
        (Path(tmp) / "config.json").write_text(json.dumps(config))
        refs = {
            dt: Model(tmp, "cuda", dt, backend="torch")
            for dt in (torch.bfloat16, torch.float32)
        }
        fast = Model(tmp, "cuda", torch.bfloat16)

    dev = [tuple(t.cuda() for t in g) for g in games]

    def run(model):
        return [step(model, [(Cache(model), *g)], every=True).float() for g in dev]

    def decode(model):
        """Every game a token at a time, one step for all: replayed CUDA graphs."""
        caches, out = [Cache(model) for _ in games], [[] for _ in games]
        for j in range(max(len(g[0]) for g in games)):
            live = [k for k, g in enumerate(games) if j < len(g[0])]
            items = [(caches[k], *(t[j : j + 1] for t in dev[k])) for k in live]
            for k, z in zip(live, step(model, items)):
                out[k].append(z.float())
        return [torch.stack(z) for z in out]

    def oracle_all():
        with torch.no_grad():
            return [oracle(m, *g).float() for g in dev]

    want = [train[a : a + len(g[0])] for a, g in zip(spans, games)]
    ref, ref32, orc = run(refs[torch.bfloat16]), run(refs[torch.float32]), oracle_all()
    graphs = decode(fast)
    assert fast.graphs.graphs and not fast.graphs.broken
    refs[torch.bfloat16].xsa = None
    plain = run(refs[torch.bfloat16])

    def gaps(a, b):
        d = torch.cat([(x - y).abs().flatten() for x, y in zip(a, b)])
        c = torch.cat([ce(x, g[0]) - ce(y, g[0]) for x, y, g in zip(a, b, dev)])
        return dict(mean_logit=d.mean().item(), max_logit=d.max().item(), mean_ce=c.mean().item(),
                    mean_abs_ce=c.abs().mean().item())  # fmt: skip

    ces = {k: torch.cat([ce(z, g[0]) for z, g in zip(v, dev)]).mean().item()
           for k, v in dict(train=want, ref=ref, ref32=ref32, graphs=graphs, oracle=orc, no_xsa=plain).items()}  # fmt: skip
    print(
        f"{len(games)} games, {sum(len(g[0]) - HEADER for g in games)} moves; CE {ces}"
    )
    floor = gaps(ref32, ref)
    out = dict(ref=gaps(want, ref), graphs=gaps(want, graphs), oracle=gaps(want, orc), ref32=gaps(want, ref32),
               floor=floor, no_xsa=gaps(want, plain))  # fmt: skip
    for k, v in out.items():
        print(k, {a: round(b, 5) for a, b in v.items()})
    for k in ("ref", "graphs", "oracle", "ref32"):
        assert (
            out[k]["mean_logit"] < 3 * floor["mean_logit"]
            and out[k]["mean_abs_ce"] < 3 * floor["mean_abs_ce"]
        ), k
    # the bound catches a missing XSA
    assert out["no_xsa"]["mean_logit"] > 3 * floor["mean_logit"]
    dist.destroy_process_group()
    print("PASS lichess_export", flush=True)


if __name__ == "__main__":
    main()
