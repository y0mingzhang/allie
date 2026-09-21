"""A/B of tile tables on one MoE layer: forward, checkpoint recompute and backward as --ckpt eager
runs them (compiled fullgraph block under torch.utils.checkpoint, deterministic), with the current
tables (old) vs the current plus tune_moe_shape's winners (new). Output, input gradient and every
parameter gradient must match bitwise; each variant is compiled apart and captured as one CUDA
graph, whose replays are timed alternately (CPU: the exactness check only).

    bench_moe_layer.py --tables t.json [--out ab.json]    # no --tables: an A/A noise check
"""

import argparse
import contextlib
import json
import statistics
from pathlib import Path

import modded_moe
import modded_smoe_aligned_linear as al
import modded_smoe_gated as gd
import torch
from modded_smoe_tiles import SCATTER, WEIGHT
from torch.utils.checkpoint import checkpoint
from tune_moe_shape import layer, shape_args


@contextlib.contextmanager
def tables(t):
    """The current tables plus t ({table: [[key, config], ...]}) until exit."""
    t = {name: {tuple(k): tuple(c) for k, c in rows} for name, rows in t.items()}
    saved = al.config, gd.input_grad, dict(SCATTER), dict(WEIGHT)

    def config(x, w, order, k, xg, yg):
        key = (order.numel(), x.shape[1], w.shape[-1], k, xg, yg)
        c = t.get("ALIGNED", {}).get(key) if x.dtype == torch.bfloat16 else None
        return c or saved[0](x, w, order, k, xg, yg)

    def input_grad(dy, w, gates, order, offsets, config=None, out=None):
        key = (offsets.numel(), order.numel(), dy.shape[-1], w.shape[-1])
        c = config or t.get("GATED_DX", {}).get(key)
        return saved[1](
            dy, w, gates, order, offsets, **({"config": c} if c else {}), out=out
        )

    SCATTER.update(t.get("SCATTER", {}))
    WEIGHT.update(t.get("WEIGHT", {}))
    al.config, gd.input_grad = config, input_grad
    try:
        yield
    finally:
        al.config, gd.input_grad = saved[:2]
        for table, old in zip((SCATTER, WEIGHT), saved[2:], strict=True):
            table.clear()
            table.update(old)


def variant(a, t, device):
    """Under tables t: one step's output and gradients by name, and (CUDA) the step captured as a
    graph, which holds what it replays on."""
    torch._dynamo.reset()  # nothing traced under the other variant's tables is reused
    with tables(t):
        m, x, dy = layer(a, device)
        f = torch.compile(m, fullgraph=True, dynamic=False)

        def step():
            y = checkpoint(f, x, use_reentrant=False, preserve_rng_state=False)
            y.backward(dy)
            return y

        got = {"out": step().detach().clone(), "x": x.grad.clone()}
        got |= {n: p.grad.clone() for n, p in m.named_parameters()}
        if device == "cpu":
            return got, None
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):  # warm up outside the capture
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        g.keep = m, f, x, dy
    return got, g


def replay_ms(g, n):
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(n):
        g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / n


def main():
    p = argparse.ArgumentParser()
    shape_args(p)
    p.add_argument(
        "--tables", help="tune_moe_shape --out JSON; its winners are the new tables"
    )
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--out")
    a = p.parse_args()
    new = json.loads(Path(a.tables).read_text())["tables"] if a.tables else {}
    rows = a.tokens * a.topk
    for name, entries in new.items():
        for k, _ in entries:  # keys for another shape would silently run an A/A
            assert k[0 if name == "ALIGNED" else 1] == rows, (
                f"{name} {k}: not {rows} rows"
            )
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    modded_moe.BLOCK_RECOMPUTE = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (old, g_old), (got, g_new) = variant(a, {}, device), variant(a, new, device)
    row = {"args": vars(a), "tables": new}
    row["mismatch"] = [n for n in old if not torch.equal(old[n], got[n])]
    if g_old is not None:
        pair, trials = (("old", g_old), ("new", g_new)), []
        for i in range(a.trials):
            trials.append(
                {n: replay_ms(g, a.replays) for n, g in pair[:: -1 if i % 2 else 1]}
            )
        row["trials"] = trials
        row["speedup"] = statistics.median(t["old"] / t["new"] for t in trials)
    print(json.dumps(row), flush=True)
    if a.out:
        with open(a.out, "x") as f:
            f.write(json.dumps(row, indent=1) + "\n")
    raise SystemExit(1 if row["mismatch"] else 0)


if __name__ == "__main__":
    main()
