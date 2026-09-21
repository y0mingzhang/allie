"""Tile tables for one MoE shape: run one eager layer fwd+bwd, record every expert-kernel call with
its actual tensors, then sweep tile configs per call, keeping only configs whose output equals the
current one bitwise.

    tune_moe_shape.py --out t.json    # the candidate: d1536 SwiGLU, E96 top-4, 64K tokens per GPU

One JSON row per (call, table); --out keeps the rows and, per call, the fastest exact config at
>= --min-gain over the current ("tables", read by bench_moe_layer.py), printed at the end as paste
snippets: an aligned config() block (modded_smoe_aligned_linear), WEIGHT entries (modded_smoe_tiles,
read by the gather and gated weight grads), SCATTER entries, and the gated input grad's explicit
config (it has no table)."""

import argparse
import functools
import itertools
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import modded_moe
import modded_smoe_aligned as aligned
import modded_smoe_gated as gd
import modded_smoe_gather_wgrad as gw
import modded_smoe_tuned as tuned
import torch
import triton.testing
from modded_arch import moe_dims
from modded_moe import MoE
from modded_smoe_tiles import SCATTER, WEIGHT
from triton.errors import TritonError

GEMM = list(itertools.product((64, 128), (64, 128, 256), (32, 64, 128), (4, 8), (3, 4)))
WGRAD = list(
    itertools.product((32, 64, 128), (64, 128, 256), (64, 128, 256), (4, 8), (2, 3, 4))
)
# moe64k8's (moe-v1-round1) router init and sequence-wise balance loss
ARCH = {"mlp": "swiglu", "moe_init": 0.006, "moe_seq": 0.001}
ALIGNED = aligned.linear
calls = []


def shape_args(p):
    p.add_argument("--width", type=int, default=1536)
    p.add_argument("--experts", type=int, default=96)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument(
        "--tokens", type=int, default=65536, help="per GPU (64 rows of 1024)"
    )
    p.add_argument("--kernel", default="scatter-dualgather")
    p.add_argument("--seed", type=int, default=0)


def layer(a, device):
    """The layer (BF16 weights, FP32 router, moe_dims widths), an input and an output gradient. Down
    weights are nonzero, or every up-projection gradient is zero and its exactness check vacuous."""
    torch.manual_seed(a.seed)
    arch = ARCH | {"moe": [a.experts, a.topk], "moe_kernel": a.kernel}
    m = MoE(a.width, *moe_dims(a.width, arch)).to(device).train()
    with torch.no_grad():
        m.down.normal_(0, 0.02)
        m.shared_down.normal_(0, 0.02)
    for n, p in m.named_parameters():
        if n != "router":
            p.data = p.data.bfloat16()
    x = torch.randn(a.tokens, a.width, device=device, dtype=torch.bfloat16)
    return m, x.requires_grad_(), torch.randn_like(x)


def bench(fn):
    if not torch.cuda.is_available():  # TRITON_INTERPRET smoke run: exactness only
        t = time.perf_counter()
        fn()
        return 1e3 * (time.perf_counter() - t)
    return triton.testing.do_bench_cudagraph(fn, rep=100)


def record(name, fn):
    def keep(v):
        return v.detach().clone() if torch.is_tensor(v) else v

    def wrapped(*a, **kw):
        out = fn(*a, **kw)
        kept = {k: keep(v) for k, v in kw.items() if k != "out"}
        calls.append((name, fn, [keep(v) for v in a], kept))
        return out

    return wrapped


def override(table, key, fn):
    """run(config): fn() with table[key] = config (None: the current selection)."""

    def run(c):
        old = table.get(key)
        if c is not None:
            table[key] = c
        try:
            return fn()
        finally:
            table.pop(key, None)
            if old is not None:
                table[key] = old

    return run


def runners(name, fn, a, kw):
    """(table, key, grid, run(config)) for one recorded call."""
    call = functools.partial(fn, *a, **kw)
    explicit = lambda c: fn(*a, **(kw | ({"config": c} if c else {})))
    match name:
        case "scatter":
            x, w, _, order, k = a[:5]
            xg, yg = kw.get("x_grouped", False), kw.get("y_grouped", False)
            key = (w.shape[0], order.numel(), x.shape[1], w.shape[-1], k, xg, yg)
            return "SCATTER", key, GEMM, override(SCATTER, key, call)
        case "wgrad":
            dy, x, _, e = a[:4]
            key = (e, dy.shape[0], x.shape[1], dy.shape[1])
            return "WEIGHT", key, WGRAD, override(WEIGHT, key, lambda: call()[0])
        case "aligned":
            x, w, order, _, k, xg, yg = a[:7]
            key = (order.numel(), x.shape[1], w.shape[-1], k, xg, yg)
            return "ALIGNED", key, GEMM, lambda c: fn(*a[:7], c or a[7])
        case "gather_wgrad":  # up wgrad in scatter-gather/-dualgather
            dy, x, _, offsets = a[:4]
            key = (offsets.numel(), dy.shape[0], x.shape[1], dy.shape[1])
            return "WEIGHT", key, WGRAD, explicit
        case "gated_wgrad":  # down wgrad in -dualgather
            dy, x, _, order, offsets = a[:5]
            key = (offsets.numel(), order.numel(), x.shape[-1], dy.shape[-1])
            return "WEIGHT", key, WGRAD, explicit
        case "gated_dx":  # down dgrad in -dualgather
            dy, w, _, order, offsets = a[:5]
            key = (offsets.numel(), order.numel(), dy.shape[-1], w.shape[-1])
            return "GATED_DX", key, GEMM, explicit


def aligned_runner(a, kw):
    """A scatter call through aligned.linear (expert-aligned tiles), for shapes config() lacks."""
    x, w, se, order, k = a[:5]
    offsets = modded_moe.counts(se, w.shape[0]).cumsum(0)
    xg, yg = kw.get("x_grouped", False), kw.get("y_grouped", False)
    key = (order.numel(), x.shape[1], w.shape[-1], k, xg, yg)
    return "ALIGNED", key, GEMM, lambda c: ALIGNED(x, w, order, offsets, k, xg, yg, c)


def op(m, table, key):
    """The expert GEMM a (table, key) row tunes, from its (K, N)."""
    d, u, h = m.up.shape[2], m.up.shape[1], m.down.shape[1]
    names = {}
    for kn, n in zip(
        ((d, u), (h, d), (u, d), (d, h)), ("up", "down", "up-dgrad", "down-dgrad")
    ):
        names[kn] = f"{names[kn]}/{n}" if kn in names else n
    kn = tuple(key[-2:] if table in ("WEIGHT", "GATED_DX") else key[-5:-3])
    return names[kn] + ("-wgrad" if table == "WEIGHT" else "")


def winners(rows):
    """{table: [[key, config], ...]}: per call, its fastest adopted row."""
    best = {
        r["call"]: r for r in sorted(rows, key=lambda r: -r["best_ms"]) if r["adopt"]
    }
    tables = {}
    for r in best.values():
        tables.setdefault(r["table"], []).append([r["key"], r["best"]])
    return tables


def snippet(tables, a):
    tag = f"d{a.width} E{a.experts} top-{a.topk} at {a.tokens // 1024}K tokens (tune_moe_shape)"
    tight = lambda t: str(tuple(t)).replace(" ", "")
    out = []
    if rows := tables.get("ALIGNED"):
        if a.experts not in (96, 128):
            out.append(
                f"# config() returns None for E{a.experts}: widen its expert check"
            )
        out += [
            "# modded_smoe_aligned_linear.config()",
            f"    if rows=={a.tokens * a.topk}:  # {tag}",
        ]
        out += [
            "        return {",
            *(f"            {tight(k[1:])}:{tight(c)}," for k, c in rows),
        ]
        out.append("        }.get(key)")
    for table in ("WEIGHT", "SCATTER"):
        if rows := tables.get(table):
            out += [f"# modded_smoe_tiles.{table}: {tag}"]
            out += [f"    {tuple(k)}: {tuple(c)}," for k, c in rows]
    for k, c in tables.get("GATED_DX", []):
        out.append(
            f"# modded_smoe_gated.input_grad (no table): config={tuple(c)} at {tuple(k)}"
        )
    return "\n".join(out) or "# nothing adopted: the current tiles stay"


def main():
    p = argparse.ArgumentParser()
    shape_args(p)
    p.add_argument("--min-gain", type=float, default=1.03)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.use_deterministic_algorithms(True)
    tuned.scatter2scatter = record("scatter", tuned.scatter2scatter)
    tuned.group_bwd_W = record("wgrad", tuned.group_bwd_W)
    aligned.linear = record("aligned", aligned.linear)
    gw.backward = record("gather_wgrad", gw.backward)
    gd.weight_grad = record("gated_wgrad", gd.weight_grad)
    gd.input_grad = record("gated_dx", gd.input_grad)
    m, x, dy = layer(a, "cuda" if torch.cuda.is_available() else "cpu")
    m(x).backward(dy)
    jobs, seen = [], set()
    for i, (name, fn, args, kw) in enumerate(calls):
        own = runners(name, fn, args, kw)
        alt = [aligned_runner(args, kw)] if name == "scatter" else []
        for table, key, grid, run in [own, *alt]:
            if (table, key) not in seen:
                seen.add((table, key))
                jobs.append((i, table, key, grid, run, own[3]))
    print("calls:", sorted({op(m, t, k) for _, t, k, *_ in jobs}), flush=True)
    rows = []
    for i, table, key, grid, run, current in jobs:
        t0 = time.monotonic()
        ref = current(None).clone()
        assert ref.any(), f"{table} {key}: zero reference, exactness would be vacuous"
        base = bench(functools.partial(current, None))
        best, exact = (None, base), 0
        for c in grid:
            try:
                if not torch.equal(run(c), ref):
                    continue
            except (TritonError, RuntimeError, AssertionError):
                continue
            exact += 1
            ms = bench(functools.partial(run, c))
            if ms < best[1]:
                best = (c, ms)
        assert torch.equal(current(None), ref), (
            "the current path changed during the sweep"
        )
        gain = base / best[1]
        rows.append({
            "call": i, "op": op(m, table, key), "table": table, "key": list(key),
            "current_ms": base, "best": best[0], "best_ms": best[1], "gain": gain,
            "exact": exact, "adopt": bool(best[0]) and gain >= a.min_gain,
            "seconds": time.monotonic() - t0,
        })  # fmt: skip
        print(json.dumps(rows[-1]), flush=True)
        out = {"args": vars(a), "rows": rows, "tables": winners(rows)}
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(snippet(winners(rows), a))


if __name__ == "__main__":
    main()
