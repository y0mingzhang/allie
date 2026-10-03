"""allie.lichess's model against the reference scores of Allie-v3.0 on the Maia-3 blitz benchmark.

The reference is allie.eval.maia3.score_moe on the training forward (compiled, GPU, BF16): the
per-position legal CE and top-1 in <maia3-bench>/bigrun-v2/step-00143051/scores.npz. This runs
the bot's own code path (fresh Cache per position, model.step) on a stratified subsample and
reports the paired CE difference (95% game bootstrap), top-1, and the per-position change in the
played move's probability. --save keeps the legal-move distributions so two runs (CPU vs GPU,
BF16 vs FP32, fewer experts) can be compared move by move with --against.

usage: python analysis/lichess/parity.py --model DIR [--device cpu] [--per-band 1250] --out X.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from allie import paths
from allie.lichess.model import Cache, Model, step
from allie.lichess.tokens import HEADER, START, advance

G = paths.DATA / "strat-eval-v1"
BENCH = paths.DATA / "maia3-bench"
REFERENCE = BENCH / "bigrun-v2/step-00143051/scores.npz"


def positions(index):
    with np.load(G / "strat.npz") as z:
        rows = z["rows"].astype(np.int64)
    with np.load(G / "feats.npz") as z:
        feats = z["feats"]
    with np.load(BENCH / "legal.npz") as z:
        pos, target, legal, offsets = z["pos"], z["target"], z["legal"], z["offsets"]
    with np.load(BENCH / "games.npz") as z:
        sel = z["sel"][z["keep"]]
    out = []
    for i in index:
        (r, c), (g, ply, cell, *_) = pos[i], sel[i]
        bos = np.flatnonzero(rows[r, : c + 1] == 2348)[-1]
        assert c - bos == HEADER - 1 + ply
        out.append(dict(
            prefix=rows[r, bos : c + 1], features=feats[r, bos : c + 1].astype(np.float32),
            legal=legal[offsets[i] : offsets[i + 1]].astype(np.int64), target=int(target[i]), game=int(g),
            cell=int(cell),
        ))  # fmt: skip
    return out


def subsample(per_band, seed=20260924):
    with np.load(BENCH / "games.npz") as z:
        cell = z["sel"][z["keep"]][:, 2]
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate(
        [rng.choice(np.flatnonzero(cell == c), per_band, replace=False) for c in (4, 5, 6, 7)]
    ))  # fmt: skip


def boards(prefix):
    b = [START] * HEADER
    for t in prefix[HEADER:]:
        b.append(advance(b[-1], int(t)))
    return torch.tensor(np.frombuffer(b"".join(b), np.uint8).reshape(-1, 68))


def bootstrap(game, d, rng, n=2000):
    u, inv = np.unique(game, return_inverse=True)
    s, c = np.bincount(inv, d), np.bincount(inv)
    draws = rng.integers(0, len(u), (n, len(u)))
    m = s[draws].sum(1) / c[draws].sum(1)
    return [float(d.mean()), *np.percentile(m, [2.5, 97.5]).tolist()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--experts", type=int)
    p.add_argument("--int8", action="store_true")
    p.add_argument("--threads", type=int)
    p.add_argument("--per-band", type=int, default=1250)
    p.add_argument("--tokens", type=int, default=2048, help="tokens per forward")
    p.add_argument("--out", required=True)
    p.add_argument("--save", help="npz of the legal-move probabilities")
    p.add_argument("--against", help="another run's --save npz to compare with")
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.perf_counter()
    m = Model(a.model, a.device, getattr(torch, a.dtype), a.experts, a.int8)
    load = time.perf_counter() - t0
    index = subsample(a.per_band)
    P = positions(index)
    width = max(len(x["legal"]) for x in P)
    probs = np.zeros((len(P), width), np.float32)
    ce, top1 = np.zeros(len(P)), np.zeros(len(P), bool)
    t0, lo = time.perf_counter(), 0
    while lo < len(P):
        hi, total = lo, 0
        while hi < len(P) and (hi == lo or total + len(P[hi]["prefix"]) <= a.tokens):
            total += len(P[hi]["prefix"])
            hi += 1
        items = [(Cache(m, len(x["prefix"])), torch.as_tensor(x["prefix"]),
                  torch.as_tensor(x["features"]), boards(x["prefix"])) for x in P[lo:hi]]  # fmt: skip
        z = step(m, items).double().cpu()
        for i, x in zip(range(lo, hi), P[lo:hi]):
            lg = torch.log_softmax(z[i - lo, 378:2346][torch.as_tensor(x["legal"])], 0)
            t = int(np.flatnonzero(x["legal"] == x["target"])[0])
            ce[i], top1[i] = -lg[t].item(), lg.argmax().item() == t
            probs[i, : len(lg)] = lg.exp().numpy()
        lo = hi
    seconds = time.perf_counter() - t0
    with np.load(REFERENCE) as z:
        ref_ce, ref_top1 = z["ce_legal"][index], z["top1"][index]
    game = np.array([x["game"] for x in P])
    rng = np.random.default_rng(0)
    d = ce - ref_ce
    dp = np.exp(-ce) - np.exp(-ref_ce)
    out = dict(
        model=a.model, device=a.device, dtype=a.dtype, int8=a.int8, experts=a.experts or m.topk,
        threads=torch.get_num_threads(), positions=len(P), load_seconds=round(load, 1),
        seconds=round(seconds, 1), ms_per_position=round(1000 * seconds / len(P), 1),
        ce=float(ce.mean()), reference_ce=float(ref_ce.mean()),
        ce_minus_reference=bootstrap(game, d, rng),
        top1=float(top1.mean()), reference_top1=float(ref_top1.mean()),
        top1_correct_agreement=float((top1 == ref_top1).mean()),
        played_move_probability_change=dict(mean=float(dp.mean()), p99_abs=float(
            np.percentile(abs(dp), 99)), max_abs=float(abs(dp).max())),
        bands={str(c): float(d[[x["cell"] == c for x in P]].mean()) for c in (4, 5, 6, 7)},
    )  # fmt: skip
    if a.against:
        with np.load(a.against) as z:
            q = z["probs"]
        assert q.shape == probs.shape
        diff = np.abs(probs - q).max(1)
        kl = (
            q * (np.log(np.maximum(q, 1e-30)) - np.log(np.maximum(probs, 1e-30)))
        ).sum(1)
        out["against"] = dict(
            file=a.against, max_abs_prob_diff=float(diff.max()),
            p99_abs_prob_diff=float(np.percentile(diff, 99)),
            mean_abs_prob_diff=float(diff.mean()), kl_mean=float(kl.mean()),
            kl_max=float(kl.max()), argmax_agreement=float((probs.argmax(1) == q.argmax(1)).mean()),
        )  # fmt: skip
    if a.save:
        np.savez(a.save, probs=probs, index=index, ce=ce)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
