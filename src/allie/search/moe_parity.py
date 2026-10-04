"""Parity of MoEOracle with the training forward on golden-eval rows, in separable checks (TF32 on, as in training).

A. Restatement: oracle.forward() driven by the training's own flex attention, row positions,
   boards and smear mask on packed golden rows, against the eager training forward (bitwise?).
B. Tree cache: nodes reached one token at a time (16-move chains batched across games, plus a
   sibling branch per root) against a fresh prefill of the same prefixes.
C. Matched positions: oracle prefill + chains with each game's original row offset as rotary
   origin, against the compiled training forward (what the evaluators run). Only the attention kernel (SDPA over
   gathered keys vs flex) and compiler fusion differ.
D. Search positions: the same with in-game rotary positions (origin 0), as search runs it. For
   scale: the compiled forward with every game packed SHIFT tokens later, and the eager forward.
Each comparison gives the raw logits, KL(ref || other) over the move softmax, the signed mean
difference of the played move's log-probability, and the W/D/L probabilities and predicted
think seconds that steer the tree.

usage: python -m allie.search.moe_parity --checkpoint CKPT [--out report.json]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

from allie import paths
from allie.search.board import encode, predicted_seconds
from allie.search.moe_oracle import MoEOracle, forward, network
from allie.search.native import from_prefix

G = paths.DATA / "strat-eval-v1"
SHIFT = 37


def compare(ref, got, target=None):
    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    lr = ref[:, 378:2346] - np.logaddexp.reduce(ref[:, 378:2346], axis=1, keepdims=True)
    lg = got[:, 378:2346] - np.logaddexp.reduce(got[:, 378:2346], axis=1, keepdims=True)
    kl = (np.exp(lr) * (lr - lg)).sum(1)
    out = dict(
        n=len(ref),
        logit_max=float(np.abs(ref - got).max()),
        kl_mean=float(kl.mean()),
        kl_p99=float(np.quantile(kl, 0.99)),
        kl_max=float(kl.max()),
        top1=float((lr.argmax(1) == lg.argmax(1)).mean()),
        wdl_prob_max=float(
            np.abs(softmax(ref[:, 2413:2416], 1) - softmax(got[:, 2413:2416], 1)).max()
        ),
        seconds_max=float(
            np.abs(predicted_seconds(ref) - predicted_seconds(got)).max()
        ),
        seconds_mean=float(
            np.abs(predicted_seconds(ref) - predicted_seconds(got)).mean()
        ),
    )
    if target is not None:
        t = np.asarray(target)
        ok = (t >= 378) & (t < 2346)
        d = (lg - lr)[np.flatnonzero(ok), t[ok] - 378]
        out.update(
            target_logp_diff=float(d.mean()),
            target_logp_se=float(d.std() / len(d) ** 0.5),
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--chain", type=int, default=16)
    p.add_argument("--out")
    p.add_argument("--source", help="a pre-package checkpoint's frozen flat source")
    p.add_argument("--slots", type=int, default=1 << 18)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    oracle = MoEOracle(a.checkpoint, a.source, slots=a.slots, rows=min(a.slots, 1 << 17))
    m = oracle.model
    mm = network(torch.load(oracle.checkpoint, map_location="cpu", weights_only=False, mmap=True), a.source)

    with np.load(G / "strat.npz") as z:
        rows = z["rows"][: a.rows].astype(np.int64)
    with np.load(G / "feats.npz") as z:
        feats = z["feats"][: a.rows, :-1].astype(np.int64)
    inf = torch.load(
        oracle.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )["inference"]
    ws = (inf["ws_short"] * 128, inf["ws_long"] * 128)
    schedule = mm.core.ForwardScheduleConfig(None, inf["ws_short"], inf["ws_long"])
    net = torch.compile(m, dynamic=False, fullgraph=True)

    @torch.no_grad()
    def train_forward(model, rows, feats):
        x = torch.as_tensor(rows[:, :-1], device="cuda")
        f = torch.as_tensor(feats, device="cuda").flatten(0, 1)
        context = mm.make_context(x, *ws)
        z = model(x.flatten(), x.flatten(), context, schedule, feat_seq=f)
        return z.float().view(*x.shape, -1).cpu().numpy(), x, f, context

    eager, x, f, context = train_forward(m, rows, feats)
    ref, *_ = train_forward(net, rows, feats)
    shifted = np.concatenate((np.full((len(rows), SHIFT), 2348), rows[:, :-SHIFT]), 1)
    sfeats = np.concatenate((np.full((len(rows), SHIFT, 3), -1), feats[:, :-SHIFT]), 1)
    ref2, *_ = train_forward(net, shifted, sfeats)

    # A
    n = m.num_layers
    long = {round(i * (n - 1) / 15) for i in (0, 4, 11, 15)}
    same = context.same_previous[:, None]

    def attend(i, q, k, v):
        return mm.attention(
            q[None], k[None], v[None], context, ws[i in long], m.yarn.attn_scale
        )[0]

    def previous(e):
        return torch.cat((e[:1] * 0, e[:-1])) * same

    ids = x.flatten()
    restated, _ = forward(
        m,
        ids,
        torch.arange(len(ids), device="cuda"),
        f,
        context.board,
        previous,
        attend,
    )
    restated = restated.float().view(*x.shape, -1).cpu().numpy()
    report = dict(restatement=dict(bitwise=bool(np.array_equal(restated, eager)),
                                   logit_max=float(np.abs(restated - eager).max())))  # fmt: skip

    games = []  # (row, start, move-only length inside the row, also after the shift)
    for r in range(a.rows):
        starts = list(np.flatnonzero(rows[r, :-1] == 2348)) + [1024]
        for s, e in zip(starts[:-1], starts[1:]):
            toks = rows[r, s:e]
            ok = np.flatnonzero(toks[11:] >= 2346)
            k = 11 + int(ok[0]) if len(ok) else len(toks)
            k = min(k, 1024 - SHIFT - s)
            if k >= 11 + 4:
                games.append((r, int(s), k))
    cuts = [int(rng.integers(11, k - 2)) for _, _, k in games]
    prefixes = [rows[r, s : s + c] for (r, s, _), c in zip(games, cuts)]
    pfeats = [
        feats[r, s : s + c].astype(np.float32) for (r, s, _), c in zip(games, cuts)
    ]
    at = np.array([(r, s + c - 1) for (r, s, _), c in zip(games, cuts)]).T

    def run(origins):
        """Prefill the roots, then chains of true moves one token at a time, then one sibling each."""
        oracle.reset()
        z, prow = oracle.prefill(prefixes, pfeats, origins=origins)
        rowof, length = dict(enumerate(prow)), dict(enumerate(cuts))
        chain, got = [], []
        for _ in range(a.chain):
            live = [i for i, (_, _, k) in enumerate(games) if length[i] < k]
            if not live:
                break
            spans = [
                (games[i][0], games[i][1], games[i][1] + length[i] + 1) for i in live
            ]
            lens = np.array([length[i] + 1 for i in live])
            toks = np.array([rows[r, e - 1] for r, _, e in spans])
            ff = np.stack([feats[r, e - 1] for r, _, e in spans]).astype(np.float32)
            bb = np.stack([encode(rows[r, s:e][None])[0, -1] for r, s, e in spans])
            zz, new = oracle.extend(
                np.array([rowof[i] for i in live]), toks, lens, ff, bb
            )
            chain.extend(spans)
            got.extend(zz)
            for j, i in enumerate(live):
                rowof[i], length[i] = new[j], length[i] + 1
        alt, par, toks, lens, ff, bb = [], [], [], [], [], []
        brng = np.random.default_rng(2)
        for i, pre in enumerate(prefixes):
            legal = [
                t
                for t in from_prefix(pre).legal()
                if t != rows[games[i][0], games[i][1] + cuts[i]]
            ]
            if legal:
                t = int(brng.choice(legal))
                q = np.append(pre, t)
                alt.append((q, np.concatenate((pfeats[i], pfeats[i][-1:]))))
                par.append(prow[i])
                toks.append(t)
                lens.append(len(q))
                ff.append(pfeats[i][-1])
                bb.append(encode(q[None])[0, -1])
        branch, _ = oracle.extend(
            np.array(par), np.array(toks), np.array(lens), np.stack(ff), np.stack(bb)
        )
        return z, chain, np.stack(got), alt, branch

    t0 = time.perf_counter()
    z, chain, got, alt, branch = run(None)
    seconds = time.perf_counter() - t0
    cr, cc = np.array([(r, e - 1) for r, _, e in chain]).T
    target = rows[cr, cc + 1]
    # B
    fresh, _ = oracle.prefill(
        [rows[r, s:e] for r, s, e in chain],
        [feats[r, s:e].astype(np.float32) for r, s, e in chain],
    )
    fresh_branch, _ = oracle.prefill([q for q, _ in alt], [f for _, f in alt])
    report["B_cache_chain_vs_prefill"] = compare(fresh, got, target)
    report["B_cache_branch_vs_prefill"] = compare(fresh_branch, branch)
    # C
    origins = [r * 1024 + s for r, s, _ in games]
    mz, mchain, mgot, _, _ = run(origins)
    assert mchain == chain
    report["C_matched_prefill_vs_train"] = compare(
        ref[at[0], at[1]], mz, rows[at[0], at[1] + 1]
    )
    report["C_matched_chain_vs_train"] = compare(ref[cr, cc], mgot, target)
    report["C_matched_chain_vs_eager"] = compare(eager[cr, cc], mgot, target)
    # D
    report["D_prefill_vs_train"] = compare(ref[at[0], at[1]], z, rows[at[0], at[1] + 1])
    report["D_chain_vs_train"] = compare(ref[cr, cc], got, target)
    report["D_scale_train_vs_repacked"] = compare(
        ref[cr, cc], ref2[cr, cc + SHIFT], target
    )
    report["D_scale_train_vs_eager"] = compare(ref[cr, cc], eager[cr, cc], target)
    report.update(games=len(games), seconds=seconds, checkpoint=str(oracle.checkpoint), source=a.source,
                  tf32=torch.backends.cuda.matmul.allow_tf32, device=torch.cuda.get_device_name(0))  # fmt: skip
    print(json.dumps(report, indent=1), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
