"""Refit allie.search coverage's output calibration (budget_policies) for one model on the dev split,
from calib_search.py outputs, with the search-bench fit (maia3-bench/search/fit.py: same functional
form and optimizer), at budgets 128 and 256. Writes the bot's calibration: the given base with those
policies replaced, '5' taken from --five (that model's 5-simulation refit), and provenance.

python calib_outcal.py DEV_POSITIONS.npz SEARCH_DIR BASE.json OUT.json [--five calibration-ann2.json] [--model NAME]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/yimingz3/src/allie/results/recipe10x/maia3-bench/search")
import fit as bench  # noqa: E402  (search-bench's refit: logprob, fit, score, policy_json, TC)


def dataset(meta, chunks, budget):
    """fit.py's load() dict from calib_search.py chunks at one budget."""
    idx, logp, q, lens, heads, target = [], [], [], [], [], []
    for z in chunks:
        for n, i in enumerate(z["index"]):
            a, b = z["offsets"][n], z["offsets"][n + 1]
            legal = z["legal"][a:b].tolist()
            idx.append(int(i))
            logp.append(np.log(np.maximum(z["prior"][a:b].astype(float), 1e-300)))
            q.append(z[f"cov_q{budget}"][a:b].astype(float))
            lens.append(b - a)
            heads.append(z["heads"][n])
            target.append(legal.index(int(meta[i, 8])))
    n, w = len(idx), max(lens)
    mask = np.arange(w)[None] < np.array(lens)[:, None]
    L, Q = np.full((n, w), -np.inf), np.zeros((n, w))
    for r, (lp, qq) in enumerate(zip(logp, q, strict=True)):
        L[r, : len(lp)], Q[r, : len(qq)] = lp, qq
    heads = np.array(heads, float)
    t = np.exp(heads[:, :63] - heads[:, :63].max(1, keepdims=True))
    m = meta[idx]
    seconds = m[:, 7].astype(float)
    known = seconds >= 0
    prior = np.exp(L)
    mean = (prior * Q).sum(1)
    f = np.c_[
        -(prior * np.where(mask, L, 0)).sum(1),
        np.log(0.01 + np.sqrt((prior * (Q - mean[:, None]) ** 2).sum(1))),
        m[:, 6],  # ply: the prefix length less the 11 header tokens
        np.log1p((t / t.sum(1, keepdims=True)) @ bench.TC),
        np.where(known, np.log1p(np.maximum(seconds, 0)), 0),
        known & (seconds <= 15),
        ~known,
    ]
    return dict(index=np.array(idx), logp=np.where(mask, L, 0), q=Q, mask=mask, f=f, known=known,
                band=m[:, 9] % 4, target=np.array(target), nodes=np.zeros(n), budget=np.full(n, budget))  # fmt: skip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("positions")
    p.add_argument("search")
    p.add_argument("base")
    p.add_argument("out")
    p.add_argument(
        "--five",
        default="/home/yimingz3/src/allie/results/recipe10x/maia3-bench/search/calibration-ann2.json",
    )
    p.add_argument(
        "--model",
        default="allie-2.0 annealed (results/pretrain/ann-all-p05-t1907, step 1907)",
    )
    a = p.parse_args()
    meta = np.load(a.positions)["meta"]
    chunks = [
        np.load(f)
        for f in sorted(Path(a.search).glob("[0-9]*.npz"))
        if ".partial" not in f.name
    ]
    cal = json.loads(Path(a.base).read_text())
    old = cal["budget_policies"]
    results = {}
    for budget in (128, 256):
        d = dataset(meta, chunks, budget)
        par, norm, iters = bench.fit(d)
        new = bench.policy_json(par, norm, iters)
        before = bench.score(*params(old[str(budget)]), d)[0].mean()
        after = bench.score(par, norm, d)[0].mean()
        results[str(budget)] = dict(
            dev_n=len(d["target"]),
            dev_ce_before=float(before),
            dev_ce_after=float(after),
            iterations=int(iters),
        )
        print(budget, json.dumps(results[str(budget)]), flush=True)
        cal["budget_policies"][str(budget)] = new
    five = json.loads(Path(a.five).read_text())
    cal["budget_policies"]["5"] = five["budget_policies"]["5"]
    cal["provenance"] = dict(
        model=a.model,
        refit=(
            "budget_policies 128 and 256 refit on this model's dev search outputs (calib_outcal.py with "
            "search-bench fit.py's form and optimizer); 5 from "
            + a.five
            + "; router, backup, cpuct and "
            "old_parameters as " + a.base
        ),  # fmt: skip
        dev="data/dev.jsonl + data/dev_expert.jsonl (July 2026 blitz, disjoint from strat-eval-v1), 3000 positions per mover band",
        results=results,
    )
    Path(a.out).write_text(json.dumps(cal, indent=1) + "\n")


def params(policy):
    """A calibration json policy back to fit.py's (params, norm) for scoring."""
    import torch

    g = sorted(policy["root"], key=int)
    par = dict(alpha=torch.tensor([policy["root"][k]["alpha"] for k in g]),
               beta=torch.tensor([policy["root"][k]["beta"] for k in g]),
               gate=torch.tensor(policy["gate"]), temperature=torch.tensor(policy["temperature"]))  # fmt: skip
    return par, (np.array(policy["mean"]), np.array(policy["scale"]))


if __name__ == "__main__":
    main()
