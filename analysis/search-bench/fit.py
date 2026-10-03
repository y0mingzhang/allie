"""Refit search/policy.py's output calibration on the dev split and apply it to the benchmark.

Same functional form as the frozen policy (per-band alpha/beta mixture over five beta scales,
feature-gated, then the fitted temperature): all 21 coefficients refit jointly by maximum
likelihood on dev search outputs of the same model and budget, ridge 0.01 on gate and
temperature, feature standardization recomputed on dev. Elo-adaptive picks each position's
policy by its allocated budget, as Search does: 128 reuses the fixed-128 refit, and 256/512/1000
are fit on the dev positions routed to them. "temperature" refits only per-band alpha on the legal
policy (beta = 0), the no-search control for what recalibration alone buys.

Writes <tag>/devcal-<point>.npz (per-position CE, top-1 on the bench) and calibration-<tag>.json
next to this file: search/calibration.json with budget_policies replaced by the refits (5/8/25/460
are extra keys, reachable only through adapter.Bench); router, backup and old_parameters frozen.

usage: fit.py <tag> [budgets,...]   (needs <tag>/<budget> for the bench and <tag>-dev/<budget> for dev)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

import adapter

torch.set_default_dtype(torch.float64)
D = adapter.DATA / "search"
HERE = Path(__file__).resolve().parent
F = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0])
TC = np.r_[np.arange(16), 16 * np.exp(np.arange(47) / 7.06)]
FIXED = ["5", "8", "25", "128", "460"]


def load(folder, P):
    z = [np.load(f) for f in sorted(folder.glob("[0-9]*[0-9].npz"))]
    cat = lambda k: np.concatenate([x[k] for x in z])
    ix = cat("index")
    lens = np.concatenate([np.diff(x["offsets"]) for x in z])
    n, w = len(ix), lens.max()
    mask = np.arange(w)[None] < lens[:, None]
    logp, q = np.full((n, w), -np.inf), np.zeros((n, w))
    logp[mask] = np.log(np.maximum(cat("prior"), 1e-300))
    q[mask] = cat("values")
    heads = cat("root_heads")
    t = np.exp(heads[:, :63] - heads[:, :63].max(1, keepdims=True))
    seconds = np.array([P[i]["features"][-1, 0] for i in ix], float)
    known = seconds >= 0
    prior = np.exp(logp)
    mean = (prior * q).sum(1)
    f = np.c_[
        -(prior * np.where(mask, logp, 0)).sum(1),
        np.log(0.01 + np.sqrt((prior * (q - mean[:, None]) ** 2).sum(1))),
        cat("prefill") - 11,
        np.log1p((t / t.sum(1, keepdims=True)) @ TC),
        np.where(known, np.log1p(np.maximum(seconds, 0)), 0),
        known & (seconds <= 15),
        ~known,
    ]
    return dict(
        index=ix,
        logp=np.where(mask, logp, 0),
        q=q,
        mask=mask,
        f=f,
        known=known,
        band=np.array([P[i]["cell"] % 4 for i in ix]),
        target=cat("target_slot"),
        nodes=cat("nodes"),
        budget=np.maximum(cat("simulations"), 128),
    )


def rows(d, keep):
    return {k: v[keep] for k, v in d.items()}


def logprob(par, d, norm):
    x = torch.tensor(
        np.c_[np.ones(len(d["f"])), np.clip((d["f"] - norm[0]) / norm[1], -3, 3)]
    )
    z, q, mask = (torch.tensor(d[k]) for k in ("logp", "q", "mask"))
    b = torch.tensor(d["band"])
    a, beta = par["alpha"][b], par["beta"][b]
    # finite fill: -inf would put 0 * inf = NaN into the backward pass
    comp = (a[None, :, None] * z + (F[:, None] * beta)[:, :, None] * q).masked_fill(
        ~mask, -1e9
    )
    comp = comp.log_softmax(-1)
    mu = (x @ par["gate"]) * torch.tensor(d["known"])
    lw = (-0.5 * F.log()[None] ** 2 + mu[:, None] * F.log()).log_softmax(-1)
    lp = torch.logsumexp(lw.T[:, :, None] + comp, 0)
    eta = (1 + x[:, :5] @ par["temperature"]).clamp(0.5, 2)
    return (eta[:, None] * lp).masked_fill(~mask, -torch.inf).log_softmax(-1)


def fit(d, only_alpha=False):
    """(params, (mean, scale), LBFGS iterations)."""
    norm = (d["f"].mean(0), np.maximum(d["f"].std(0), 1e-6))
    par = dict(
        alpha=torch.ones(4),
        beta=torch.zeros(4) if only_alpha else torch.full((4,), 3.0),
        gate=torch.zeros(8),
        temperature=torch.zeros(5),
    )
    free = [par["alpha"]] if only_alpha else list(par.values())
    for p in free:
        p.requires_grad_(True)
    opt = torch.optim.LBFGS(
        free, max_iter=500, tolerance_grad=1e-9, line_search_fn="strong_wolfe"
    )
    t = torch.tensor(d["target"])

    def loss():
        opt.zero_grad()
        nll = -logprob(par, d, norm)[torch.arange(len(t)), t].sum()
        out = nll + 0.5 * 0.01 * sum((p**2).sum() for p in free[2:])
        out.backward()
        return out

    opt.step(loss)
    return {k: v.detach() for k, v in par.items()}, norm, opt.state[free[0]]["n_iter"]


def score(par, norm, d):
    with torch.no_grad():
        lp = logprob(par, d, norm).numpy()
    return -lp[np.arange(len(d["target"])), d["target"]], lp.argmax(1) == d["target"]


def policy_json(par, norm, iters):
    return dict(
        fold=None,
        root={
            str(g): dict(
                alpha=float(par["alpha"][g]),
                beta=float(par["beta"][g]),
                converged=iters < 500,
                iterations=int(iters),
            )
            for g in range(4)
        },
        mean=norm[0].tolist(),
        scale=norm[1].tolist(),
        gate=par["gate"].tolist(),
        temperature=par["temperature"].tolist(),
    )


def main():
    tag = sys.argv[1]
    bench, dev = adapter.positions(), adapter.dev_positions()
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    have = lambda b: (only is None or b in only) and (D / f"{tag}-dev" / b).exists() and (D / tag / b).exists()
    cal = json.loads((adapter.REPO / "search/calibration.json").read_text())
    fits, out = {}, {}

    def emit(key, ce, top1, bd, dev_ce):
        np.savez(
            D / tag / f"devcal-{key}.npz",
            index=bd["index"],
            ce=ce,
            top1=top1,
            nodes=bd["nodes"],
        )
        out[key] = dict(
            dev_n=len(dev_ce),
            dev_ce=float(dev_ce.mean()),
            bench_n=len(ce),
            bench_ce=float(ce.mean()),
            bench_acc=float(100 * top1.mean()),
        )
        print(key, json.dumps(out[key]), flush=True)

    if have("legal"):
        dd, bd = load(D / f"{tag}-dev" / "legal", dev), load(D / tag / "legal", bench)
        par, norm, _ = fit(dd, only_alpha=True)
        emit("temperature", *score(par, norm, bd), bd, score(par, norm, dd)[0])
        out["temperature"]["alpha"] = par["alpha"].tolist()
    for b in FIXED:
        if have(b):
            dd, bd = load(D / f"{tag}-dev" / b, dev), load(D / tag / b, bench)
            fits[b] = fit(dd)
            emit(b, *score(*fits[b][:2], bd), bd, score(*fits[b][:2], dd)[0])
    if have("adaptive") and "128" in fits:
        dd, bd = (
            load(D / f"{tag}-dev" / "adaptive", dev),
            load(D / tag / "adaptive", bench),
        )
        ce, top1, dce = (
            np.zeros(len(bd["target"])),
            np.zeros(len(bd["target"]), bool),
            np.zeros(len(dd["target"])),
        )
        for b in np.unique(bd["budget"]):
            if b != 128:
                fits[str(b)] = fit(rows(dd, dd["budget"] == b))
            kb, kd = bd["budget"] == b, dd["budget"] == b
            ce[kb], top1[kb] = score(*fits[str(b)][:2], rows(bd, kb))
            dce[kd] = score(*fits[str(b)][:2], rows(dd, kd))[0]
        emit("adaptive", ce, top1, bd, dce)
    for b, (par, norm, iters) in fits.items():
        cal["budget_policies"][b] = policy_json(par, norm, iters)
    cal["provenance"] = dict(
        base="search/calibration.json",
        refit="budget_policies only; adaptive router, backup, cpuct and "
        "old_parameters (fixed 64/1000, projection) frozen",
        model=tag,
        fit=str(Path(__file__).resolve()),
        dev="data/dev.jsonl + data/dev_expert.jsonl (July 2026 blitz, disjoint from strat-eval-v1), "
        "3000 positions per mover band, dev-positions.npz",
        results=out,
    )
    (HERE / f"calibration-{tag}.json").write_text(json.dumps(cal, indent=1) + "\n")


if __name__ == "__main__":
    main()
