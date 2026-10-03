"""Aggregate the per-position scores into the two report views, with game-clustered CIs."""
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path("/data/group_data/dei-group/yimingz3/allie/maia3-bench")
SCORES = DATA / "scores"
CELLS = {4: "blitz/<1400", 5: "blitz/1400-2000", 6: "blitz/2000-2400", 7: "blitz/>=2400"}
REPS = 2000


def views():
    with np.load(DATA / "games.npz") as z:
        sel, keep = z["sel"], z["keep"]
    s = sel[keep]  # game, ply, cell, mover clock, opponent clock
    ours = s[:, 3] >= 30
    return s, {"all": np.ones(len(s), bool), "maia_protocol": (s[:, 1] >= 20) & ours}


def load():
    s, masks = views()
    with np.load(DATA / "games.npz") as z:
        keep = z["keep"]
    out = {}
    for f in sorted(SCORES.glob("*.npz")):
        z = np.load(f)
        if "ce_legal" not in z:  # maia scorers store only the sampled positions
            assert np.array_equal(z["keep"], keep), f
            out[f.stem] = dict(ce=z["ce"], top1=z["top1"])
        elif len(z["ce_legal"]) == len(keep):  # already restricted to the sample
            out[f.stem] = dict(ce=z["ce_legal"], top1=z["top1"], ce_full=z["ce"])
        else:
            out[f.stem] = dict(ce=z["ce_legal"][keep], top1=z["top1"][keep], ce_full=z["ce"][keep])
    return s, masks, out


def boot(games, x, rng):
    """Cluster bootstrap over games: returns REPS resampled means of x."""
    uniq, idx = np.unique(games, return_inverse=True)
    order = np.argsort(idx, kind="stable")
    starts = np.searchsorted(idx[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(order))
    sums = np.add.reduceat(x[order], starts) if len(order) else np.zeros(0)
    ns = (ends - starts).astype(float)
    pick = rng.integers(0, len(uniq), (REPS, len(uniq)))
    return sums[pick].sum(1) / ns[pick].sum(1)


def stats(games, x, rng):
    b = boot(games, x, rng)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return dict(mean=float(x.mean()), lo=float(lo), hi=float(hi), n=int(len(x)))


def main():
    s, masks, scored = load()
    rng = np.random.default_rng(7)
    report = {}
    for vname, vmask in masks.items():
        report[vname] = {}
        for name, d in scored.items():
            per_cell, acc_cells, ce_cells = {}, [], []
            for c, label in CELLS.items():
                m = vmask & (s[:, 2] == c)
                per_cell[label] = dict(
                    accuracy=stats(s[m, 0], d["top1"][m].astype(float), rng),
                    ce=stats(s[m, 0], d["ce"][m], rng),
                )
                acc_cells.append(d["top1"][m].astype(float).mean())
                ce_cells.append(d["ce"][m].mean())
            report[vname][name] = dict(
                cells=per_cell,
                macro_accuracy=float(np.mean(acc_cells)),
                macro_ce=float(np.mean(ce_cells)),
                pooled_accuracy=stats(s[vmask, 0], scored[name]["top1"][vmask].astype(float), rng),
                pooled_ce=stats(s[vmask, 0], scored[name]["ce"][vmask], rng),
            )
        # paired differences against our dense model on identical positions
        base = "ours-129m"
        if base in scored:
            report[vname]["paired_vs_" + base] = {
                name: dict(
                    d_accuracy=stats(s[vmask, 0],
                                     (d["top1"][vmask].astype(float) - scored[base]["top1"][vmask]), rng),
                    d_ce=stats(s[vmask, 0], d["ce"][vmask] - scored[base]["ce"][vmask], rng),
                )
                for name, d in scored.items() if name != base
            }
    speed = {p.stem: json.loads(p.read_text()) for p in sorted(SCORES.glob("*.json"))}
    out = dict(views=list(masks), positions=int(len(s)),
               per_view_counts={k: int(v.sum()) for k, v in masks.items()},
               report=report, speed=speed)
    (SCORES / "aggregate.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({v: {m: dict(acc=round(r["macro_accuracy"], 4), ce=round(r["macro_ce"], 4))
                          for m, r in report[v].items() if not m.startswith("paired")}
                      for v in report}, indent=2))


if __name__ == "__main__":
    main()
