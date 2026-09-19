"""Compute multipliers of data-v1 mixing policies against the control mix.

Control's compute-optimal curve is the isoflop-v1 law for ours, shifted by the mean residual of the
data-v1 control runs. Original-val metrics use the saved shared-floor fit; strat-eval-v1 metrics
(macro over 16 cells, expert macro over the four >=2400 cells) refit the law on the isoflop-v1 ours
runs' strat scores. A policy's multiplier at budget C is C'/C where L_control(C') = L_policy(C).
Budgets are N*D as in isoflop-v1. Writes data-v1-multipliers.json and prints a markdown table.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isoflop_fit import additive, best_loss

ROOT = Path("/home/yimingz3/src/allie/results/recipe10x")
EVAL = ROOT.parent / "lm-eval"
METRICS = ("move", "expert2400", "strat_macro", "strat_expert")
BUDGETS = {"1e16": 1e16, "3e16": 3e16, "1e17": 1e17, "3e17": 3e17}
TARGET = dict(move=2.0, expert2400=10.0, strat_macro=2.0, strat_expert=10.0)
# Label of each tag's control runs, or (label, tag the controls come from); a tag without controls
# is not scored (no fallback). Round 3 is scored against B_2; its external-source screens (b2x) use
# the B_2 controls of round3g / round3p.
B2 = "mover_rule+up4+cf3+lr5+t0.2+w0.2"
BASELINES = {
    "pf115h": "control+clk",
    "pf052hb2": B2,
    "pf115hb2": B2,
    "pf052hb2x": (B2, "pf052hb2"),
}


def strat(name):
    f = EVAL / name / "strat-v1.json"
    if not f.exists():
        return dict(strat_macro=np.nan, strat_expert=np.nan)
    s = json.loads(f.read_text())
    return dict(strat_macro=s["macro"], strat_expert=s["expert_macro"])


def law(metric):
    if metric in ("move", "expert2400"):
        f = json.loads((ROOT / f"isoflop-v1/fit-{metric}.json").read_text())
        f = f["shared_E"]["fit"]["ours"]
        return np.array(
            [np.log(f["E"]), np.log(f["A"]), f["alpha"], np.log(f["B"]), f["beta"]]
        )
    pts = []
    for p in sorted((ROOT / "isoflop-v1/results").glob("iso-ours-*.json")):
        r = json.loads(p.read_text())
        pts.append((r["n_nonembed"], r["tokens"], r["budget"], strat(p.stem)[metric]))
    pts = np.array([x for x in pts if np.isfinite(x[3])])
    assert len(pts) >= 8, f"only {len(pts)} isoflop-v1 runs scored on strat-eval-v1"
    return additive(dict(ours=pts), shared=False)[:5]


def results():
    rows = []
    for f in sorted(ROOT.glob("data-v1-*/results/*.json")):
        r = json.loads(f.read_text())
        rows.append(
            dict(
                policy=r["policy"]
                + ("+clk" if r.get("clock") else "")
                + ("+elo" if r.get("elo") else "")
                + (f"+cf{r['feats']}" if r.get("feats") else "")
                + (f"+lr{r['input_lr']:g}" if "input_lr" in r else "")
                + "".join(
                    f"+{k[4]}{r[k]:g}" for k in ("aux_time", "aux_wdl") if r.get(k)
                ),
                tag=r.get("tag", ""),
                seed=r["seed"],
                budget=r.get("budget", "3e16"),
                move=r["ce"]["move"],
                expert2400=r["ce"]["expert2400"],
                **strat(f.stem),
            )
        )
    return rows


def multiplier(curve, loss, c):
    if not np.isfinite(loss):
        return float("nan")
    g = lambda lc: curve(np.exp(lc)) - loss
    lo, hi = np.log(c) - 9, np.log(c) + 9
    return float(np.exp(brentq(g, lo, hi)) / c) if g(lo) > 0 > g(hi) else float("nan")


def main():
    """Each tag (e.g. pf031 repetition-matched, full) is measured against the control of the same tag."""
    rows, table, fit = results(), [], {}
    for tag in sorted({r["tag"] for r in rows}):
        base = BASELINES.get(tag, "control")
        base, src = base if isinstance(base, tuple) else (base, tag)
        ctrl = [r for r in rows if r["policy"] == base and r["tag"] == src]
        if not ctrl:
            print(f"[{tag or 'untagged'}] no {base} controls: not scored")
            continue
        t, f = score([r for r in rows if r["tag"] == tag], ctrl)
        table += t
        fit[tag or "untagged"] = f
    (ROOT / "data-v1-multipliers.json").write_text(
        json.dumps(dict(fit=fit, table=table, target=TARGET), indent=2) + "\n"
    )
    metrics = [m for m in METRICS if any(f"cm_{m}" in r for r in table)]
    head = " | ".join(f"{m} CE | CM {m}" for m in metrics)
    print(f"| budget | policy | seeds | {head} |")
    print("|---|---|---|" + "---|---|" * len(metrics))
    for r in table:
        cells = " | ".join(
            f"{r[m]:.4f} | {r[f'cm_{m}']:.2f}x [{r[f'cm_{m}_band'][0]:.2f}, {r[f'cm_{m}_band'][1]:.2f}]"
            if f"cm_{m}" in r
            else "- | -"
            for m in metrics
        )
        label = r["policy"] + (f"-{r['tag']}" if r["tag"] else "")
        print(f"| {r['budget']} | {label} | {r['seeds']} | {cells} |")
    for tag, f in fit.items():
        for m in (k for k in METRICS if k in f):
            print(
                f"[{tag}] {m}: control shift {f[m]['shift']:+.4f}, residual by budget {f[m]['residual']}"
            )
        for b in BUDGETS:
            if b in f:
                print(f"[{tag}] {b}: control seed std {f[b]['control_seed_std']}")


def score(rows, ctrl):
    metrics = [
        m
        for m in METRICS
        if m in ("move", "expert2400") or any(np.isfinite(r[m]) for r in ctrl)
    ]
    fit, curves = {}, {}
    for m in metrics:
        p = law(m)
        res = [
            (r["budget"], r[m] - best_loss(p, BUDGETS[r["budget"]]))
            for r in ctrl
            if np.isfinite(r[m])
        ]
        shift = float(np.mean([x for _, x in res])) if res else 0.0
        curves[m] = lambda c, p=p, s=shift: best_loss(p, c) + s
        fit[m] = dict(
            law=p.tolist(),
            shift=shift,
            residual={
                b: float(np.mean([x for bb, x in res if bb == b]))
                for b in BUDGETS
                if any(bb == b for bb, _ in res)
            },
        )
    table = []
    for b, c in BUDGETS.items():
        noise = {}
        for m in metrics:
            v = [r[m] for r in ctrl if r["budget"] == b and np.isfinite(r[m])]
            noise[m] = float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
        for pol in sorted({r["policy"] for r in rows if r["budget"] == b}):
            got = [r for r in rows if r["budget"] == b and r["policy"] == pol]
            row = dict(budget=b, policy=pol, tag=got[0]["tag"], seeds=len(got))
            for m in metrics:
                v = [r[m] for r in got if np.isfinite(r[m])]
                loss = float(np.mean(v)) if v else float("nan")
                row[m] = loss
                row[f"cm_{m}"] = multiplier(curves[m], loss, c)
                row[f"cm_{m}_band"] = [
                    multiplier(curves[m], loss + noise[m], c),
                    multiplier(curves[m], loss - noise[m], c),
                ]
            table.append(row)
        fit[b] = dict(control_seed_std=noise)
    return table, fit


if __name__ == "__main__":
    main()
