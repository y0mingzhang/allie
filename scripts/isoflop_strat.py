"""isoflop-v1 scaling fits on the golden eval (strat-eval-v1 macro and expert macro).

Same machinery as isoflop_fit.py: per-budget parabola minima and the additive law
E + A (N/1e7)^-alpha + B (D/1e8)^-beta, per recipe with its own floor, plus the shared-floor fit and
the Qwen->ours compute multiplier once both recipes have at least 8 scored runs.
Writes results/recipe10x/strat-eval-v1/fit-<metric>.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isoflop_fit import additive, describe, isoflop, multiplier

ISO = Path("/home/yimingz3/src/allie/results/recipe10x/isoflop-v1")
EVAL = ISO.parents[1] / "lm-eval"
OUT = ISO.parent / "strat-eval-v1"
METRICS = dict(strat_macro="macro", strat_expert="expert_macro")


def load(key):
    data = {}
    for recipe in ("ours", "qwen"):
        pts = []
        for f in sorted((ISO / "results").glob(f"iso-{recipe}-*.json")):
            s = EVAL / f.stem / "strat-v1.json"
            if s.exists():
                r = json.loads(f.read_text())
                pts.append(
                    (
                        r["n_nonembed"],
                        r["tokens"],
                        r["budget"],
                        json.loads(s.read_text())[key],
                    )
                )
        if len(pts) >= 8:
            data[recipe] = np.array(pts)
    return data


def main():
    for metric, key in METRICS.items():
        data = load(key)
        out = dict(
            metric=metric,
            points={r: len(d) for r, d in data.items()},
            isoflop=isoflop(data),
        )
        for r, d in data.items():
            theta = additive({r: d}, shared=False)
            p = (
                theta[:5] if r == "ours" else theta[5:]
            )  # unpack slots: ours first, qwen second
            law = describe(np.r_[p, p], {"ours": d}, False)["ours"]
            out[f"{r}_law"] = law
        if len(data) == 2:
            for shared in (True, False):
                theta = additive(data, shared)
                out["shared_E" if shared else "separate_E"] = dict(
                    fit=describe(theta, data, shared),
                    multiplier={
                        f"{c:.0e}": multiplier(theta, shared, c)
                        for c in (1e17, 3e17, 1e18, 1e19)
                    },
                )
        (OUT / f"fit-{metric}.json").write_text(
            json.dumps(out, indent=2, default=float) + "\n"
        )
        print(f"== {metric}: points {out['points']}")
        for r, v in out["isoflop"].items():
            mins = ", ".join(
                f"{m['budget']:.0e}: L*={m['loss_opt']:.4f} N*={m['n_opt'] / 1e6:.0f}M"
                for m in v["minima"]
            )
            print(f"  {r} minima {mins}; N* exponent {v['n_opt_exponent']}")
        for r in data:
            law = out[f"{r}_law"]
            print(
                f"  {r} law E={law['E']:.3f} A={law['A']:.3f} alpha={law['alpha']:.3f} B={law['B']:.3f} beta={law['beta']:.3f} rmse={law['rmse']:.4f}"
            )
        for k in ("shared_E", "separate_E"):
            if k in out:
                print(f"  {k} Qwen->ours multiplier {out[k]['multiplier']}")


if __name__ == "__main__":
    main()
