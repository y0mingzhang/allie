"""A gpu_kl.py run as harness.py output under one KL backup, for scale_score.py: per chunk every root's moves,
prior, heads and per budget its Q (nan unsearched) and leaves (network evaluations times views).

python to_harness.py RUN OUT [--backup own:opp[:soft]] [--value-views 0,1] [--budgets 8,...,4096]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "src/allie/search")]
import kl  # noqa: E402
from score_kl import parse  # noqa: E402
from evaluate import soft_backup  # noqa: E402


def restrict(res, sel):
    """Harness chunk arrays for the roots in `sel` only."""
    off = res["offsets"]
    moves = np.repeat(sel, np.diff(off))
    out = {k: (v[sel] if k in ("index", "heads") or k.startswith("leaves") else v[moves] if k in ("legal", "prior", "prior_v") or k.startswith(("q", "cov_")) else v)
           for k, v in res.items()}  # fmt: skip
    out["offsets"] = np.r_[0, np.cumsum(np.diff(off)[sel])]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("out")
    p.add_argument("--backup", default="4:0", help="kl.backup spec own:opp[:s][:kKAPPA], or cov (allie.search's soft backup)")
    p.add_argument(
        "--value-views",
        default="",
        help="the views whose W/D/L the backup reads (default the run's)",
    )
    p.add_argument("--budgets", default="8,16,32,64,128,256,512,1024,2048,4096")
    p.add_argument("--subset", default="", help="keep only these positions (an index .npy)")
    p.add_argument("--calib", action="store_true", help="calib_cells.py's chunk keys (legal, prior, heads, cov_q{b}k)")
    a = p.parse_args()
    budgets = [int(b) for b in a.budgets.split(",")]
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    views = len(json.loads((run / "args.json").read_text()).get("views", "0").split(","))
    for f in sorted(run.glob("[0-9]*.npz")):
        if ".partial" in f.name:
            continue
        z = dict(np.load(f))
        if a.value_views:
            z["wdl"] = z["wdlv"][:, [int(v) for v in a.value_views.split(",")]].mean(1)
        roots, n = z["roots"], len(z["roots"])
        off = np.r_[0, np.cumsum(z["root_len"])]
        kid = np.full(off[-1], -1)
        d1 = np.flatnonzero(z["depth"] == 1)
        r = z["owner"][d1]
        slot = {
            (int(o), int(t)): off[o] + j
            for o in range(n)
            for j, t in enumerate(z["root_moves"][off[o] : off[o + 1]])
        }
        kid[[slot[int(o), int(t)] for o, t in zip(r, z["token"][d1])]] = d1
        res = dict(index=z["index"], budgets=np.array(budgets), legal=z["root_moves"], prior=z["root_probs"],
                   heads=z["heads"], offsets=off)  # fmt: skip
        if "root_probs_v" in z:  # the roots' policy under each view
            res["prior_v"] = z["root_probs_v"]
        assert (roots == np.flatnonzero(z["parent"] < 0)).all()
        for b in budgets:
            keep = z["cost" if "cost" in z else "tag"] <= b
            V = soft_backup(z, keep) if a.backup == "cov" else kl.backup(z, keep, **parse(a.backup))[0]
            live = (kid >= 0) & keep[np.maximum(kid, 0)]
            q = np.where(live, -V[np.maximum(kid, 0)], np.nan)
            if a.backup != "cov" and parse(a.backup)["squash"]:  # unsearched at the root's own value, on the same scale
                x = parse(a.backup)["squash"] * np.clip(z["wdl"][roots, 0] - z["wdl"][roots, 2], -1, 1)
                q = np.where(live, q, np.repeat(np.arctanh(x), z["root_len"]))
            if a.calib:  # calib_cells' chunks: Q in W - L units (cov_q) and on the backup's log-odds scale (cov_x)
                s = parse(a.backup)["squash"]
                res[f"cov_x{b}k"] = q.astype(np.float32)
                q = np.clip(np.tanh(q) / s, -1, 1) if s else q
                res[f"cov_q{b}k"] = q.astype(np.float32)
                continue
            res[f"q{b}"] = q.astype(np.float32)
            leaves = np.bincount(
                z["owner"][keep & ~z["terminal"] & (z["parent"] >= 0)], minlength=n
            )
            res[f"leaves{b}"] = (views * leaves).astype(np.int32)
        if a.subset:
            res = restrict(res, np.isin(res["index"], np.load(a.subset)))
            if not len(res["index"]):
                continue
        np.savez(out / f.name, **res)
    print(f"{out}: {len(list(out.glob('*.npz')))} chunks", flush=True)


if __name__ == "__main__":
    main()
