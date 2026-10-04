"""Recompute coverage search's calibrated distributions (cov_p{N}) of calib_search.py chunks under
another output calibration, from the saved values: allie.search.policy.policy needs only the legal
policy (any per-position shift of the logits), Q, the think-time head, the ply, the mover's clock and
the cell. --check recomputes under the calibration the chunks were made with and reports the largest
difference from the stored cov_p.

python calib_recal.py POSITIONS.npz SEARCH_DIR CALIBRATION.json OUT_DIR [--check]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from allie.search.policy import policy

KEY = {8: "128", 32: "128", 128: "128", 256: "256"}  # allie.lichess.tree.POLICY


def recompute(z, meta, par):
    o = z["offsets"]
    n, w = len(z["index"]), int(np.diff(o).max())
    ids, mask = np.zeros((n, w), int), np.zeros((n, w), bool)
    root = np.zeros((n, 2432))
    for j in range(n):
        a, b = o[j], o[j + 1]
        ids[j, : b - a], mask[j, : b - a] = z["legal"][a:b] - 378, True
        root[j, 378 + ids[j, : b - a]] = np.log(
            np.maximum(z["prior"][a:b].astype(float), 1e-300)
        )
        root[j, 2350:2416] = z["heads"][j]
    m = meta[z["index"]]
    rows = [
        dict(prefix=range(11 + int(k)), cell=int(c))
        for k, c in zip(m[:, 6], m[:, 9], strict=True)
    ]
    seconds = m[:, 7].astype(float)
    out = {}
    for key in [k for k in z.files if k.startswith("cov_p")]:
        budget = int(key[5:])
        q = np.zeros((n, w))
        for j in range(n):
            q[j, : o[j + 1] - o[j]] = z[f"cov_q{budget}"][o[j] : o[j + 1]]
        p = policy(
            rows, root, q, ids, mask, seconds, par["budget_policies"][KEY[budget]]
        )
        out[key] = np.concatenate([p[j, : o[j + 1] - o[j]] for j in range(n)]).astype(
            np.float32
        )
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("positions")
    a.add_argument("search")
    a.add_argument("calibration")
    a.add_argument("out")
    a.add_argument("--check", action="store_true")
    a = a.parse_args()
    meta = np.load(a.positions)["meta"]
    par = json.loads(Path(a.calibration).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    worst = 0.0
    for f in sorted(Path(a.search).glob("[0-9]*.npz")):
        if ".partial" in f.name or (out / f.name).exists():
            continue
        z = np.load(f)
        new = recompute(z, meta, par)
        if a.check:
            worst = max(worst, *(float(np.abs(new[k] - z[k]).max()) for k in new))
            print(f.name, "largest |recomputed - stored|", worst, flush=True)
            continue
        keep = {k: z[k] for k in z.files if not k.startswith("cov_p")}
        np.savez((out / f.name).with_suffix(".partial.npz"), **keep, **new)
        (out / f.name).with_suffix(".partial.npz").replace(out / f.name)
    print("done", "worst difference" if a.check else "", worst if a.check else "")


if __name__ == "__main__":
    main()
