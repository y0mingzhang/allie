"""check(argv, run, store): fail closed unless a stored orchard argv (omake.py, before path relocation) is the frozen
planned run's modelexp.train_args: every flag that sets what the run computes equals the planned run's value, no flag
is missing or unknown; only --data (a path) and the checkpoint / eval cadence flags are operational."""

import json

DATA_FLAGS = {"--clock-feats": "feats", "--input-lr-mul": "input_lr", "--aux-time": "aux_time", "--aux-wdl": "aux_wdl"}
OPERATIONAL = {"--data", "--eval-every", "--checkpoint-every", "--keep-checkpoints", "--val-rows"}


def check(argv, r):
    extra = list(map(str, r.get("extra_args", [])))
    assert argv[len(argv) - len(extra) :] == extra, f"extra args {argv[len(argv) - len(extra):]} != {extra}"
    a, kv, i = argv[: len(argv) - len(extra)], {}, 0
    while i < len(a):
        f = a[i]
        assert f.startswith("--") and f not in kv, f"argv: unexpected {f!r} at {i}"
        if f == "--deterministic":
            kv[f], i = "", i + 1
        else:
            kv[f], i = a[i + 1], i + 2
    want = {
        "--name": r["name"], "--width": r["width"], "--layers": r["layers"], "--head-dim": 64, "--steps": r["steps"],
        "--initial-batch-rows": 512, "--micro-batch": r.get("micro_batch", 16), "--lr-scale": r.get("lr", 1),
        "--seed": r["seed"], "--deterministic": "", "--mix": r["policy"], "--mix-pool-frac": r["pool_frac"],
        "--mix-stores": ",".join(r["stores"]), "--mix-months": ",".join(r["months"]),
        "--wsd-end-step": r["steps"], "--wsd-decay-start": r["decay_start"],
    }  # fmt: skip
    want |= {f: r[k] for f, k in DATA_FLAGS.items() if r.get(k)}
    bad = [f for f, v in want.items() if kv.get(f) != str(v)]
    bad += [f for f, v in (("--wsd-schedule", r["schedule"]), ("--arch", r["arch"])) if json.loads(kv.get(f, "null")) != v]
    hist = kv.get("--mix-history")
    bad += ["--mix-history"] * (bool(r.get("history")) != bool(hist) or bool(hist) and not hist.endswith("/history-counts.json"))
    bad += sorted(set(kv) - set(want) - OPERATIONAL - {"--wsd-schedule", "--arch", "--mix-history"})
    assert not bad, f"{r['name']}: stored argv differs from the frozen run in {bad}"
