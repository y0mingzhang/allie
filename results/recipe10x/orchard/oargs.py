"""check(argv, run, study, store): fail closed unless a stored orchard argv (omake.py, before path relocation) is the
frozen planned run's modelexp.train_args: every flag is present with the planned value, input paths are the study's own
history counts and the store's validation rows (orun.py relocates both), no flag is unknown; only the checkpoint cadence's
value is operational (orun.py sets it)."""

import json

DATA_FLAGS = {"--clock-feats": "feats", "--input-lr-mul": "input_lr", "--aux-time": "aux_time", "--aux-wdl": "aux_wdl"}


def check(argv, r, study, store):
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
        "--wsd-end-step": r["steps"], "--wsd-decay-start": r["decay_start"], "--eval-every": 10**7,
        "--keep-checkpoints": 2, "--val-rows": 1024, "--data": f"{store}/lichess_tokens_v2",
    }  # fmt: skip
    want |= {"--mix-history": f"{study}/history-counts.json"} if r.get("history") else {}
    want |= {f: r[k] for f, k in DATA_FLAGS.items() if r.get(k)}
    bad = [f for f, v in want.items() if kv.get(f) != str(v)]
    bad += [f for f, v in (("--wsd-schedule", r["schedule"]), ("--arch", r["arch"])) if json.loads(kv.get(f, "null")) != v]
    bad += ["--checkpoint-every"] * ("--checkpoint-every" not in kv)
    bad += sorted(set(kv) - set(want) - {"--wsd-schedule", "--arch", "--checkpoint-every"})
    assert not bad, f"{r['name']}: stored argv differs from the frozen run in {bad}"
