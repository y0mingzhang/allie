"""Score an orchard-trained run of a frozen moe-knobs study on babel, through the study's own run_one (which sees
done.json at steps and only evaluates). First checks the hand-back: orchard.json present, trained to its steps, and
config.json's arguments equal to the plan's (modelexp.numerics; months compared by store/month name, since orchard
stages them elsewhere, and the history file read from this study's copy, since orchard reads its mirror); then writes
the run's owner.json as the frozen driver computes it.

  score_handback.py STUDY_DIR INDEX [--check]     --check: verify only, no owner.json, no scoring
"""

import json
import os
import sys
from pathlib import Path

study, i, *flags = sys.argv[1:]
S = Path(os.path.abspath(study))  # as the Slurm array tasks claim runs
sys.path.insert(0, str(S))
import modelexp as mx

plan = json.loads((S / "plan.json").read_text())
bad = [k for k, v in plan["hashes"].items() if k != "baseline" and mx.sha(S / k) != v]
assert not bad, f"frozen copies changed since plan: {sorted(bad)}"
r = plan["runs"][int(i)]
out = mx.pretrained(r)
assert (out / "orchard.json").exists(), f"{out}: not an orchard hand-back"
assert json.loads((out / "done.json").read_text())["stop_reason"] == "steps"
args = json.loads((out / "config.json").read_text())["args"]
# orchard's mirror of the history file, checked against plan.json by orun.py
if (h := args.get("mix_history")) and Path(h).parent.name == S.name:
    args["mix_history"] = str(S / Path(h).name)
want = mx.numerics(mx.parse(mx.train_args(S, r)))
got = mx.numerics(args)
tail = lambda v: [str(Path(p).parent.name) + "/" + Path(p).name for p in v.split(",")]
diff = [k for k in want if k not in ("mix_months", "mix_stores") and want[k] != got[k]]
diff += ["mix_months"] * (tail(want["mix_months"]) != tail(got["mix_months"]))
assert not diff, f"{r['name']}: config differs from the plan in {diff}"
print(
    f"{r['name']}: hand-back matches the plan ({json.loads((out / 'orchard.json').read_text())})"
)
if "--check" in flags:
    sys.exit()
owner = out / "owner.json"
if not owner.exists():
    assert not (mx.ROOT / "results/lm-eval" / r["name"]).exists()
    mx.write(owner, json.dumps(mx.owner(S, r), indent=1) + "\n")
mx.seconds_left = lambda: 4 * 3600
os.environ.setdefault("SLURM_JOB_ID", "local")
assert mx.run_one(S, r, os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
print(
    json.dumps(
        json.loads((S / "results" / f"{r['name']}.json").read_text())["strat"]["macro"]
    )
)
