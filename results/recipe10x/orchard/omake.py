"""Run on babel: freeze a planned study for the orchard lane. Writes <study>/orchard/args.json (each run's exact
train_args from the study's frozen modelexp.py, and its GPU count) and mirrors every file plan.json hashes (plan,
source-ours, evaluator-ours, modelexp.py, modded_arch.py, history counts) to orchard:~/allie/studies/<study>/.
Usage: omake.py <study name under results/recipe10x, or a path> --gpus N [--step-seconds S: first-leg estimate for the ~15 min checkpoint cadence] [run name ...]"""

import hashlib, json, subprocess, sys
from pathlib import Path

import oargs

R = Path("/home/yimingz3/src/allie/results/recipe10x")
argv = sys.argv[1:]
gpus = int(argv.pop(argv.index("--gpus") + 1))
argv.remove("--gpus")
sps = float(argv.pop(argv.index("--step-seconds") + 1)) if "--step-seconds" in argv else None
if sps:
    argv.remove("--step-seconds")
study = R / argv[0] if "/" not in argv[0] else Path(argv[0]).resolve()
sys.path[:0] = [str(study), str(study / "source-ours")]
import modelexp as mx

plan = json.loads((study / "plan.json").read_text())
runs = [r for r in plan["runs"] if len(argv) < 2 or r["name"] in argv[1:]]
args = {
    r["name"]: dict(args=list(map(str, mx.train_args(study, r))), gpus=gpus, step_seconds=sps)
    for r in runs
}
for r in runs:
    oargs.check(args[r["name"]]["args"], r)
o = study / "orchard"
o.mkdir(exist_ok=True)
old = (
    json.loads((o / "args.json").read_text())["runs"]
    if (o / "args.json").exists()
    else {}
)
spec = dict(
    study=str(study),
    plan_sha256=hashlib.sha256((study / "plan.json").read_bytes()).hexdigest(),
    store="/data/group_data/dei-group/yimingz3/allie",
    source={},
    runs=old | args,
)
(o / "args.json").write_text(json.dumps(spec, indent=1) + "\n")
files = sorted(
    {k.split("/")[0] for k in plan["hashes"] if (study / k).exists()}
    | {"plan.json", "orchard"}
)
subprocess.run(["ssh", "orchard", f"mkdir -p allie/studies/{study.name}"], check=True)
subprocess.run(
    [
        "rsync",
        "-a",
        "--exclude",
        "__pycache__",
        *[str(study / f) for f in files],
        f"orchard:allie/studies/{study.name}/",
    ],
    check=True,
)
print(
    f"{study.name}: {len(args)} runs frozen for orchard on {gpus} GPUs each ({', '.join(files)})"
)
