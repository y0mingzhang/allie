"""Score finished runs on the stratified July evaluation (strat-eval-v1).

plan     freeze the evaluators and list finished runs lacking strat-v1.json: isoflop-v1 ours and
         Qwen exports, data-wave runs
submit   submit a single-GPU preempt array over the pending list
task     evaluate one run of the submitted list: ours with its frozen source, Qwen from its export
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/yimingz3/src/allie")
STUDY = ROOT / "results/recipe10x/strat-eval-v1"
ISO = ROOT / "results/recipe10x/isoflop-v1"
OLD_EVAL = ISO / "evaluator-ours"


def finished():
    runs = [
        dict(name=f.stem, kind="ours", source=str(ISO / "source-ours"))
        for f in sorted((ISO / "results").glob("iso-ours-*.json"))
    ]
    runs += [
        dict(
            name=f.stem,
            kind="qwen",
            identity=str(ISO / "exports" / f.stem / "identity.json"),
        )
        for f in sorted((ISO / "results").glob("iso-qwen-*.json"))
    ]
    runs += [
        dict(name=f.stem, kind="ours", source=str(f.parents[1] / "source-ours"))
        for f in sorted(
            (ROOT / "results/recipe10x").glob("data-v1-wave*/results/*.json")
        )
    ]
    busy = inflight()
    return [
        r
        for r in runs
        if r["name"] not in busy
        and not (ROOT / "results/lm-eval" / r["name"] / "strat-v1.json").exists()
    ]


def inflight():
    """Runs listed by submitted arrays that are still queued or running."""
    log = STUDY / "jobs.jsonl"
    jobs = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    known = {Path(j["listing"]).name for j in jobs}
    busy = set()
    for j in jobs:
        q = subprocess.run(
            ["squeue", "-h", "-j", j["job"]], capture_output=True, text=True
        )
        if q.stdout.strip():
            busy |= {r["name"] for r in json.loads(Path(j["listing"]).read_text())}
    for listing in STUDY.glob("runs-*.json"):  # submitted before jobs.jsonl existed
        if (
            listing.name not in known
            and time.time() - listing.stat().st_mtime < 3 * 3600
        ):
            busy |= {r["name"] for r in json.loads(listing.read_text())}
    return busy


def plan():
    ev = STUDY / "evaluator"
    ev.mkdir(parents=True, exist_ok=True)
    (STUDY / "logs").mkdir(exist_ok=True)
    for f in (
        ROOT / "scripts/eval_strat.py",
        ROOT / "scripts/eval_qwen_strat.py",
        OLD_EVAL / "ce_alignment.py",
        OLD_EVAL / "modded_runtime.py",
    ):
        if not (ev / f.name).exists():
            shutil.copy2(f, ev)
    shutil.copy2(__file__, STUDY / "strat_eval_all.py")
    runs = finished()
    (STUDY / "pending.json").write_text(json.dumps(runs, indent=2) + "\n")
    print(f"{len(runs)} runs pending: {[r['name'] for r in runs]}")


def submit():
    runs = json.loads((STUDY / "pending.json").read_text())
    assert runs, "nothing pending"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    listing = STUDY / f"runs-{stamp}.json"
    listing.write_text(json.dumps(runs, indent=2) + "\n")
    script = STUDY / f"run-{stamp}.sbatch"
    script.write_text(f"""#!/bin/bash
#SBATCH --job-name=strat-eval
#SBATCH --account=dippolit
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=babel-x9-32
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-{len(runs) - 1}%4
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output={STUDY}/logs/%x-%A_%a.out
exec {ROOT}/.venv/bin/python {STUDY}/strat_eval_all.py task {listing}
""")
    job = subprocess.check_output(
        ["sbatch", "--parsable", str(script)], text=True
    ).strip()
    with open(STUDY / "jobs.jsonl", "a") as f:
        f.write(json.dumps(dict(listing=str(listing), job=job)) + "\n")
    print(job)


def task(listing):
    r = json.loads(Path(listing).read_text())[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    out = ROOT / "results/lm-eval" / r["name"]
    if (out / "strat-v1.json").exists():
        return
    env = os.environ | dict(
        ALLIE_PROJECT_ROOT=str(ROOT),
        OMP_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
        TORCHINDUCTOR_COMPILE_THREADS="4",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        CUDA_MODULE_LOADING="LAZY",
        DISABLE_FP8="1",
        TORCHINDUCTOR_CACHE_DIR="/scratch/yimingz3/allie/strat-inductor",
        TRITON_CACHE_DIR="/scratch/yimingz3/allie/strat-triton",
    )
    if r.get("kind") == "qwen":  # older listings carry no kind: ours
        cmd = [
            sys.executable,
            STUDY / "evaluator/eval_qwen_strat.py",
            "--identity",
            r["identity"],
            "--out",
            out,
        ]
    else:
        src = Path(r["source"])
        py = subprocess.check_output(
            [sys.executable, src / "modded_runtime_stage.py"], env=env, text=True
        ).strip()
        cmd = [
            py, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
            STUDY / "evaluator/eval_strat.py",
            "--checkpoint", ROOT / "results/pretrain" / r["name"] / "last.pt",
            "--source", src,
            "--split", "strat",
            "--batch", 16,
        ]  # fmt: skip
    with open(STUDY / "logs" / f"{r['name']}.log", "a") as f:
        subprocess.run(
            [str(x) for x in cmd],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )


if __name__ == "__main__":
    dict(plan=plan, submit=submit, task=task)[sys.argv[1]](*sys.argv[2:])
