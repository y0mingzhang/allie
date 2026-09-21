"""IsoFLOP grid: ours vs faithful chess-v2 Qwen at fixed N*D, N = non-embedding params.

plan    freeze sources, assign runs to chips to minimize time to last result, write sbatch
submit  submit both arrays once
task    run one array task: stage data, train or resume, evaluate, publish result
status  one line per run
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/yimingz3/src/allie")
STUDY = ROOT / "results/recipe10x/isoflop-v1"
DATA = "/scratch/yimingz3/allie/lichess_tokens_v2"
DEPS = "/data/group_data/dei-group/yimingz3/allie/envs/qwen-reproduction-deps-v1/site-packages"
OLD_EVAL = ROOT / "results/recipe10x/tiny-scaling-v1/ours-l8-w128-s2048/evaluator"
GRID = {
    3e16: [(8, 256), (16, 256), (8, 512), (12, 512)],
    1e17: [(16, 256), (12, 512), (12, 768), (24, 768)],
    3e17: [(8, 512), (16, 512), (16, 768), (16, 1024), (32, 1024)],
}
ROWS = dict(ours=512, qwen=448)
OURS_SCHEDULE = dict(
    warmup_steps=32,
    mtp_steps=64,
    split_step=65,
    batch_rows=512,
    plateau=4.0,
    final_lr=0.2,
    decay_shape="linear",
)
# chess-v2 finished branch: 100 warmup, 66550 stable, 30250 cosine decay to 1% of 5e-3.
QWEN_SCHEDULE = dict(warmup=100, lr=0.005, decay_frac=30250 / 96900, min_lr_frac=0.01)
CLASSES = dict(
    general=dict(
        partition="general",
        qos="normal",
        gpu="L40S",
        slots=8,
        slowdown=1.0,
        extra="#SBATCH --exclude=babel-o5-20,babel-o5-24,babel-n5-32,babel-q5-32\n",
    ),
    dei=dict(
        partition="dei-group",
        qos="dei_group_qos",
        gpu="A6000",
        slots=8,
        slowdown=1.4,
        extra="",
    ),
)


def write(path, value):
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def runs():
    out = []
    for budget, shapes in GRID.items():
        for layers, width in shapes:
            n = 12 * width * width * layers
            for recipe in ("ours", "qwen"):
                steps = round(budget / n / (ROWS[recipe] * 1024))
                # Measured single-L40S step time on the tiny grid; qwen ~1.7x per row.
                cost = 1.0 if recipe == "ours" else 1.7 * ROWS["qwen"] / 512
                out.append(
                    dict(
                        name=f"iso-{recipe}-l{layers}-w{width}-{budget:.0e}".replace(
                            "+", ""
                        ),
                        recipe=recipe,
                        layers=layers,
                        width=width,
                        budget=budget,
                        n_nonembed=n,
                        steps=steps,
                        tokens=steps * ROWS[recipe] * 1024,
                        micro=16 if width <= 512 else 8,
                        hours=steps * (0.8 + 0.029 * n / 1e6) * cost / 3600,
                    )
                )
    return out


def assign(rs):
    slots = [(0.0, c) for c, v in CLASSES.items() for _ in range(v["slots"])]
    queues = {c: [] for c in CLASSES}
    for r in sorted(rs, key=lambda r: -r["hours"]):
        finish = [t + r["hours"] * CLASSES[c]["slowdown"] for t, c in slots]
        i = min(range(len(slots)), key=finish.__getitem__)
        slots[i] = (finish[i], slots[i][1])
        queues[slots[i][1]].append(r["name"])
    return queues, max(t for t, _ in slots)


def plan():
    assert not STUDY.exists(), "never overwrite a prepared study"
    for d in (
        "source-ours",
        "source-qwen",
        "evaluator-ours",
        "logs",
        "results",
        "exports",
    ):
        (STUDY / d).mkdir(parents=True)
    src = ROOT / "scripts"
    for f in [*src.glob("modded_*.py"), src / "lm_data.py", src / "lm_checkpoint.py"]:
        shutil.copy2(f, STUDY / "source-ours")
    for f in (
        "train_native_distributed.py",
        "scaled_native_qwen.py",
        "historical_qwen_runtime.py",
        "native_fp32_vector_adam.py",
        "lm_data.py",
        "lm_checkpoint.py",
        "export_distributed_native.py",
        "eval_scaled_native.py",
    ):
        shutil.copy2(src / f, STUDY / "source-qwen")
    for f in ("eval_modded.py", "ce_alignment.py", "modded_runtime.py"):
        shutil.copy2(OLD_EVAL / f, STUDY / "evaluator-ours")
    shutil.copy2(__file__, STUDY / "isoflop.py")
    rs = runs()
    queues, eta = assign(rs)
    write(
        STUDY / "plan.json",
        dict(
            grid={f"{b:.0e}": s for b, s in GRID.items()},
            rows=ROWS,
            ours_schedule=OURS_SCHEDULE,
            qwen_schedule=QWEN_SCHEDULE,
            classes=CLASSES,
            queues=queues,
            eta_hours=eta,
            runs=rs,
            n_definition="12*width^2*layers non-embedding; total counts recorded per result",
        ),
    )
    for c, v in CLASSES.items():
        longest = max(r["hours"] for r in rs if r["name"] in queues[c]) * v["slowdown"]
        (STUDY / f"{c}.sbatch").write_text(f"""#!/bin/bash
#SBATCH --job-name=iso-{c}
#SBATCH --account=dippolit
#SBATCH --partition={v["partition"]}
#SBATCH --qos={v["qos"]}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:{v["gpu"]}:1
#SBATCH --time={min(48, int(2 * longest) + 2)}:00:00
#SBATCH --array=0-{len(queues[c]) - 1}%{v["slots"]}
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output={STUDY}/logs/%x-%A_%a.out
{v["extra"]}exec {ROOT}/.venv/bin/python {STUDY}/isoflop.py task {c}
""")
    print(json.dumps(dict(queues=queues, eta_hours=round(eta, 1)), indent=2))


def submit():
    assert not (STUDY / "submitted.json").exists(), "already submitted"
    ids = {
        c: subprocess.check_output(
            ["sbatch", "--parsable", str(STUDY / f"{c}.sbatch")], text=True
        ).strip()
        for c in CLASSES
    }
    write(STUDY / "submitted.json", dict(at=time.time(), jobs=ids))
    print(ids)


def sh(cmd, env, log):
    with open(log, "a") as f:
        subprocess.run(
            [str(x) for x in cmd],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )


def task(c):
    p = json.loads((STUDY / "plan.json").read_text())
    name = p["queues"][c][int(os.environ["SLURM_ARRAY_TASK_ID"])]
    r = next(x for x in p["runs"] if x["name"] == name)
    result = STUDY / "results" / f"{name}.json"
    if result.exists():
        return
    started = time.monotonic()
    left = subprocess.check_output(
        [
            "squeue",
            "-h",
            "-o",
            "%L",
            "-j",
            f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}",
        ],
        text=True,
    ).strip()
    days, _, hms = left.rpartition("-")
    seconds = 0
    for x in hms.split(":"):
        seconds = seconds * 60 + int(x) if x.isdigit() else seconds
    seconds += 86400 * int(days or 0)
    env = os.environ | dict(
        ALLIE_PROJECT_ROOT=str(ROOT),
        OMP_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
        TORCHINDUCTOR_COMPILE_THREADS="4",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        CUDA_MODULE_LOADING="LAZY",
        DISABLE_FP8="1",
        NCCL_CUMEM_HOST_ENABLE="0",
        NCCL_IB_DISABLE="1",
        NCCL_P2P_DISABLE="1",
        TORCHINDUCTOR_CACHE_DIR="/scratch/yimingz3/allie/isoflop-v1-inductor",
        TRITON_CACHE_DIR="/scratch/yimingz3/allie/isoflop-v1-triton",
    )
    if r["recipe"] == "qwen":
        env["PYTHONPATH"] = DEPS
    log = STUDY / "logs" / name
    sh([sys.executable, ROOT / "scripts/stage_corpus.py"], env, f"{log}.stage.log")
    run = ROOT / "results/pretrain" / name
    done = run / "done.json"
    if not (done.exists() and json.loads(done.read_text())["stop_reason"] == "steps"):
        seconds = str(max(600, seconds - int(time.monotonic() - started) - 900))
        resume = (run / "last.pt").exists()
        if r["recipe"] == "ours":
            src = STUDY / "source-ours"
            py = subprocess.check_output(
                [sys.executable, src / "modded_runtime_stage.py"], env=env, text=True
            ).strip()
            cmd = [
                py,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=1",
                src / "modded_train.py",
                "--name",
                name,
                "--width",
                r["width"],
                "--layers",
                r["layers"],
                "--head-dim",
                64,
                "--steps",
                r["steps"],
                "--extension-steps",
                0,
                "--initial-batch-rows",
                512,
                "--micro-batch",
                r["micro"],
                "--lr-scale",
                1,
                "--seed",
                42,
                "--eval-every",
                10**7,
                "--checkpoint-every",
                256,
                "--keep-checkpoints",
                2,
                "--val-rows",
                1024,
                "--max-seconds",
                seconds,
                "--deterministic",
                "--data",
                DATA,
                "--wsd-schedule",
                json.dumps(OURS_SCHEDULE, sort_keys=True),
                "--wsd-end-step",
                r["steps"],
                "--wsd-decay-start",
                32,
            ]
            if resume:
                cmd += ["--resume", run / "last.pt"]
        else:
            if run.exists() and not resume:
                shutil.rmtree(run)  # crashed before its first checkpoint
            shape = dict(
                layers=r["layers"],
                width=r["width"],
                ff=3 * r["width"],
                heads=r["width"] // 128,
                kv_heads=r["width"] // 256,
                head_dim=128,
            )
            cmd = [
                ROOT / ".venv/bin/python",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=1",
                STUDY / "source-qwen/train_native_distributed.py",
                "--out",
                run,
                "--steps",
                r["steps"],
                "--warmup",
                QWEN_SCHEDULE["warmup"],
                "--lr",
                QWEN_SCHEDULE["lr"],
                "--schedule",
                "wsd",
                "--decay-frac",
                QWEN_SCHEDULE["decay_frac"],
                "--min-lr-frac",
                QWEN_SCHEDULE["min_lr_frac"],
                "--global-rows",
                ROWS["qwen"],
                "--micro",
                r["micro"],
                "--seed",
                42,
                "--checkpoint-every",
                256,
                "--max-seconds",
                seconds,
                "--data",
                DATA,
                "--shape-json",
                json.dumps(shape),
            ]
            if resume:
                cmd += ["--resume", run / "last.pt"]
        sh(cmd, env, f"{log}.train.log")
        if json.loads(done.read_text())["stop_reason"] != "steps":
            aid = f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}"
            subprocess.run(["scontrol", "requeue", aid], check=True)
            return
    report = ROOT / "results/lm-eval" / name / "original-val.json"
    if not report.exists():
        shutil.rmtree(report.parent, ignore_errors=True)
        if r["recipe"] == "ours":
            src = STUDY / "source-ours"
            py = subprocess.check_output(
                [sys.executable, src / "modded_runtime_stage.py"], env=env, text=True
            ).strip()
            sh(
                [
                    py,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=1",
                    STUDY / "evaluator-ours/eval_modded.py",
                    "--checkpoint",
                    run / "last.pt",
                    "--source",
                    src,
                    "--split",
                    "original_val",
                    "--batch",
                    16,
                ],
                env,
                f"{log}.eval.log",
            )
        else:
            export = STUDY / "exports" / name
            if not (export / "identity.json").exists():
                shutil.rmtree(export, ignore_errors=True)
                sh(
                    [
                        ROOT / ".venv/bin/python",
                        STUDY / "source-qwen/export_distributed_native.py",
                        "--checkpoint",
                        run / "last.pt",
                        "--out",
                        export,
                    ],
                    env,
                    f"{log}.eval.log",
                )
            sh(
                [
                    ROOT / ".venv/bin/python",
                    STUDY / "source-qwen/eval_scaled_native.py",
                    "--identity",
                    export / "identity.json",
                    "--out",
                    report.parent,
                    "--rows",
                    5371,
                    "--batch",
                    8,
                ],
                env,
                f"{log}.eval.log",
            )
    ev = json.loads(report.read_text())
    config = run / ("config.json" if r["recipe"] == "ours" else "metadata.json")
    assert [ev[k + "_count"] for k in ("move", "expert2400", "expert2600")] == [
        4616637,
        396483,
        151660,
    ]
    write(
        result,
        r
        | dict(
            parameters=json.loads(config.read_text())["parameters"],
            ce={k: ev[k + "_ce"] for k in ("move", "expert2400", "expert2600")},
            report=str(report),
            job=os.environ["SLURM_JOB_ID"],
            node=os.uname().nodename,
            gpu=CLASSES[c]["gpu"],
            task_seconds=time.monotonic() - started,
        ),
    )


def step_of(r):
    prog = ROOT / "results/pretrain" / r["name"] / "progress.json"
    if prog.exists():
        return json.loads(prog.read_text())["step"]
    log = STUDY / "logs" / f"{r['name']}.train.log"
    steps = (
        [
            json.loads(l)["step"]
            for l in log.read_text().splitlines()
            if l.startswith('{"step"') and "train_ce" in l
        ]
        if log.exists()
        else []
    )
    return steps[-1] if steps else 0


def bar(frac, width=30):
    full = int(frac * width * 8)
    return (
        "█" * (full // 8)
        + (" ▏▎▍▌▋▊▉"[full % 8] if full < width * 8 else "")
        + " " * (width - full // 8 - 1)
    )


def status():
    p = json.loads((STUDY / "plan.json").read_text())
    done_h = 0.0
    for b in sorted({r["budget"] for r in p["runs"]}):
        print(f"ND={b:.0e}")
        for r in sorted(
            (r for r in p["runs"] if r["budget"] == b),
            key=lambda r: (r["n_nonembed"], r["recipe"]),
        ):
            res = STUDY / "results" / f"{r['name']}.json"
            if res.exists():
                frac, tail = (
                    1.0,
                    "move %.4f" % json.loads(res.read_text())["ce"]["move"],
                )
            else:
                step = step_of(r)
                frac = step / r["steps"]
                tail = f"{step}/{r['steps']}" + (" eval" if step == r["steps"] else "")
            done_h += frac * r["hours"]
            print(
                f"  {r['recipe']:4} {r['layers']:2d}x{r['width']:<4d} |{bar(frac)}| {100 * frac:5.1f}%  {tail}"
            )
    total = sum(r["hours"] for r in p["runs"])
    n = len(list((STUDY / "results").glob("*.json")))
    print(
        f"\nall  {n}/{len(p['runs'])} done |{bar(done_h / total, 40)}| {100 * done_h / total:5.1f}% of planned GPU-hours"
    )


if __name__ == "__main__":
    dict(plan=plan, submit=submit, status=status, task=lambda: task(sys.argv[2]))[
        sys.argv[1]
    ]()
