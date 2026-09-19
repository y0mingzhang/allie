"""Two node-day run of our recipe on 8 L40S at the IsoFLOP compute-optimal size.

freeze             copy sources and evaluator into the study, write plan.json
task bench         60-step throughput and memory check, writes bench.json
task train STEPS   resumable training, then scoring on the original validation set
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/yimingz3/src/allie")
STUDY = ROOT / "results/recipe10x/bigrun-v1"
DATA = "/scratch/yimingz3/allie/lichess_tokens_v2"
OLD_EVAL = ROOT / "results/recipe10x/isoflop-v1/evaluator-ours"
SHAPE = dict(layers=28, width=1792, head_dim=64)
MICRO = 4  # micro-batch 8 runs out of memory in the first forward on 44 GiB L40S
TOKENS_PER_STEP = 512 * 1024
SCHEDULE = dict(
    warmup_steps=32,
    mtp_steps=64,
    split_step=65,
    batch_rows=512,
    plateau=4.0,
    final_lr=0.2,
    decay_shape="linear",
)


def freeze():
    assert not STUDY.exists(), "never overwrite a frozen study"
    for d in ("source-ours", "evaluator-ours", "logs"):
        (STUDY / d).mkdir(parents=True)
    src = ROOT / "scripts"
    for f in [*src.glob("modded_*.py"), src / "lm_data.py", src / "lm_checkpoint.py"]:
        shutil.copy2(f, STUDY / "source-ours")
    for f in ("eval_modded.py", "ce_alignment.py", "modded_runtime.py"):
        shutil.copy2(OLD_EVAL / f, STUDY / "evaluator-ours")
    shutil.copy2(__file__, STUDY / "bigrun.py")
    (STUDY / "plan.json").write_text(
        json.dumps(
            dict(
                shape=SHAPE,
                n_nonembed=12 * SHAPE["width"] ** 2 * SHAPE["layers"],
                micro=MICRO,
                schedule=SCHEDULE,
                gpus="8xL40S",
                data=DATA,
                basis="isoflop-v1 measured optima (N* ~ C^0.55) extrapolated to 2 node-days, ND ~2.9e19",
                approval="user approved a two-day 8xL40S run on 2026-09-18, superseding the cumulative big-pool bookkeeping; prior charges preserved",
            ),
            indent=2,
        )
        + "\n"
    )


def seconds_left():
    left = subprocess.check_output(
        ["squeue", "-h", "-o", "%L", "-j", os.environ["SLURM_JOB_ID"]], text=True
    ).strip()
    days, _, hms = left.rpartition("-")
    seconds = 0
    for x in hms.split(":"):
        seconds = seconds * 60 + int(x)
    return seconds + 86400 * int(days or 0)


def stop_workers(root_pid):
    """SIGTERM the torchrun workers (not the agent) so every rank checkpoints and exits."""
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        try:
            proc = Path(f"/proc/{pid}")
            for task in (proc / "task").iterdir():
                pending.extend(map(int, (task / "children").read_text().split()))
            argv = (proc / "cmdline").read_bytes().split(b"\0")
            parent = next(
                l.split()[1]
                for l in (proc / "status").read_text().splitlines()
                if l.startswith("PPid:")
            )
            if (
                any(a.endswith(b"/modded_train.py") for a in argv)
                and (Path("/proc") / parent / "comm").read_text().strip()
                == "pt_elastic"
            ):
                os.kill(pid, signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError, StopIteration):
            pass


def run(cmd, env, log):
    with open(log, "a") as f:
        return subprocess.run(
            [str(x) for x in cmd], env=env, stdout=f, stderr=subprocess.STDOUT
        )


def task(mode, steps):
    name = "bigrun-v1-bench" if mode == "bench" else "bigrun-v1"
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
        TORCHINDUCTOR_CACHE_DIR="/scratch/yimingz3/allie/bigrun-v1-inductor",
        TRITON_CACHE_DIR="/scratch/yimingz3/allie/bigrun-v1-triton",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
    )
    log = STUDY / "logs" / name
    assert (
        run(
            [sys.executable, ROOT / "scripts/stage_corpus.py"], env, f"{log}.stage.log"
        ).returncode
        == 0
    )
    src = STUDY / "source-ours"
    py = subprocess.check_output(
        [sys.executable, src / "modded_runtime_stage.py"], env=env, text=True
    ).strip()
    out = ROOT / "results/pretrain" / name
    done = out / "done.json"
    if mode == "bench" or not (
        done.exists() and json.loads(done.read_text())["stop_reason"] == "steps"
    ):
        cmd = [
            py,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            src / "modded_train.py",
            "--name",
            name,
            "--width",
            SHAPE["width"],
            "--layers",
            SHAPE["layers"],
            "--head-dim",
            SHAPE["head_dim"],
            "--steps",
            steps,
            "--extension-steps",
            0,
            "--initial-batch-rows",
            512,
            "--micro-batch",
            MICRO,
            "--lr-scale",
            1,
            "--seed",
            42,
            "--eval-every",
            4096 if mode == "train" else 10**7,
            "--checkpoint-every",
            2048 if mode == "train" else 10**7,
            "--keep-checkpoints",
            2,
            "--val-rows",
            1024,
            "--max-seconds",
            max(600, seconds_left() - 2400),
            "--deterministic",
            "--data",
            DATA,
            "--wsd-schedule",
            json.dumps(SCHEDULE, sort_keys=True),
            "--wsd-end-step",
            steps,
            "--wsd-decay-start",
            32,
        ]
        if mode == "bench":
            cmd += ["--stop-after", 150]
        elif (out / "last.pt").exists():
            cmd += ["--resume", out / "last.pt"]
        with open(f"{log}.train.log", "a") as f:
            proc = subprocess.Popen(
                [str(x) for x in cmd],
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        signal.signal(signal.SIGUSR1, lambda *_: stop_workers(proc.pid))
        signal.signal(signal.SIGTERM, lambda *_: stop_workers(proc.pid))
        while proc.poll() is None:
            time.sleep(5)
        assert proc.returncode == 0, f"training exited {proc.returncode}"
    if mode == "bench":
        rows = [
            json.loads(l)
            for l in open(f"{log}.train.log")
            if l.startswith('{"step"') and "tokens_per_second" in l
        ]
        # Only after the step-65 embedding split and auxiliary-loss switch-off.
        tps = sorted(r["tokens_per_second"] for r in rows if r["step"] >= 100)[1]
        (STUDY / "bench.json").write_text(
            json.dumps(
                dict(
                    tokens_per_second=tps,
                    max_memory_gb=rows[-1]["max_memory_gb"],
                    seconds_per_step=TOKENS_PER_STEP / tps,
                    steps_in_45h=int(tps * 45 * 3600 / TOKENS_PER_STEP),
                ),
                indent=2,
            )
            + "\n"
        )
        return
    if json.loads(done.read_text())["stop_reason"] != "steps":
        subprocess.run(["scontrol", "requeue", os.environ["SLURM_JOB_ID"]], check=True)
        return
    report = ROOT / "results/lm-eval" / name / "original-val.json"
    if not report.exists():
        env["CUDA_VISIBLE_DEVICES"] = "0"
        assert (
            run(
                [
                    py,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=1",
                    STUDY / "evaluator-ours/eval_modded.py",
                    "--checkpoint",
                    out / "last.pt",
                    "--source",
                    src,
                    "--split",
                    "original_val",
                    "--batch",
                    16,
                ],
                env,
                f"{log}.eval.log",
            ).returncode
            == 0
        )
    ev = json.loads(report.read_text())
    (STUDY / "result.json").write_text(
        json.dumps(
            dict(
                steps=steps,
                tokens=int(steps) * TOKENS_PER_STEP,
                report=str(report),
                ce={k: ev[k + "_ce"] for k in ("move", "expert2400", "expert2600")},
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    if sys.argv[1] == "freeze":
        freeze()
    else:
        task(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 51500)
