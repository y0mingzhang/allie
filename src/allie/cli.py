"""Command-line entry points: allie-train (a config file to a torchrun of allie.train.trainer) and allie-eval
(allie.eval.score on one GPU). Experiments run through allie-exp (allie.experiments.modelexp)."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from allie import paths

# the environment experiments.modelexp runs the trainer and evaluator in; the caller's values win
ENV = dict(
    OMP_NUM_THREADS="4",
    TORCHINDUCTOR_COMPILE_THREADS="4",
    CUBLAS_WORKSPACE_CONFIG=":4096:8",
    CUDA_MODULE_LOADING="LAZY",
    NCCL_CUMEM_HOST_ENABLE="0",
    NCCL_IB_DISABLE="1",
    NCCL_P2P_DISABLE="1",
    PYTHONUNBUFFERED="1",
)


def flags(config_path):
    """allie.train.trainer flags of a config file (configs/*.json): ALLIE_DATA-relative data paths resolved,
    mix_history relative to the config's directory."""
    config_path = Path(config_path).resolve()
    t = json.loads(config_path.read_text())["trainer"]
    t["data"] = str(paths.DATA / t["data"])
    t["mix_stores"] = ",".join(str(paths.DATA / s) for s in t["mix_stores"])
    t["mix_months"] = ",".join(str(paths.DATA / m) for m in t["mix_months"])
    t["mix_history"] = str(config_path.parent / t["mix_history"])
    out = []
    for k, v in t.items():
        if v is False or v is None:
            continue
        out.append("--" + k.replace("_", "-"))
        if v is not True:
            out.append(json.dumps(v, sort_keys=True) if isinstance(v, dict) else str(v))
    return out


def torchrun(nproc, target, args, env):
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
    ]
    return subprocess.call(
        [*cmd, "-m", target, *map(str, args)], env=ENV | os.environ | env
    )


def train():
    p = argparse.ArgumentParser(
        description="Train from a config file; resumes the run's last checkpoint."
    )
    p.add_argument("config", help="e.g. configs/allie-2.0.json")
    p.add_argument("--name", help="run name (default: the config's)")
    p.add_argument("--nproc", type=int, default=8, help="GPUs on this node")
    a, extra = p.parse_known_args()
    config = Path(a.config).resolve()
    name = a.name or json.loads(config.read_text())["name"]
    args = ["--name", name, *flags(config), *extra]
    last = paths.ROOT / "results/pretrain" / name / "last.pt"
    if last.exists() and not any(
        x == "--resume" or x.startswith("--resume=") for x in extra
    ):
        args += ["--resume", last]
    # the config's mix table, unless the environment names another recipe directory
    env = {
        "ALLIE_RECIPES": os.environ.get("ALLIE_RECIPES", str(config.parent / "recipes"))
    }
    sys.exit(torchrun(a.nproc, "allie.train.trainer", args, env))


def evaluate():
    p = argparse.ArgumentParser(
        description="Main evaluation (or original validation) of a checkpoint, one GPU."
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="a run's last.pt (or another checkpoint pointer)",
    )
    p.add_argument("--split", choices=["strat", "original_val"], default="strat")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--source", help="a pre-package checkpoint's frozen flat source")
    a = p.parse_args()
    args = ["--checkpoint", a.checkpoint, "--split", a.split, "--batch", a.batch]
    args += ["--source", a.source] * bool(a.source)
    sys.exit(torchrun(1, "allie.eval.score", args, {}))
