"""The checks in tests/checks/, each run as its own program (they set up process groups, signal handlers and
compiled graphs, so each gets a fresh process). Checks run where their marks allow and skip otherwise: `gpu` ones
need CUDA, `data` ones the game stores under ALLIE_DATA, `history` ones this repository's git history (they
compare against earlier commits). Examples:

    uv run pytest                                 # every check this machine can run
    uv run pytest -m "not gpu and not data"       # CPU checks without data
    uv run pytest -k moe_center                   # one check
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import triton

from allie import paths

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "tests/checks"
# allie.experiments.modelexp reads its baseline study on import
BASELINE = Path(os.environ.get("ALLIE_PROJECT_ROOT", "/home/yimingz3/src/allie"))
BASELINE /= "results/recipe10x/data-v1-B3.json"
HISTORY = ["git", "-C", str(CHECKS), "cat-file", "-e", "90fd74c^{commit}"]

skip = pytest.mark.skipif
GPU = [pytest.mark.gpu, skip(not torch.cuda.is_available(), reason="needs a CUDA GPU")]
TWO = [skip(torch.cuda.device_count() < 2, reason="needs 2 GPUs")]
DATA = [
    pytest.mark.data,
    skip(
        not (paths.DATA / "lichess_tokens_v2").exists(),
        reason="needs the game stores under ALLIE_DATA",
    ),
]
HIST = [
    pytest.mark.history,
    skip(
        subprocess.run(HISTORY, capture_output=True).returncode != 0,
        reason="needs git history",
    ),
]
STUDY = [skip(not BASELINE.exists(), reason=f"needs the baseline study {BASELINE}")]
TRITON36 = [
    skip(
        not triton.__version__.startswith("3.6"),
        reason="checks Triton 3.6, the pinned training runtime's",
    )
]
NOCOMPILE = {"TORCH_COMPILE_DISABLE": "1"}
CONFIG = ROOT / "configs/allie-v3.0/resume-config.json"


def check(name, *args, torchrun=0, env=None, marks=(), id=None):
    return pytest.param(name, args, torchrun, env or {}, marks=marks, id=id or name)


@pytest.mark.parametrize(
    "name, args, torchrun, env",
    [
        check("async_checkpoint"),
        check("center_first"),
        check("modelexp_run", marks=STUDY),
        check("moe_quantile"),
        check("prune_final"),
        check("signal_handlers", marks=TRITON36),
        check("zero_masters"),
        check("moe_center", marks=HIST),
        check("moe_gate_floor", marks=HIST),
        check("moe_keep", marks=HIST),
        check("moe_log_gates", marks=HIST),
        check("moe_nongemm", marks=HIST),
        check("fused_blend", marks=DATA),
        check("game_blocks", env=NOCOMPILE, marks=DATA),
        check("moe_shard", env=NOCOMPILE, marks=DATA),
        # Allie-v3.0's months, stores and history, with its recipe table
        check(
            "mix_recent",
            CONFIG,
            env={"ALLIE_RECIPES": str(CONFIG.parents[1] / "recipes")},
            marks=DATA,
        ),
        check("attn_kernel", marks=GPU + DATA),
        check(
            "checkpoint_resume", torchrun=1, marks=GPU + DATA, id="checkpoint_resume-1"
        ),
        check(
            "checkpoint_resume",
            torchrun=2,
            marks=GPU + TWO + DATA,
            id="checkpoint_resume-2",
        ),
        check("header_feats", torchrun=1, marks=GPU + DATA),
        check("nanogpt_knobs", torchrun=1, marks=GPU + DATA),
        check(
            "moe_nongemm",
            "kernels",
            "topk",
            "counts",
            "seqraw",
            marks=GPU,
            id="moe_nongemm-cuda",
        ),
        check("fused_blend", "cuda", marks=GPU + DATA, id="fused_blend-cuda"),
        check(
            "game_blocks", "--device", "cuda", marks=GPU + DATA, id="game_blocks-cuda"
        ),
    ],
)
def test_check(name, args, torchrun, env):
    cmd = [sys.executable]
    if torchrun:
        cmd += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={torchrun}",
        ]
    pythonpath = os.pathsep.join(
        filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")])
    )
    env = (
        os.environ
        | {"PYTHONPATH": pythonpath, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        | env
    )
    cmd += [str(CHECKS / f"{name}.py"), *map(str, args)]
    subprocess.run(cmd, env=env, check=True, timeout=3600)
