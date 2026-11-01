"""Storage roots, overridable by environment: ALLIE_DATA holds the game stores, evaluation sets and benchmark
files; ALLIE_PROJECT_ROOT (default: this checkout) holds results/: studies, runs and scores; ALLIE_SCRATCH holds
what a job can rebuild, best on node-local disk: compiled encoders, compiler caches, staged runtimes, local runs."""

import os
from pathlib import Path

DATA = Path(os.environ.get("ALLIE_DATA", "/data/group_data/dei-group/yimingz3/allie"))
ROOT = Path(os.environ.get("ALLIE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SCRATCH = Path(os.environ.get("ALLIE_SCRATCH", Path.home() / ".cache/allie"))
# the Maia-3 inference code (github.com/CSSLab/maia-chess), for the benchmark's Maia-3 scorer
MAIA3_REPO = Path(os.environ.get("MAIA3_REPO", DATA / "maia3-bench/maia3-repo"))
