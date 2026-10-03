"""The search tests build the native rules and tree modules: they need pybind11 and the chess-library headers
(ALLIE_CHESS_INCLUDE, else vendor/chess-library/include in this checkout)."""

import importlib.util
import os
from pathlib import Path

include = Path(
    os.environ.get(
        "ALLIE_CHESS_INCLUDE",
        Path(__file__).resolve().parents[2] / "vendor/chess-library/include",
    )
)
collect_ignore_glob = (
    []
    if importlib.util.find_spec("pybind11") and (include / "chess.hpp").exists()
    else ["test_*.py"]
)
