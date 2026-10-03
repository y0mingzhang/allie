"""eval.score.training_code loads Allie-v3.0 (a pre-package checkpoint) through this package only while the
package is the version its equivalence was tested on; anything else needs the checkpoint's frozen source."""

import json
from pathlib import Path

import pytest

from allie.eval import score

PLAN = Path(__file__).resolve().parents[1] / "configs/allie-v3.0/plan.json"


def allie_v3():
    """The source_sha256 Allie-v3.0's checkpoint records: its frozen source's Python files."""
    hashes = json.loads(PLAN.read_text())["hashes"]
    src = {
        k.removeprefix("source-ours/"): v
        for k, v in hashes.items()
        if k.startswith("source-ours/")
    }
    return {"source_sha256": {k: v for k, v in src.items() if k.endswith(".py")}}


def test_tested_package_loads_allie_v3():
    assert score.training_code(allie_v3())[1].__name__ == "allie.model.network"


def test_changed_package_refuses_allie_v3(monkeypatch):
    changed = score.source_hashes() | {"model/moe.py": "0" * 64}
    monkeypatch.setattr(score, "source_hashes", lambda: changed)
    with pytest.raises(AssertionError, match="another version of this package"):
        score.training_code(allie_v3())


def test_other_flat_checkpoint_needs_its_source():
    state = allie_v3()
    state["source_sha256"]["modded_moe.py"] = "0" * 64
    with pytest.raises(AssertionError, match="--source"):
        score.training_code(state)
