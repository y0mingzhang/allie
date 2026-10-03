"""prune() keeps final.pt's directory (modded_train --final-model-only): a rerun publishes a second final checkpoint of
the same step whose random id may sort below the first one's, so step order alone would delete the current target.
CPU only:

    python scripts/test_prune_final.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modded_checkpoints import prune, publish  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        log = out / "checkpoints.jsonl"
        dirs = [
            "step-00000002-aaaaaaaa",
            "step-00000003-ffffffff",
            "step-00000003-eeeeeeee",
            "step-00000003-00000000",
        ]
        for name in dirs:
            d = out / "checkpoints" / name
            d.mkdir(parents=True)
            (d / "model.pt").write_bytes(b"")
            with log.open("a") as f:
                f.write(json.dumps({"directory": f"checkpoints/{name}"}) + "\n")
        publish(out, out / "checkpoints" / dirs[0], 2, 1, ["last.pt"])
        for keep in (2, 3):
            current = (
                out / "checkpoints" / dirs[-1]
            )  # the rerun's final save, lowest id of the step-3 ones
            publish(out, current, 3, 1, ["final.pt"])
            prune(out, keep, current)
            left = {p.name for p in (out / "checkpoints").iterdir()}
            assert {dirs[0], dirs[-1]} <= left, (keep, left)
    print("ok")


if __name__ == "__main__":
    main()
