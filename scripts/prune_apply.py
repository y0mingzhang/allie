"""Apply results/prune-plan.json: delete the listed heavy files, leave PRUNED.json tombstones.

Skips any owner modified after the plan was written. Run only after the user approves the plan.
"""

import json
import shutil
import time
from pathlib import Path

ROOT = Path("/home/yimingz3/src/allie/results")
REASON = "user 2026-09-18: old runs not planned for re-evaluation are reproducible and can be dropped"


def main():
    plan = json.loads((ROOT / "prune-plan.json").read_text())
    freed = 0
    for e in plan["plan"]:
        owner = ROOT / e["owner"]
        if owner.stat().st_mtime > plan["created"]:
            print(f"skip {e['owner']}: modified after the plan")
            continue
        for rel, _ in e["items"]:
            p = ROOT / rel
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        (owner / "PRUNED.json").write_text(
            json.dumps(
                dict(at=time.time(), reason=REASON, removed=e["items"]), indent=1
            )
            + "\n"
        )
        freed += sum(b for _, b in e["items"])
    print(f"freed {freed / 2**40:.2f} TB")


if __name__ == "__main__":
    main()
