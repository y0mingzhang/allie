"""freeze.py LABEL [WAVE...]: write results/recipe10x/<study>/round.py = round.py.template with RECIPE = RECIPES[LABEL]
and plan each wave (default: the three budgets) from COMMIT. Never submits; never overwrites a frozen study."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/yimingz3/src/allie")
HERE = Path(__file__).resolve().parent
COMMIT = "890aa38"  # branch sweep
RECIPES = dict(c8s200f0v4="table:c8s200f0v4-fcfbf8858a28", control="control")
STUDY = dict(sw1e17="sweep-1e17", sw3e17="sweep-3e17", sw1e18="sweep-1e18", swchk3="sweep-check3", swchkh="sweep-checkh")

label, *waves = sys.argv[1:]
text = (HERE / "round.py.template").read_text()
text, n = re.subn(r"^RECIPE = None .*$", f"RECIPE = {(label, RECIPES[label])!r}", text, flags=re.M)
assert n == 1
for w in waves or ("sw1e17", "sw3e17", "sw1e18"):
    study = ROOT / "results/recipe10x" / f"{STUDY[w]}-{label}"
    assert not (study / "plan.json").exists(), f"{study} is frozen"
    study.mkdir(exist_ok=True)
    (study / "round.py").write_text(text)
    mx = ROOT / ".worktrees/sweep/scripts/modelexp.py"
    subprocess.run([ROOT / ".venv/bin/python", mx, "plan", study / "round.py", w, COMMIT], check=True)
    print(study, "planned")
