"""The source files a checkpoint records the sha256 of: everything that defines the model and its training.

Torch-free, so experiments.modelexp can hash a frozen copy of the package without importing torch.
"""

import hashlib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
HASHED = ("model/*.py", "train/*.py", "data/packed.py", "data/mix.py", "data/vocab.py")


def source_hashes(package=PACKAGE):
    """{path relative to the allie package: sha256} of the hashed files of `package` (an allie directory)."""
    package = Path(package)
    files = sorted({f for g in HASHED for f in package.glob(g)})
    return {f.relative_to(package).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
