"""Reference code loaded from commits before the package layout (git show REV:scripts/...) imports its siblings by
their flat names (modded_smoe, modded_shard, ...), which then resolved to the current scripts/ next to it. alias()
maps those names to the current package modules, the same pairing."""

import importlib
import sys

FLAT = {
    "modded_arch": "allie.model.arch",
    "modded_smoe": "allie.model.moe_kernels",
    "chess_vocab": "allie.data.vocab",
}


def alias():
    for flat, module in FLAT.items():
        sys.modules.setdefault(flat, importlib.import_module(module))
