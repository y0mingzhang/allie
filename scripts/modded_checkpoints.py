"""Bound future checkpoint storage while preserving published recovery pointers."""
import json
from pathlib import Path
import re
import shutil

import torch


def prune(out, keep, current):
    if keep == 0:
        return []
    assert keep >= 2
    root = (out/'checkpoints').resolve()

    def checked(path):
        path = path.resolve()
        assert path.parent == root and re.fullmatch(r'step-\d{8}-[0-9a-f]{8}', path.name)
        return path

    protected = set()
    for name in ('last.pt', 'best.pt', *[p.name for p in out.glob('fork-*.pt')]):
        if (out/name).exists():
            pointer = torch.load(out/name, map_location='cpu', weights_only=False)
            assert pointer['format'] == 'allie-modded-medium-1'
            protected.add(checked(out/pointer['directory']))
    known = protected | {checked(current)}
    log = out/'checkpoints.jsonl'
    if log.exists():
        for line in log.read_text().splitlines():
            entry = json.loads(line)
            # Old/unpublished directories are not removed by this policy.
            if 'directory' in entry:
                path = checked(out/entry['directory'])
                if path.is_dir():
                    known.add(path)
    latest = sorted(known, key=lambda p:(int(p.name.split('-')[1]), p.name))[-keep:]
    removed = []
    for path in sorted(known-protected-set(latest)):
        shutil.rmtree(path)
        removed.append(path.name)
    return removed
