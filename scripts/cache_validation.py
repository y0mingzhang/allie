"""Concatenate verified original validation shards, preserving exact row order.

This is a derived cache, never a replacement for the original corpus/splits.
One11MB file avoids100 slow metadata/open operations on a cold storage client.
"""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path('/data/group_data/dei-group/yimingz3/allie')
source = ROOT/'lichess_tokens_v2'
manifest = json.loads((source/'manifest.json').read_text())
assert manifest['revision'] == '20a899ddf344ccaea74e273509a60e5a511125f8'
files = sorted((x for x in manifest['files'] if x['path'].endswith('/val.npy')), key=lambda x:x['path'])
assert len(files) == 100
arrays = []
for entry in files:
    path = source/entry['path']
    with path.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == entry['sha256']
        f.seek(0)
        array = np.load(f).reshape(-1, 1025)
        assert array.dtype == np.uint16
        arrays.append(array)
rows = np.concatenate(arrays)
assert rows.shape == (5371, 1025)
out = ROOT/'validation_cache'
(out/'000').mkdir(parents=True, exist_ok=True)
target = out/'000/val.npy'
with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.partial', delete=False) as f:
    temp = Path(f.name)
    np.save(f, rows)
try:
    if target.exists():
        assert np.array_equal(np.load(target), rows), 'Existing derived validation cache differs'
    else:
        temp.replace(target)
finally:
    temp.unlink(missing_ok=True)
with target.open('rb') as f:
    sha = hashlib.file_digest(f, 'sha256').hexdigest()
report = dict(revision=manifest['revision'], row_order='lexicographically sorted original val.npy paths, then original row order',
              original_files=files, derived_file='000/val.npy', rows=len(rows), sha256=sha)
(out/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(rows=len(rows), sha256=sha, cache=str(out))))
