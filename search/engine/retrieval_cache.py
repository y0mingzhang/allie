"""Verified durable feature shards -> a reusable node-local mmap cache."""
import hashlib
import io
import json
import os
import time
from pathlib import Path
import numpy as np
from .balanced_eval import digest


def read(bank):
    start = time.monotonic()
    bank = Path(bank)
    report_path = bank/'results.json'
    report = json.loads(report_path.read_text())
    key = digest(report_path)
    root = Path('/scratch/yimingz3/allie/search-retrieval')/key
    created = not (root/'manifest.json').exists()
    if created:
        root.parent.mkdir(parents=True, exist_ok=True)
        stage = root.with_name(key+'.partial-'+str(os.getpid()))
        stage.mkdir()
        arrays = {}
        at = 0
        for name, sha in report['shards'].items():
            # One NFS read supplies both the checksum and decoded arrays.
            raw = (bank/name).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == sha
            with np.load(io.BytesIO(raw)) as f:
                n = len(f['target'])
                for k in ('hidden', 'target', 'cell', 'game_ix', 'position'):
                    x = f[k]
                    if k not in arrays:
                        arrays[k] = np.lib.format.open_memmap(stage/(k+'.npy'), mode='w+', dtype=x.dtype,
                            shape=(report['positions'], *x.shape[1:]))
                    arrays[k][at:at+n] = x
                at += n
        assert at == report['positions']
        for a in arrays.values():
            a.flush()
        del arrays
        manifest = dict(source_report_sha256=key, source_plan_sha256=report['plan_sha256'],
            files={p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in stage.glob('*.npy')},
            creation_seconds=time.monotonic()-start,
            semantics='All source shards hash-verified during copy. Scratch is only a cache; durable original shards remain authoritative. Warm cache checks source identity and sizes, not rehashing multi-GB arrays on every query.')
        (stage/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        stage.replace(root)
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['source_report_sha256'] == key
    assert all((root/name).stat().st_size == info['bytes'] for name, info in manifest['files'].items())
    arrays = {Path(name).stem: np.load(root/name, mmap_mode='r') for name in manifest['files']}
    assert arrays['hidden'].shape == (report['positions'], 512)
    return arrays, dict(created=created, seconds=time.monotonic()-start, path=str(root), manifest=manifest)
