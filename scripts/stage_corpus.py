"""Rebuild a node-local cache from the hash-verified durable original corpus."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import time
import tempfile

p=argparse.ArgumentParser()
p.add_argument('--destination',default='/scratch/yimingz3/allie/lichess_tokens_v2')
a=p.parse_args()
source=Path('/data/group_data/dei-group/yimingz3/allie/lichess_tokens_v2')
dest=Path(a.destination);manifest=json.loads((source/'manifest.json').read_text())
assert manifest['revision']=='20a899ddf344ccaea74e273509a60e5a511125f8'
assert len(manifest['files'])==200
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def transfer(entry):
    path=dest/entry['path'];path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size==entry['size'] and digest(path)==entry['sha256']:return
    # Independent Slurm jobs may restore the same node cache concurrently.
    # Each copy owns its temporary file; verified final files are interchangeable.
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.partial',delete=False) as f:
        temp=Path(f.name)
    try:
        shutil.copyfile(source/entry['path'],temp)
        assert temp.stat().st_size==entry['size'] and digest(temp)==entry['sha256'],str(temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
start=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for i,_ in enumerate(pool.map(transfer,manifest['files']),1):
        if i%20==0:print(f'Cache verified {i}/200 files in {time.monotonic()-start:.1f}s',flush=True)
with tempfile.NamedTemporaryFile(mode='w',dir=dest,prefix='manifest-',suffix='.partial',delete=False) as f:
    f.write(json.dumps(manifest,indent=2));manifest_temp=Path(f.name)
manifest_temp.replace(dest/'manifest.json')
print('STAGED',dest,flush=True)
