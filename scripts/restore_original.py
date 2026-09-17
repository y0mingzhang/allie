"""Restore the user's original packed training corpus at a pinned revision."""
import json
import os
from pathlib import Path
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import HfApi,snapshot_download

root=Path(__file__).resolve().parents[1]
dest=Path('/scratch/yimingz3/allie/lichess_tokens_v2')
repo='yimingzhang/lichess_tokens_v2'
api=HfApi()
previous=root/'results'/'original-data.json'
revision=json.loads(previous.read_text())['revision'] if previous.exists() else api.dataset_info(repo).sha
manifest=dict(repo=repo,revision=revision,local_path=str(dest),description='Original chess-v2 packed train and validation data; no filtering or retokenization')
(root/'results'/'original-data.json').write_text(json.dumps(manifest,indent=2))
files=json.loads((root/'results'/'original-data-files.json').read_text())
# Restore all small validation files first, then four pilot shards; these are
# unchanged original shards, not a newly selected game/rating distribution.
train=[f['path'] for f in files if f['path'].endswith('/train.npy')]
val=[f['path'] for f in files if f['path'].endswith('/val.npy')]
for patterns in [val,train[:4],train[4:]]:
    print('Downloading',len(patterns),'files',flush=True)
    snapshot_download(repo_id=repo,repo_type='dataset',revision=revision,local_dir=str(dest),allow_patterns=patterns,max_workers=8)
    print('Completed phase',len(patterns),flush=True)
print('Original corpus restored',manifest,flush=True)
