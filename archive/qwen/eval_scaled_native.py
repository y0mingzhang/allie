"""Aligned HF scoring for scaled native checkpoints; original validation only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

ROOT = Path(os.environ.get('ALLIE_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
DURABLE = Path('/data/group_data/dei-group/yimingz3/allie')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def ratings(rows):
    # Same target-side rating parser as both frozen training evaluators.
    n, length = rows.shape
    pos = np.arange(length)[None, :]
    starts = np.maximum.accumulate(np.where(rows == 2348, pos, 0), axis=1)
    ar = np.arange(n)[:, None]
    white = sum(rows[ar, np.minimum(starts+k, length-1)]*p for k, p in zip(range(3, 7), (1000, 100, 10, 1)))
    black = sum(rows[ar, np.minimum(starts+k, length-1)]*p for k, p in zip(range(7, 11), (1000, 100, 10, 1)))
    return np.where((pos-starts-11) % 2 == 0, white, black)[:, 1:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--identity', required=True)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--rows', type=int, choices=(8,5371), default=5371)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    identity_path = Path(a.identity)
    identity = json.loads(identity_path.read_text())
    assert identity['format'] == 'scaled-native-export-v1'
    assert identity['all_tensor_payloads_exact']
    assert identity['tokens'] == identity['step']*identity['source_metadata']['recipe']['global_rows']*1024
    model_path = Path(identity['path'])
    for name, expected in identity['files_sha256'].items():
        assert digest(model_path/name) == expected, name
    cache = DURABLE/'validation_cache'
    manifest = json.loads((cache/'manifest.json').read_text())
    assert manifest['revision'] == '20a899ddf344ccaea74e273509a60e5a511125f8'
    assert manifest['rows'] == 5371 and len(manifest['original_files']) == 100
    assert digest(cache/manifest['derived_file']) == manifest['sha256']
    rows = np.load(cache/manifest['derived_file'], mmap_mode='r').reshape(-1, 1025)
    assert len(rows) == 5371
    rows = rows[:a.rows]
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    parameters = sum(x.numel() for x in model.parameters())
    assert parameters == identity['parameters']
    totals = np.zeros((len(rows), 6), np.float64)
    with torch.inference_mode():
        for lo in range(0, len(rows), a.batch):
            data = np.asarray(rows[lo:lo+a.batch], dtype=np.int64)
            x = torch.as_tensor(data[:, :-1], device='cuda')
            y = torch.as_tensor(data[:, 1:], device='cuda')
            scores = model(x, use_cache=False).logits[..., 378:2346].float()
            nll = scores.logsumexp(-1)-scores.gather(-1, (y-378).clamp(0, 1967)[..., None]).squeeze(-1)
            valid = (y >= 378) & (y < 2346)
            elo = torch.as_tensor(ratings(data), device='cuda')
            for j, mask in enumerate((valid, valid & (elo >= 2400), valid & (elo >= 2600))):
                totals[lo:lo+len(data), 2*j] = (nll.double()*mask).sum(-1).cpu().numpy()
                totals[lo:lo+len(data), 2*j+1] = mask.sum(-1).cpu().numpy()
            if lo % (a.batch*64) == 0:
                print(json.dumps(dict(rows_done=lo+len(data), seconds=time.monotonic()-started)), flush=True)
    assert np.isfinite(totals).all()
    np.savez(out/'original-val-rows.npz', row=np.arange(len(rows)),
             nll_sums=totals[:, ::2], counts=totals[:, 1::2])
    sums = totals.sum(0)
    report = dict(model=str(model_path), model_identity=identity,
                  model_identity_sha256=digest(identity_path), parameters=parameters,
                  rows=len(rows), dataset_revision=manifest['revision'],
                  validation_cache_sha256=manifest['sha256'],
                  normalization='1968 original move IDs378..2345; no legal masking',
                  forward_protocol='BF16 Qwen SDPA causal full original rows, use_cache=False',
                  inference_torch=torch.__version__, batch=a.batch,
                  gpu=torch.cuda.get_device_name(), job_id=os.environ.get('SLURM_JOB_ID'),
                  evaluator_sha256=digest(__file__), seconds=time.monotonic()-started,
                  final_test_accessed=False)
    for j, key in enumerate(('move', 'expert2400', 'expert2600')):
        report[key+'_ce'] = sums[2*j]/sums[2*j+1]
        report[key+'_count'] = int(sums[2*j+1])
    if a.rows == 5371:
        assert [report[k+'_count'] for k in ('move','expert2400','expert2600')] == [4616637,396483,151660]
    report['row_statistics_sha256'] = digest(out/'original-val-rows.npz')
    (out/'original-val.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
