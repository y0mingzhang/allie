"""Count move-target rating cohorts in the original training corpus, on CPU."""
import concurrent.futures
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
from modded_train import ratings

DATA = Path('/data/group_data/dei-group/yimingz3/allie/lichess_tokens_v2')
OUT = Path('/data/group_data/dei-group/yimingz3/allie/results/corpus-rating-counts')
EDGES = [0, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 10000]
SOURCE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
RATINGS_SHA = hashlib.sha256(inspect.getsource(ratings).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name+f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def count_file(entry):
    path = DATA/entry['path']
    output = OUT/(entry['path'].replace('/', '_')+'.json')
    if output.exists():
        previous = json.loads(output.read_text())
        assert previous['source_sha256'] == entry['sha256'] and previous['rating_edges'] == EDGES
        assert previous['counter_source_sha256'] == SOURCE_SHA and previous['ratings_source_sha256'] == RATINGS_SHA
        return previous
    assert path.stat().st_size == entry['size']
    rows = np.load(path, mmap_mode='r').reshape(-1, 1025)
    assert rows.dtype == np.uint16
    histogram = np.zeros(len(EDGES)-1, dtype=np.int64)
    move_count = 0
    for lo in range(0, len(rows), 256):
        batch = np.asarray(rows[lo:lo+256], dtype=np.int64)
        assert np.all(batch[:, 0] == 2348), 'Rating reconstruction requires the original BOS-aligned rows'
        y = batch[:, 1:]
        valid = (y >= 378) & (y < 2346)
        elo = ratings(batch)[valid]
        assert np.all((elo >= EDGES[0]) & (elo < EDGES[-1]))
        histogram += np.histogram(elo, bins=EDGES)[0]
        move_count += int(valid.sum())
    assert int(histogram.sum()) == move_count
    result = dict(source=entry['path'], source_sha256=entry['sha256'], rows=len(rows),
        counter_source_sha256=SOURCE_SHA, ratings_source_sha256=RATINGS_SHA,
        input_tokens=len(rows)*1024, move_targets=move_count, rating_edges=EDGES,
        move_counts=histogram.tolist())
    atomic_json(output, result)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((DATA/'manifest.json').read_text())
    assert manifest['revision'] == '20a899ddf344ccaea74e273509a60e5a511125f8'
    entries = [e for e in manifest['files'] if Path(e['path']).name == 'train.npy']
    assert len(entries) == 100
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(count_file, entry) for entry in entries]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
            print(json.dumps(dict(completed_shards=len(records), total_shards=len(entries))), flush=True)
    counts = np.sum([r['move_counts'] for r in records], axis=0)
    total = int(counts.sum())
    report = dict(complete=True, dataset_revision=manifest['revision'], train_shards=len(records),
        train_rows=sum(r['rows'] for r in records), input_tokens=sum(r['input_tokens'] for r in records),
        move_targets=total, rating_edges=EDGES, move_counts=counts.tolist(),
        move_fractions=(counts/total).tolist(),
        expert2400_move_targets=int(counts[6:].sum()), expert2600_move_targets=int(counts[7:].sum()),
        labeling='Same target-side rating function as modded_train.evaluate; original move IDs378..2345.',
        limitations='Counts exposures in one original packed-corpus pass, not deduplicated games or independent examples. '
            'No training-mixture changes; validation and final test are not read. '
            'Uses previously SHA-verified durable corpus; checks sizes and records manifest hashes without rehashing all data.',
        source_sha256=SOURCE_SHA, ratings_source_sha256=RATINGS_SHA)
    assert report['train_rows'] == 54_368_123
    atomic_json(OUT/'summary.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
