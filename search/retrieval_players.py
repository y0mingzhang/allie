"""Hashed participant sidecar for a same-player exclusion ablation; no model labels."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from . import strat_dev as common


def main():
    bank = common.ROOT/'retrieval-bank-large-v1'
    sample = common.ROOT/'aug-tune-expanded-v1/sample.json'
    out = common.ROOT/'retrieval-players-large-v1'
    assert not out.exists()
    manifest = json.loads((bank/'manifest.json').read_text())
    with np.load(bank/'bank.npz') as f:
        bank_sites = f['game'].tolist()
    rows = json.loads(sample.read_text())['positions']
    query_sites = [r['site'] for r in rows]
    wanted = set(bank_sites+query_sites)
    paths = [Path(p) for p in manifest['read_shards']]
    aug = common.ROOT/'aug-tune-v1'
    aug_manifest = json.loads((aug/'manifest.json').read_text())
    month = Path(aug_manifest['source'])
    frac = np.array(aug_manifest['frac'])
    # Reproduce exactly which bucket prefix shards the August inventory scanned.
    for b in json.loads((month/'buckets.json').read_text()):
        fmt = b['code']//10000-1
        if not 1 <= fmt <= 4:
            continue
        grid = common.cm.Grid(b['code'])
        possible = np.unique(np.r_[(fmt-1)*4+np.searchsorted(common.UPPER, grid.welo, side='right'),
            (fmt-1)*4+np.searchsorted(common.UPPER, grid.belo, side='right')])
        need = int(np.ceil(frac*b['games'])[possible].max())
        seen = 0
        for shard in b['shards']:
            if seen >= need:
                break
            p = month/shard
            paths.append(p)
            seen += pq.ParquetFile(p).metadata.num_rows
    choices = pa.array(sorted(wanted))

    def scan(p):
        table = pq.ParquetFile(p).read(columns=['site', 'white', 'black'])
        table = table.filter(pc.is_in(table['site'], value_set=choices))
        return table.to_pylist()

    found = {}
    for part in ThreadPoolExecutor(2).map(scan, list(dict.fromkeys(paths))):
        for r in part:
            value = tuple((r[k] or '').casefold() for k in ('white', 'black'))
            assert r['site'] not in found or found[r['site']] == value
            found[r['site']] = value
    assert not wanted-set(found), ('Missing player metadata', len(wanted-set(found)))
    # Unknown participant ID0 never excludes an unrelated unknown participant.
    reverse = {}

    def player_id(name):
        if not name or name in ('?', 'anonymous'):
            return 0
        value = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little')
        assert value != 0
        assert value not in reverse or reverse[value] == name, 'Player hash collision'
        reverse[value] = name
        return value

    convert = lambda sites: np.array([[player_id(p) for p in found[s]] for s in sites], dtype=np.uint64)
    bank_players, query_players = convert(bank_sites), convert(query_sites)
    stage = out.with_name(out.name+'.partial')
    stage.mkdir()
    with (stage/'players.npz').open('wb') as f:
        np.savez_compressed(f, bank_players=bank_players, query_players=query_players,
            bank_game=bank_sites, query_site=query_sites, query_game=[r['game'] for r in rows], query_ply=[r['ply'] for r in rows])
    report = dict(bank_sha256=common.sha(bank/'bank.npz'), sample_sha256=common.sha(sample),
        source_sha256=common.sha(Path(__file__)), players_sha256=common.sha(stage/'players.npz'),
        bank_games=len(bank_sites), query_positions=len(rows), unique_known_players=len(reverse),
        unknown_bank_participants=int((bank_players==0).sum()), unknown_query_participants=int((query_players==0).sum()),
        semantics='Same-player ablation excludes every neighbor game sharing EITHER participant with the query game. Only participant identity is read; no outcomes, future moves or target choice enter the exclusion. Names are casefolded and SHA256 truncated to64 bits with collision checks; unknown0 never matches. Sidecar private and no player names in reports.')
    (stage/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    stage.replace(out)
    print('Participant sidecar ready', report['bank_games'], report['query_positions'], 'unknown', report['unknown_query_participants'], flush=True)


if __name__ == '__main__':
    main()
