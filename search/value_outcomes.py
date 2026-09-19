"""August-only final-result sidecar for value-head calibration diagnostics."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from . import strat_dev as common


def main():
    out=common.ROOT/'aug-value-outcomes-v1';assert not out.exists()
    sample=common.ROOT/'aug-tune-expanded-v1/sample.json'
    rows=json.loads(sample.read_text())['positions'];wanted={r['site'] for r in rows}
    inventory=common.ROOT/'aug-tune-v1'
    manifest=json.loads((inventory/'manifest.json').read_text());month=Path(manifest['source']);frac=np.array(manifest['frac'])
    paths=[]
    for b in json.loads((month/'buckets.json').read_text()):
        fmt=b['code']//10000-1
        if not 1<=fmt<=4:continue
        grid=common.cm.Grid(b['code'])
        possible=np.unique(np.r_[(fmt-1)*4+np.searchsorted(common.UPPER,grid.welo,side='right'),
            (fmt-1)*4+np.searchsorted(common.UPPER,grid.belo,side='right')])
        need=int(np.ceil(frac*b['games'])[possible].max());seen=0
        for shard in b['shards']:
            if seen>=need:break
            p=month/shard;paths.append(p);seen+=pq.ParquetFile(p).metadata.num_rows
    choices=pa.array(sorted(wanted))
    def scan(path):
        table=pq.ParquetFile(path).read(columns=['site','result'])
        return table.filter(pc.is_in(table['site'],value_set=choices)).to_pylist()
    found={}
    with ThreadPoolExecutor(2) as pool:
        for part in pool.map(scan,paths):
            for r in part:
                assert r['site'] not in found or found[r['site']]==r['result']
                found[r['site']]=r['result']
    assert not wanted-set(found)
    result=np.array([found[r['site']] for r in rows]);assert np.isin(result,[0,1,2]).all()
    side=np.array([(len(r['prefix'])-11)%2 for r in rows])
    wdl=np.array([[0,2],[2,0],[1,1]])[result,side]
    stage=out.with_name(out.name+'.partial');stage.mkdir()
    np.savez_compressed(stage/'labels.npz',result=result,mover_wdl=wdl,game=[r['game'] for r in rows],site=[r['site'] for r in rows],ply=[r['ply'] for r in rows])
    report=dict(sample_sha256=common.sha(sample),source_sha256=common.sha(Path(__file__)),labels_sha256=common.sha(stage/'labels.npz'),
        source_month=str(month),read_shards=[str(p) for p in paths],positions=len(rows),games=len(wanted),
        encoding='Result0 white win,1 black win,2draw; root next-mover class0win/1draw/2loss. Side from (prefix_length-11)%2. Same as frozen trainer aux target convention.',
        semantics='Future game outcomes used ONLY as AUGUST supervised calibration labels, never prediction inputs. July golden outcomes are neither loaded nor used. Same game-disjoint fit/CV/confirmation assignment as all previous August experiments; potentially model-training-seen.')
    (stage/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');stage.replace(out)
    print('Outcome sidecar',len(wanted),'games',len(rows),'positions',len(paths),'shards',flush=True)

if __name__=='__main__':main()
