"""User-approved August tuning; may be training-seen, never called held-out."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
from . import strat_dev as common


def main():
    out=common.ROOT/'aug-tune-v1';month=common.BASE/'data-v1/2026-08'
    assert not out.exists(),'Immutable tuning inventory'
    golden=json.loads((common.GOLD/'manifest.json').read_text())
    frac=np.minimum(1.,25000/np.maximum(golden['population_moves'],1))
    skip=set();exclusions={}
    for split in ('dev','dev_expert','test','test_expert'):
        path=Path('/home/yimingz3/src/allie/data')/(split+'.jsonl')
        exclusions[path.name]=common.sha(path)
        for line in path.open():skip.add(json.loads(line)['game-id'].rsplit('/',1)[-1])
    # Exclude identical full token documents from golden too; no golden losses read.
    assert common.sha(common.GOLD/'strat.npz')==golden['sha256']
    golden_hashes=set()
    with np.load(common.GOLD/'strat.npz') as z:
        for row in z['rows']:
            starts=np.flatnonzero(row==common.BOS)
            for a,b in zip(starts,np.r_[starts[1:],len(row)]):
                if b-a>=12:golden_hashes.add(hashlib.sha256(row[a:b].astype(np.int16).tobytes()).hexdigest())
    buckets=[b for b in json.loads((month/'buckets.json').read_text()) if 1<=b['code']//10000-1<=4]
    def build(b):
        fmt=b['code']//10000-1;grid=common.cm.Grid(b['code'])
        possible=np.unique(np.r_[(fmt-1)*4+np.searchsorted(common.UPPER,grid.welo,side='right'),
                                (fmt-1)*4+np.searchsorted(common.UPPER,grid.belo,side='right')])
        take=np.ceil(frac*b['games']).astype(int);need=int(take[possible].max())
        records=[];seen=0;excluded=0
        for shard in b['shards']:
            if seen>=need:break
            table=common.cm.read(month/shard,list(dict.fromkeys(common.cm.COLUMNS+common.cm.CLOCK_COLUMNS+['site'])))
            g=common.cm.Games(table,clock=True);sites=table.column('site').to_pylist()
            wc,bc=common.selection(g,sites,skip);idx=np.arange(seen,seen+g.n)
            ws=(wc>=0)&(idx<take[np.maximum(wc,0)]);bs=(bc>=0)&(idx<take[np.maximum(bc,0)])
            for i in np.flatnonzero(ws|bs):
                t,_=g.tokens(i,True,True)
                if hashlib.sha256(np.asarray(t[:common.cm.ROW],np.int16).tobytes()).hexdigest() in golden_hashes:
                    excluded+=1;continue
                lab=np.full(len(t),-1,np.int8)
                if ws[i]:lab[11:len(t)-1:2]=wc[i]
                if bs[i]:lab[12:len(t)-1:2]=bc[i]
                records.append(dict(tokens=t,labels=lab,game=sites[i],clocks=g.clock(i,len(t),drop=False),feats=g.feats(i,len(t),drop=False)))
            seen+=g.n
        return records,excluded
    records=[];excluded=0
    with ThreadPoolExecutor(4) as executor:
        for j,(part,n) in enumerate(executor.map(build,buckets)):
            records.extend(part);excluded+=n
            if j%64==0:print('August tuning',j+1,'/',len(buckets),'games',len(records),flush=True)
    data=common.packed(records,True)
    counts=np.bincount(data['labels'][data['labels']>=0],minlength=16)
    assert (counts>=2048).all(),counts
    assert not {r['game'] for r in records}&skip
    stage=out.with_name(out.name+'.partial');stage.mkdir()
    for name,keys in [('strat',('rows','labels')),('clocks',('clocks',)),('feats',('feats',))]:
        with (stage/(name+'.npz')).open('wb') as f:np.savez(f,**{k:data[k] for k in keys})
    (stage/'games.json').write_text(json.dumps(data['games'])+'\n')
    proof=dict(source=str(month),cells=golden['cells'],games=len(records),rows=len(data['rows']),scored_moves=counts.tolist(),
        target=25000,frac=frac.tolist(),training_seen_possible=True,
        semantics='August2026 tuning, user-approved despite training overlap. Game-disjoint fit/confirmation folds only protect parameter fitting; only July golden establishes held-out model/search gains.',
        golden_sha256=golden['sha256'],golden_duplicate_documents_excluded=excluded,excluded_split_sha256=exclusions,
        source_sha256={str(p):common.sha(p) for p in [Path(__file__),Path(common.__file__),common.SOURCE/'chessmix.py',common.SOURCE/'chess_vocab.py',month/'buckets.json']},
        files_sha256={p.name:common.sha(p) for p in stage.iterdir()})
    (stage/'manifest.json').write_text(json.dumps(proof,indent=2)+'\n');stage.replace(out)
    print('August tuning ready',counts.tolist(),flush=True)


if __name__=='__main__':main()
