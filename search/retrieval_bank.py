"""Temporally clean retrieval-bank pilot from June2026, with game exclusion.

Reads split moves only to make exclusion fingerprints, never as bank targets or
selection feedback. No golden/test losses or predictions are read.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
from . import strat_dev as common

OUT=common.ROOT/'retrieval-bank-v3'
MONTH=common.BASE/'data-v1/2026-06'

def move_hash(tokens):
    a=np.asarray(tokens,np.int16)
    a=a[(a>=378)&(a<2346)]
    return hashlib.sha256(a.tobytes()).hexdigest()

def main():
    assert not OUT.exists(),'Do not overwrite the bank'
    from .engine.native_board import MOVES
    mapping={m:378+i for i,m in enumerate(MOVES)}
    golden=json.loads((common.GOLD/'manifest.json').read_text())
    assert common.sha(common.GOLD/'strat.npz')==golden['sha256']
    blocked=set();skip=set();inputs={}
    for split in ('dev','dev_expert','test','test_expert'):
        p=Path('/home/yimingz3/src/allie/data')/(split+'.jsonl');inputs[str(p)]=common.sha(p)
        for line in p.open():
            r=json.loads(line);skip.add(r['game-id'].rsplit('/',1)[-1])
            moves=[mapping[m] for m in r['human-move']]
            blocked.add(move_hash(moves));blocked.add(move_hash(moves[:1014]))
    # Complete documents and conservative truncated-document fingerprints. Ignore
    # headers/outcome markers so a duplicate with changed metadata is also removed.
    for p in (common.GOLD/'strat.npz',common.ROOT/'aug-tune-v1/strat.npz'):
        inputs[str(p)]=common.sha(p)
        with np.load(p) as z:
            for row in z['rows']:
                starts=np.flatnonzero(row==common.BOS)
                for a,b in zip(starts,np.r_[starts[1:],len(row)]):
                    if b-a>=12:blocked.add(move_hash(row[a+11:b]))
    p=common.ROOT/'aug-tune-v1/games.json';inputs[str(p)]=common.sha(p)
    skip.update(r['game'] for r in json.loads(p.read_text()))
    frac=np.minimum(1.,32000/np.maximum(np.array(golden['population_moves']),1))
    buckets=[b for b in json.loads((MONTH/'buckets.json').read_text()) if 1<=b['code']//10000-1<=4]
    def build(b):
        fmt=b['code']//10000-1;grid=common.cm.Grid(b['code'])
        possible=np.unique(np.r_[(fmt-1)*4+np.searchsorted(common.UPPER,grid.welo,side='right'),
                                 (fmt-1)*4+np.searchsorted(common.UPPER,grid.belo,side='right')])
        take=np.ceil(frac*b['games']).astype(int);need=int(take[possible].max())
        records=[];seen=0;excluded=0;paths=[]
        for shard in b['shards']:
            if seen>=need:break
            p=MONTH/shard;paths.append(str(p))
            table=common.cm.read(p,common.cm.COLUMNS+['site'])
            g=common.cm.Games(table);sites=table.column('site').to_pylist()
            wc,bc=common.selection(g,sites,skip);idx=np.arange(seen,seen+g.n)
            ws=(wc>=0)&(idx<take[np.maximum(wc,0)]);bs=(bc>=0)&(idx<take[np.maximum(bc,0)])
            for i in np.flatnonzero(ws|bs):
                t,_=g.tokens(i,True,True)
                h=move_hash(t[11:]);hp=move_hash(t[11:1025])
                if h in blocked or hp in blocked:excluded+=1;continue
                t=t[:1025].astype(np.int16);lab=np.full(len(t),-1,np.int8)
                last=len(t)-1 if t[-1] in (2346,2347) else len(t)
                if ws[i]:lab[11:last:2]=wc[i]
                if bs[i]:lab[12:last:2]=bc[i]
                records.append(dict(tokens=t,labels=lab,game=sites[i],move_sha256=h))
            seen+=g.n
        return records,excluded,paths
    records=[];excluded=0;read_paths=[]
    with ThreadPoolExecutor(2) as ex:
        for j,(part,n,paths) in enumerate(ex.map(build,buckets)):
            records.extend(part);excluded+=n;read_paths.extend(paths)
            if j%128==0:print('Retrieval bank buckets',j+1,'/',len(buckets),'games',len(records),flush=True)
    unique=[];seen=set();deduplicated=0
    for r in records:
        if r['move_sha256'] in seen:deduplicated+=1;continue
        seen.add(r['move_sha256']);unique.append(r)
    assert not seen&blocked and not {r['game'] for r in unique}&skip
    lengths=np.array([len(r['tokens']) for r in unique],np.int32)
    offsets=np.r_[0,np.cumsum(lengths)].astype(np.int64)
    tokens=np.concatenate([r['tokens'] for r in unique]);labels=np.concatenate([r['labels'] for r in unique])
    before=np.bincount(labels[labels>=0],minlength=16)
    rng=np.random.default_rng(1926759)
    for c in range(16):
        idx=np.flatnonzero(labels==c)
        if len(idx)>32000:labels[rng.choice(idx,len(idx)-32000,replace=False)]=-1
    counts=np.bincount(labels[labels>=0],minlength=16);assert (counts>0).all()
    assert ((tokens[labels>=0]>=378)&(tokens[labels>=0]<2346)).all()
    stage=OUT.with_name(OUT.name+'.partial');stage.mkdir()
    with (stage/'bank.npz').open('wb') as f:
        np.savez_compressed(f,tokens=tokens,labels=labels,offsets=offsets,game=np.array([r['game'] for r in unique]),
            move_sha256=np.array([r['move_sha256'] for r in unique]))
    inputs[str(MONTH/'buckets.json')]=common.sha(MONTH/'buckets.json')
    for p in [Path(__file__),Path(common.__file__),common.SOURCE/'chessmix.py',common.SOURCE/'chess_vocab.py']:inputs[str(p)]=common.sha(p)
    proof=dict(source=str(MONTH),latest_month='2026-06',target_per_cell=32000,seed=1926759,
        games=len(unique),positions=counts.tolist(),positions_before_cap=before.tolist(),
        excluded_duplicate_documents=excluded,deduplicated_bank_games=deduplicated,
        blocked_move_hashes=len(blocked),excluded_game_ids=len(skip),input_sha256=inputs,read_shards=read_paths,
        bank_sha256=common.sha(stage/'bank.npz'),
        semantics='Retrieval augmentation using additional training-corpus examples, not guaranteed to have been sampled by the checkpoint. Strictly before July golden. Same16-cell labels, bot movers and val leaks masked. Excludes dev/test/golden/August tuning duplicate move documents independent of headers. Test game moves used ONLY for exclusion fingerprints, never for fitting/scoring. Bank size/build/query memory and cost must accompany search nodes and any conditional training-equivalent CM.')
    (stage/'manifest.json').write_text(json.dumps(proof,indent=2)+'\n');stage.replace(OUT)
    print('Bank ready',counts.tolist(),'games',len(unique),'excluded',excluded,'deduplicated',deduplicated,flush=True)

if __name__=='__main__':main()
