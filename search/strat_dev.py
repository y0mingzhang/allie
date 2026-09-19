"""Balanced development from unused July games; exact golden reconstruction guard.

Reuses frozen training tokenization and golden cell/mask/packing rules. Start each
bucket's development slice AFTER the largest golden prefix for either side's cell.
Reconstruct the golden arrays in memory and require exact equality before publishing
anything. Golden files are read-only. Test splits are read only for exclusion IDs.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'
BASE=Path('/data/group_data/dei-group/yimingz3/allie')
SOURCE=BASE/'results/recipe10x/data-v1-round3g/source-ours'
sys.path.insert(0,str(SOURCE))
import chessmix as cm
from chess_vocab import BOS

GOLD=BASE/'strat-eval-v1'
OUT=ROOT/'strat-dev-v1'
MONTH=BASE/'data-v1/2026-07'
HISTORY=[BASE/'data-v1-hist'/f'2024-{m:02d}' for m in range(4,8)]
CONFIG=BASE/'results/pretrain/r2-3e16-control-t20-w20-pf052h-s42/config.json'
UPPER=[1400,2000,2400,10000]


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def selection(g,sites,skip):
    ok=g.rated.astype(bool)&~g.leak&np.array([s not in skip for s in sites])
    ok&=(g.fmt>=1)&(g.fmt<=4)
    base=(g.fmt.astype(int)-1)*4
    cell=lambda elo,bot:np.where(ok&~bot,base+np.searchsorted(UPPER,elo,side='right'),-1)
    return cell(g.welo,g.wbot),cell(g.belo,g.bbot)


def packed(records,sides=False):
    rows=[];labs=[];clocks=[];features=[];meta=[]
    token_parts=[];label_parts=[];clock_parts=[];feat_parts=[];size=0
    def flush():
        nonlocal token_parts,label_parts,clock_parts,feat_parts,size
        rows.append(np.pad(np.concatenate(token_parts),(0,cm.ROW-size),constant_values=BOS))
        labs.append(np.pad(np.concatenate(label_parts),(0,cm.ROW-size),constant_values=-1))
        if sides:
            clocks.append(np.pad(np.concatenate(clock_parts),(0,cm.ROW-size),constant_values=0))
            features.append(np.pad(np.concatenate(feat_parts),((0,cm.ROW-size),(0,0)),constant_values=-1))
        token_parts=[];label_parts=[];clock_parts=[];feat_parts=[];size=0
    for record in records:
        tokens,labels=record['tokens'][:cm.ROW],record['labels'][:cm.ROW]
        if size+len(tokens)>cm.ROW and size:flush()
        if sides:
            meta.append(dict(game=record['game'],row=len(rows),start=size,length=len(tokens),
                token_sha256=hashlib.sha256(np.asarray(tokens,np.int16).tobytes()).hexdigest()))
            clock_parts.append(record['clocks'][:cm.ROW]);feat_parts.append(record['feats'][:cm.ROW])
        token_parts.append(tokens);label_parts.append(labels);size+=len(tokens)
    if size:flush()
    result=dict(rows=np.array(rows,np.int16),labels=np.array(labs,np.int8))
    if sides:result.update(clocks=np.array(clocks,np.int16),feats=np.array(features,np.int32),games=meta)
    return result


def main():
    assert not OUT.exists(),'Do not overwrite a prepared development inventory'
    manifest=json.loads((GOLD/'manifest.json').read_text())
    config=json.loads(CONFIG.read_text())
    assert config['args']['mix_stores']==str(BASE/'data-v1'), 'Historical fallback must be unseen by this checkpoint'
    assert all((p/'stats.json').exists() and (p/'buckets.json').exists() for p in HISTORY)
    assert sha(GOLD/'strat.npz')==manifest['sha256']
    skip=set();exclusions={}
    for split in ('dev','dev_expert','test','test_expert'):
        p=Path('/home/yimingz3/src/allie/data')/(split+'.jsonl')
        exclusions[p.name]=sha(p)
        for line in p.open():skip.add(json.loads(line)['game-id'].rsplit('/',1)[-1])
    golden_frac=np.array(manifest['frac'])
    dev_frac=np.minimum(1.,25000/np.maximum(np.array(manifest['population_moves']),1))
    buckets=[b for b in json.loads((MONTH/'buckets.json').read_text()) if 1<=b['code']//10000-1<=4]

    def build_bucket(b):
        fmt=b['code']//10000-1
        grid=cm.Grid(b['code'])
        possible=np.unique(np.r_[(fmt-1)*4+np.searchsorted(UPPER,grid.welo,side='right'),
                                  (fmt-1)*4+np.searchsorted(UPPER,grid.belo,side='right')])
        gt=np.ceil(golden_frac*b['games']).astype(int)
        dt=np.ceil(dev_frac*b['games']).astype(int)
        offset=int(gt[possible].max())
        need=min(b['games'],offset+int(dt[possible].max()))
        gold=[];dev=[];seen=0
        for shard in b['shards']:
            if seen>=need:break
            p=MONTH/shard
            cols=list(dict.fromkeys(cm.COLUMNS+cm.CLOCK_COLUMNS+['site']))
            table=cm.read(p,cols)
            g=cm.Games(table,clock=True)
            sites=table.column('site').to_pylist()
            wc,bc=selection(g,sites,skip)
            idx=np.arange(seen,seen+g.n)
            gw=(wc>=0)&(idx<gt[np.maximum(wc,0)])
            gb=(bc>=0)&(idx<gt[np.maximum(bc,0)])
            dw=(wc>=0)&(idx>=offset)&(idx<offset+dt[np.maximum(wc,0)])
            db=(bc>=0)&(idx>=offset)&(idx<offset+dt[np.maximum(bc,0)])
            assert not ((gw|gb)&(dw|db)).any(),'Cross-cell game overlap'
            for label,out,ws,bs in [('gold',gold,gw,gb),('dev',dev,dw,db)]:
                for i in np.flatnonzero(ws|bs):
                    t,_=g.tokens(i,True,True);lab=np.full(len(t),-1,np.int8)
                    if ws[i]:lab[11:len(t)-1:2]=wc[i]
                    if bs[i]:lab[12:len(t)-1:2]=bc[i]
                    record=dict(tokens=t,labels=lab,game=sites[i])
                    if label=='dev':record.update(clocks=g.clock(i,len(t),drop=False),feats=g.feats(i,len(t),drop=False))
                    out.append(record)
            seen+=g.n
        return gold,dev,dict(code=b['code'],offset=offset,scanned=min(seen,b['games']),games=b['games'])

    golden=[];development=[];offsets=[]
    with ThreadPoolExecutor(4) as executor:
        for j,(g,d,detail) in enumerate(executor.map(build_bucket,buckets)):
            golden.extend(g);development.extend(d);offsets.append(detail)
            if j%32==0:print('strat dev buckets',j+1,'/',len(buckets),'games',len(development),flush=True)
    old=packed(golden)
    with np.load(GOLD/'strat.npz') as z:
        np.testing.assert_array_equal(old['rows'],z['rows'])
        np.testing.assert_array_equal(old['labels'],z['labels'])
    gids={r['game'] for r in golden};dids={r['game'] for r in development}
    assert len(golden)==manifest['games'] and not gids&dids and not dids&skip
    gh={hashlib.sha256(np.asarray(r['tokens'][:cm.ROW],np.int16).tobytes()).hexdigest() for r in golden}
    dh={hashlib.sha256(np.asarray(r['tokens'][:cm.ROW],np.int16).tobytes()).hexdigest() for r in development}
    assert not gh&dh,'Duplicate game tokens across development and golden'
    # July's golden fraction for classical experts is 1.0: no disjoint July
    # games remain. This checkpoint never sampled data-v1-hist. Use only that
    # cell from four unseen months; this is a declared temporal mismatch.
    assert golden_frac[15]==1.0
    history_sources=[];historical=[];duplicate_tokens=0
    for month in HISTORY:
        bp=month/'buckets.json';history_sources.append(bp)
        for bucket in json.loads(bp.read_text()):
            if bucket['code']//10000-1!=4:continue
            grid=cm.Grid(bucket['code'])
            if max(grid.welo.max(),grid.belo.max())<2400:continue
            for shard in bucket['shards']:
                table=cm.read(month/shard,list(dict.fromkeys(cm.COLUMNS+cm.CLOCK_COLUMNS+['site'])))
                g=cm.Games(table,clock=True);sites=table.column('site').to_pylist()
                wc,bc=selection(g,sites,skip|gids|dids)
                for i in np.flatnonzero((wc==15)|(bc==15)):
                    t,_=g.tokens(i,True,True)
                    h=hashlib.sha256(np.asarray(t[:cm.ROW],np.int16).tobytes()).hexdigest()
                    if h in gh or h in dh:duplicate_tokens+=1;continue
                    lab=np.full(len(t),-1,np.int8)
                    if wc[i]==15:lab[11:len(t)-1:2]=15
                    if bc[i]==15:lab[12:len(t)-1:2]=15
                    historical.append(dict(tokens=t,labels=lab,game=sites[i],
                        clocks=g.clock(i,len(t),drop=False),feats=g.feats(i,len(t),drop=False)))
                    dh.add(h);dids.add(sites[i])
        print('Historical expert classical',month.name,'games',len(historical),flush=True)
    development.extend(historical)
    assert not gids&dids and not dids&skip and not gh&dh
    data=packed(development,True)
    counts=np.bincount(data['labels'][data['labels']>=0],minlength=16)
    assert (counts>=2048).all(),('Insufficient development support',counts.tolist())
    stage=OUT.with_name(OUT.name+'.partial');stage.mkdir()
    for name,keys in [('strat',('rows','labels')),('clocks',('clocks',)),('feats',('feats',))]:
        path=stage/(name+'.npz');temp=path.with_suffix('.partial')
        with temp.open('wb') as f:np.savez(f,**{k:data[k] for k in keys})
        temp.replace(path)
    (stage/'games.json').write_text(json.dumps(data['games'])+'\n')
    proof=dict(source=str(MONTH),cells=manifest['cells'],games=len(development),rows=len(data['rows']),
        scored_moves=counts.tolist(),target=25000,frac=dev_frac.tolist(),offsets=offsets,
        selection='Per-bucket offset past maximum golden prefix for either side; then golden-style per-cell prefix sampling. Whole-game disjointness checked, not just scored-move disjointness.',
        golden_arrays_reconstructed_exactly=True,golden_sha256=manifest['sha256'],excluded_split_sha256=exclusions,
        checkpoint_specific=True,checkpoint_config_sha256=sha(CONFIG),
        expert_classical=dict(sources=list(map(str,HISTORY)),games=len(historical),
            duplicate_tokens_excluded=duplicate_tokens,
            caveat='2024-04..07 replaces exhausted July2026 expert-classical cell only. Temporal mismatch. Not automatically valid for other checkpoints.'),
        golden_game_ids_sha256=hashlib.sha256(json.dumps(sorted(gids)).encode()).hexdigest(),
        source_sha256={str(p):sha(p) for p in [Path(__file__),SOURCE/'chessmix.py',SOURCE/'chess_vocab.py',MONTH/'buckets.json',*history_sources]},
        files_sha256={p.name:sha(p) for p in stage.iterdir()})
    (stage/'manifest.json').write_text(json.dumps(proof,indent=2)+'\n')
    stage.replace(OUT)
    print('PASS exact golden reconstruction, whole-game/site/token/test disjointness; dev counts',counts.tolist(),flush=True)


if __name__=='__main__':main()
