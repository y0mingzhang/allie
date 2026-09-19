"""Legal commuting-history variants with identical present game state.

Windows must end before an unchanged pawn move/capture. Changed repetition
history is then outside the irreversible boundary. FEN, entire suffix legality
and automatic termination are checked, not just piece placement.
"""
import json
from pathlib import Path
import time
import numpy as np
from .balanced_eval import ROOT,GLOBAL_STOP,atomic,digest
from .native_board import Position,from_prefix,MOVE_ID


def variants(prefix,maximum=4,lookback=24):
    header=list(prefix[:11]);moves=list(prefix[11:]);states=[Position()];irreversible=-1
    for i,token in enumerate(moves):
        states.append(states[-1].child(token))
        if int(states[-1].fen().split()[4])==0:irreversible=i
    final=states[-1].fen();out=[];seen={tuple(prefix)}
    for end in range(irreversible,max(1,irreversible-lookback),-1):
        for width in (3,4):
            start=end-width
            if start<0:continue
            span=moves[start:end];replacement=[span[2],span[1],span[0]] if width==3 else span[2:]+span[:2]
            candidate=moves[:start]+replacement+moves[end:];key=tuple(header+candidate)
            if key in seen:continue
            b=states[start]
            try:
                good=True
                for token in replacement:
                    b=b.child(token)
                    if b.outcome()>=0:good=False;break
                if not good or b.fen()!=states[end].fen():continue
                # A changed earlier intermediate position must not cause an
                # automatic draw along the suffix before the irreversible move.
                for token in moves[end:]:
                    b=b.child(token)
                    if b.outcome()>=0:good=False;break
                if not good:continue
            except ValueError:
                continue
            assert b.fen()==final and len(key)==len(prefix) and key[:11]==tuple(header)
            assert end<=irreversible
            out.append(dict(prefix=list(key),start=start,end=end,irreversible_move=irreversible));seen.add(key)
            if len(out)==maximum:return out
    return out


def build():
    out=ROOT/'aug-transpositions-v1';out.mkdir(exist_ok=True);path=out/'variants.json'
    sample=ROOT/'aug-tune-v1/sample.json'
    if path.exists():
        data=json.loads(path.read_text());assert data['sample_sha256']==digest(sample) and data['generator_sha256']==digest(Path(__file__));return data
    start=time.monotonic();rows=json.loads(sample.read_text())['positions'];items=[]
    for i,r in enumerate(rows):
        v=variants(r['prefix']);items.append(v)
        if (i+1)%512==0:print('Transposition generation',i+1,'/',len(rows),flush=True)
    counts=np.array(list(map(len,items)));cells=np.array([r['cell'] for r in rows])
    data=dict(variants=items,sample_sha256=digest(sample),generator_sha256=digest(Path(__file__)),
        positions=len(rows),seconds=time.monotonic()-start,mean_variants=float(counts.mean()),
        covered_fraction=float(np.mean(counts>0)),cell_covered_fraction=[float(np.mean(counts[cells==c]>0)) for c in range(16)],
        stage='August only, deterministic generation from prefix alone. Full FEN matches at window endpoint; unchanged irreversible suffix and all suffix outcomes checked. Human history dependence is not assumed away; this is a tested smoothing intervention.')
    atomic(path,data);return data


def run(oracle,spec):
    data=build();out=ROOT/'aug-transpositions-v1';sample=ROOT/'aug-tune-v1/sample.json';rows=json.loads(sample.read_text())['positions'];bs=512
    plan=dict(sample_sha256=digest(sample),variants_sha256=digest(out/'variants.json'),source_sha256=digest(Path(__file__)),
        direct_sha256=digest(Path(__file__).with_name('direct.py')),roots_per_batch=bs,
        stage='Root-only original and up to4 legal history variants. Full-prefix queries/tokens charged, no observed future or target in generation. Missing alternatives explicitly fall back to original.')
    p=out/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic(p,plan)
    begin=time.monotonic();all_stats=[]
    for lo in range(0,len(rows),bs):
        dest=out/f'{lo:06d}.npz'
        if dest.exists():
            with np.load(dest) as z:all_stats.append(json.loads(str(z['stats'])))
            continue
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        part=rows[lo:lo+bs];n=len(part);k=max(len(r['legal']) for r in part);queries=[];owners=[]
        for i,r in enumerate(part):
            queries.append(r['prefix']);owners.append((i,0))
            for j,v in enumerate(data['variants'][lo+i],1):queries.append(v['prefix']);owners.append((i,j))
        oracle.reset();start=time.monotonic();z=oracle(queries);scores=np.zeros((n,5,k));counts=np.ones(n,int)
        for q,(i,j) in enumerate(owners):scores[i,j,:len(part[i]['legal'])]=z[q,part[i]['legal']];counts[i]=max(counts[i],j+1)
        stats=dict(seconds=time.monotonic()-start,new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds,
            original_queries=n,extra_full_prefix_queries=len(queries)-n)
        tmp=dest.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,logits=scores,counts=counts,game=np.array([r['game'] for r in part]),stats=json.dumps(stats))
        tmp.replace(dest);all_stats.append(stats);print('Transposition inference',lo+n,'/',len(rows),stats,flush=True)
    result=dict(positions=len(rows),seconds=time.monotonic()-begin,cost=all_stats)
    atomic(out/'worker.json',result);return result


def test():
    header=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions'][0]['prefix'][:11]
    prefix=header+[MOVE_ID[x] for x in ('g1f3','g8f6','b1c3','b8c6','a2a3')]
    found=variants(prefix);assert found
    original=from_prefix(prefix)
    for v in found:
        assert from_prefix(v['prefix']).fen()==original.fen()
        assert from_prefix(v['prefix']).legal()==original.legal()
        assert v['end']<=v['irreversible_move'] and v['prefix'][:11]==prefix[:11]
    # No later irreversible move means no admissible changed history.
    assert not variants(prefix[:-1])
    print('PASS legal commute, exact final FEN/legal moves and irreversible-boundary requirement',flush=True)


if __name__=='__main__':test();build()
