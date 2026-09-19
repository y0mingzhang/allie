"""Past-player one-ply search evidence, prefix only, with explicit query costs."""
import json,time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .service import ROOT,GLOBAL_STOP,atomic
from .balanced_eval import digest
from .native_board import from_prefix
from .handles import HandleOracle

def histories(prefix,limit=8):
    # Current mover's earlier decisions: current ply N shares parity with N-2.
    n=len(prefix)-11
    return [(prefix[:11+j],prefix[11+j]) for j in range(n-2,-1,-2)][:limit]

def test():
    p=list(range(11))+[400,401,402,403,404]
    h=histories(p)
    assert [target for _,target in h]==[403,401]
    assert [len(pre) for pre,_ in h]==[14,12]
    assert not histories(p[:12])
    assert histories(p[:11])==[]
    for pre,target in h:
        assert pre==p[:len(pre)] and target==p[len(pre)] and len(pre)<len(p)
    # Adding the NEXT target is intentionally not part of this API's input.
    print('PASS current-player parity and strictly prefix-only history selection')

def run(oracle,spec):
    test();out=ROOT/'golden-player-search-v1';out.mkdir(exist_ok=True)
    sample=ROOT/'golden-balanced-v1/sample.json';rows=json.loads(sample.read_text())['positions']
    from .golden_player_policy import freeze
    plan=freeze()
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    items=[];lookup={};index=np.full((len(rows),8),-1,int)
    for i,r in enumerate(rows):
        for slot,(prefix,target) in enumerate(histories(r['prefix'])):
            key=(tuple(prefix),target)
            if key not in lookup:
                board=from_prefix(prefix);legal=board.legal();assert target in legal
                if len(legal)==1:continue
                lookup[key]=len(items);items.append(dict(prefix=prefix,target=target,legal=legal))
            at=lookup[key];assert items[at]['target']==target
            index[i,slot]=at
    np.save(out/'history-index.npy',index)
    start=time.monotonic();stats=[]
    for lo in range(0,len(items),plan['roots_per_batch']):
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        path=out/f'history-{lo:06d}.npz';part=items[lo:lo+plan['roots_per_batch']]
        if not path.exists():
            begin=time.monotonic();oracle.reset();n=len(part);bridge=HandleOracle(oracle,[r['prefix'] for r in part])
            roots=bridge.root_logits;k=max(len(r['legal']) for r in part)
            ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool);q=np.zeros((n,k));target=np.zeros(n,int)
            hs=[];locations=[];cost=np.zeros(n,int)
            for i,r in enumerate(part):
                legal=r['legal'];ids[i,:len(legal)]=np.array(legal)-378;mask[i,:len(legal)]=True;target[i]=legal.index(r['target'])
                board=from_prefix(r['prefix'])
                for j,a in enumerate(legal):
                    child=board.child(a);outcome=child.outcome()
                    if outcome>=0:q[i,j]=(2*outcome-1)*(1 if board.white else -1)
                    else:
                        hs.append([n+len(hs),i,a,len(r['prefix'])+1]);locations.append((i,j));cost[i]+=1
            if hs:
                z=bridge(hs);wdl=softmax(z[:,2413:2416].astype(float),axis=1)
                values=wdl[:,2]-wdl[:,0]  # child next-mover perspective, negate for parent
                for (i,j),v in zip(locations,values):q[i,j]=v
            assert bridge.queries==cost.sum()
            logits=roots[:,378:2346][np.arange(n)[:,None],ids]
            stat=dict(seconds=time.monotonic()-begin,new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds,unique_child_nn_calls=bridge.queries)
            tmp=path.with_suffix('.partial')
            with tmp.open('wb') as f:np.savez_compressed(f,z=logits,q=q,ids=ids,mask=mask,target=target,nodes=cost,prefix_lengths=[len(r['prefix']) for r in part],
                prefix_hash=[__import__('hashlib').sha256(np.array(r['prefix'],np.int16).tobytes()).hexdigest() for r in part],stats=json.dumps(stat))
            tmp.replace(path);print('Player history',lo+n,'/',len(items),stat['seconds'],flush=True)
        with np.load(path) as z:stats.append(json.loads(str(z['stats'])))
    report=dict(positions=len(rows),unique_past_positions=len(items),past_decision_uses=int((index>=0).sum()),
        seconds=time.monotonic()-start,scoring_seconds=sum(s['seconds'] for s in stats),new_tokens=sum(s['new_tokens'] for s in stats),
        unique_child_nn_calls=sum(s['unique_child_nn_calls'] for s in stats),plan_sha256=digest(pp),index_sha256=digest(out/'history-index.npy'))
    atomic(out/'worker.json',report)
    from .golden_player_policy import analyze
    analyze()
    return report

if __name__=='__main__':test()
