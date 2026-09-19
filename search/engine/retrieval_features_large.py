"""Build an immutable next-move datastore from causally preceding hidden states."""
import json
from pathlib import Path
import time
import numpy as np
from .service import ROOT, GLOBAL_STOP, atomic
from .balanced_eval import digest
from .features import full

def batches(tokens, offsets, first, end, max_tokens=4096):
    group=[];size=0
    for i in range(first,end):
        seq=tokens[offsets[i]:offsets[i+1]].tolist()
        assert 12<=len(seq)<=1025
        if group and size+len(seq)>max_tokens:
            yield group
            group=[];size=0
        group.append((i,seq));size+=len(seq)
    if group:yield group

def run(oracle,spec):
    smoke=json.loads((ROOT/'feature-smoke-v1/results.json').read_text())
    assert all(smoke[k] for k in ('logit_identity','last_full_identity','future_mask_identity'))
    bank=ROOT/'retrieval-bank-large-v1';out=ROOT/'retrieval-features-large-v1';out.mkdir(exist_ok=True)
    source=bank/'bank.npz';manifest=json.loads((bank/'manifest.json').read_text())
    sources={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('features.py'),Path(__file__).with_name('direct.py')]}
    assert smoke['sources']['features.py']==sources['features.py']
    plan=dict(bank_sha256=digest(source),manifest_sha256=digest(bank/'manifest.json'),sources=sources,
        export_sha256=digest(ROOT/'serving-export/model.safetensors'),
        export_provenance=json.loads((ROOT/'serving-export/provenance.json').read_text()),
        games_per_shard=256, max_prefill_tokens=4096, hidden_dtype='float16', hidden_dim=512,
        alignment='Target at game token t uses hidden state at t-1. Full causal prefill; future-mask test passed. No target input in that key.',
        extra_data='Additional June 2026 training-corpus memory, not a pure model-only search method.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    with np.load(source) as z:tokens=z['tokens'];labels=z['labels'];offsets=z['offsets'];games=z['game']
    assert len(offsets)==len(games)+1 and offsets[-1]==len(tokens)==len(labels)
    start=time.monotonic();stats=[]
    for lo in range(0,len(games),plan['games_per_shard']):
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        hi=min(len(games),lo+plan['games_per_shard']);path=out/f'{lo:06d}.npz'
        if path.exists():
            with np.load(path) as z:stats.append(json.loads(str(z['stats'])))
            continue
        begin=time.monotonic();features=[];targets=[];cells=[];game_ix=[];positions=[];input_tokens=0;forward_seconds=0.
        for group in batches(tokens,offsets,lo,hi):
            oracle.reset();_,h=full(oracle,[seq for i,seq in group]);at=0
            for i,seq in group:
                lab=labels[offsets[i]:offsets[i+1]]
                ix=np.flatnonzero(lab>=0)
                assert ((ix>=11)&(ix<len(seq))).all()
                target=np.asarray(seq)[ix]-378
                assert ((target>=0)&(target<1968)).all()
                features.append(h[at+ix-1]);targets.append(target.astype(np.int16))
                cells.append(lab[ix]);game_ix.append(np.full(len(ix),i,np.int32));positions.append(ix.astype(np.int16))
                at+=len(seq)
            assert at==len(h)
            input_tokens+=oracle.new_tokens;forward_seconds+=oracle.forward_seconds
        stat=dict(lo=lo,hi=hi,positions=sum(map(len,targets)),input_tokens=input_tokens,forward_seconds=forward_seconds,seconds=time.monotonic()-begin)
        temp=path.with_suffix('.partial')
        with temp.open('wb') as f:np.savez(f,hidden=np.concatenate(features),target=np.concatenate(targets),cell=np.concatenate(cells),game_ix=np.concatenate(game_ix),position=np.concatenate(positions),stats=json.dumps(stat))
        temp.replace(path);stats.append(stat)
        print('Feature bank',hi,'/',len(games),'positions',stat['positions'],'seconds',round(stat['seconds'],2),flush=True)
    result=dict(games=len(games),positions=sum(s['positions'] for s in stats),
        input_tokens=sum(s['input_tokens'] for s in stats),forward_seconds=sum(s['forward_seconds'] for s in stats),
        build_seconds=sum(s['seconds'] for s in stats),invocation_seconds=time.monotonic()-start,
        bytes=sum(p.stat().st_size for p in out.glob('*.npz')),plan_sha256=digest(pp),
        shards={p.name:digest(p) for p in sorted(out.glob('*.npz'))})
    assert result['positions']==sum(manifest['positions'])
    atomic(out/'results.json',result)
    return result
