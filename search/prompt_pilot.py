"""Development-only strength-prompt calibration; no model training or golden data.

Change only rating digits of a causal prefix. Candidate selection uses fold 0;
fold 1 is a development check, not a fresh final holdout across research rounds.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from client import Oracle

ROOT = Path(__file__).resolve().parents[1] / 'results/search-v1'
OUT = ROOT / 'prompt-pilot'
VARIANTS = [('both', x) for x in (-600, -300, -100, 0, 100, 300, 600)] + [
    ('mover', x) for x in (-400, -200, 200, 400)]


def transform(prefix, who, delta):
    result = list(prefix)
    mover_start = 3 if (len(prefix) - 11) % 2 == 0 else 7
    for start in (3, 7):
        if who == 'mover' and start != mover_start:
            continue
        original = sum(int(prefix[start + i]) * 10 ** (3-i) for i in range(4))
        # Preserve identity even for a rare out-of-range original rating.
        value = original if delta == 0 else int(np.clip(original + delta, 400, 3500))
        result[start:start+4] = [int(x) for x in f'{value:04d}']
    assert result[:3] == prefix[:3] and result[11:] == prefix[11:]
    return result


def main():
    OUT.mkdir(exist_ok=True)
    data_path = ROOT / 'dev.json'
    rows = json.loads(data_path.read_text())['positions']
    oracle = Oracle()
    plan = dict(variants=VARIANTS, data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                checkpoint=oracle.ready['checkpoint_sha256'],
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Existing pilot dev only; fit fold0, report fold1; no golden tuning')
    plan = json.loads(json.dumps(plan))
    path = OUT / 'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        path.write_text(json.dumps(plan, indent=2)+'\n')
    root = np.concatenate([np.load(ROOT/f'cache-{i:05d}.npz')['root'][:,378:2346]
                           for i in range(0,len(rows),128)]).astype(np.float64)
    logits = []
    for who, delta in VARIANTS:
        path = OUT / f'{who}-{delta:+d}.npz'
        if not path.exists():
            started = time.monotonic()
            batches = []
            for lo in range(0,len(rows),128):
                prefixes = [transform(r['prefix'],who,delta) for r in rows[lo:lo+128]]
                if delta == 0:
                    assert all(p == r['prefix'] for p,r in zip(prefixes,rows[lo:lo+128]))
                    pred = root[lo:lo+128].astype(np.float16)
                else:
                    pred = oracle(prefixes, columns=list(range(378,2346)), compact=True)
                batches.append(pred)
            tmp = path.with_suffix('.partial')
            with tmp.open('wb') as f:
                np.savez(f, logits=np.concatenate(batches), seconds=time.monotonic()-started)
            tmp.replace(path)
            print(who,delta,time.monotonic()-started,flush=True)
        with np.load(path) as z:
            logits.append(z['logits'].astype(np.float64))
    base_index = VARIANTS.index(('both',0))
    assert np.array_equal(logits[base_index],root)
    target = np.array([r['target']-378 for r in rows]); ar=np.arange(len(rows))
    legal = np.zeros_like(root,dtype=bool)
    for i,r in enumerate(rows):legal[i,np.array(r['legal'])-378]=True
    fit = np.array([r['fold']==0 for r in rows]); expert=np.array([r['cell']%4==3 for r in rows])
    def lp(x,temp=1):
        x=np.where(legal,x/temp,-np.inf)
        return x-logsumexp(x,axis=1,keepdims=True)
    base_lp=lp(root);candidates=[]
    def record(name,probs,parameters):
        nll=-probs[ar,target];correct=probs.argmax(1)==target
        scores={}
        for label,mask in [('fit',fit),('confirmation',~fit)]:
            scores[label]=dict(ce=float(nll[mask].mean()),expert_ce=float(nll[mask&expert].mean()),
                              accuracy=float(correct[mask].mean()),expert_accuracy=float(correct[mask&expert].mean()))
        candidates.append(dict(name=name,parameters=parameters,**scores))
    for temp in (.85,1.,1.15):record('temperature',lp(root,temp),dict(kind='temperature',temperature=temp))
    for variant,x in zip(VARIANTS,logits):
        if variant==('both',0):continue
        shifted_lp=lp(x)
        for weight in (.5,1.):
            probs=shifted_lp if weight==1 else np.logaddexp(base_lp,shifted_lp)-np.log(2.)
            record('strength_mixture',probs,dict(kind='mixture',variant=variant,weight=weight))
    high=logits[VARIANTS.index(('both',300))];low=logits[VARIANTS.index(('both',-300))]
    for alpha in (-.5,.5,1.):
        record('strength_contrast',lp(root+alpha*(high-low)),dict(kind='contrast',alpha=alpha,offset=300))
    base_ce=float((-base_lp[ar,target])[fit].mean())
    viable=[c for c in candidates if c['fit']['ce']<=base_ce+.002]
    selected=min(viable,key=lambda c:c['fit']['expert_ce'])
    report=dict(stage='Development pilot, not golden; cannot convert with golden CM laws',
                positions=len(rows),selection_rule='minimum fit expert CE with fit overall CE <= legal + .002',
                selected=selected,candidates=candidates)
    (OUT/'selected.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(selected=selected,legal_ce=base_ce),indent=2),flush=True)


if __name__=='__main__':main()
