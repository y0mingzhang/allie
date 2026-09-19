"""Check the faster implementation using already-frozen development coefficients."""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[2]/'results/search-v1'


def main():
    out=ROOT/'fast-deeper-pilot';rows=json.loads((ROOT/'dev.json').read_text())['positions'];n=len(rows)
    plan=json.loads((out/'engine-plan.json').read_text());bs=plan['roots_per_batch']
    paths=[out/f'{i:06d}.npz' for i in range(0,n,bs)];assert all(p.exists() for p in paths)
    root=[];q=[]
    for lo,path in zip(range(0,n,bs),paths):
        with np.load(path) as z:
            assert list(z['game'])==[r['game'] for r in rows[lo:lo+bs]]
            root.append(z['root']);q.append(z['q'])
    root=np.concatenate(root)[:,378:2346].astype(float);q=np.concatenate(q,axis=1).astype(float)
    old=json.loads((ROOT/'deeper-pilot/results.json').read_text())
    legal=np.isfinite(q[0]);q=np.nan_to_num(q);target=np.array([r['target']-378 for r in rows]);ar=np.arange(n)
    check=np.array([r['fold']==1 for r in rows]);expert=np.array([r['cell']%4==3 for r in rows])
    methods={}
    for name,alpha,beta,depth in [('legal',1.,0.,1)]+[(f'depth{d}',*old['results'][f'depth{d}']['coefficients'],d) for d in range(1,5)]:
        z=np.where(legal,alpha*root+beta*q[depth-1],-np.inf);loss=logsumexp(z,axis=1)-z[ar,target]
        methods[name]=dict(ce=float(loss[check].mean()),expert_ce=float(loss[check&expert].mean()),
            accuracy=float((z.argmax(1)==target)[check].mean()),expert_accuracy=float((z.argmax(1)==target)[check&expert].mean()),
            alpha=alpha,beta=beta)
    report=dict(stage='Fast engine, existing small development check; frozen original coefficients, no refit',
        positions=int(check.sum()),expert_positions=int((check&expert).sum()),methods=methods,
        training_equivalent_cm=None,cm_note='No matching development learning law; golden CM pending')
    p=out/'results.json';tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(p)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
