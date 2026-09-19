"""Exact deterministic-oracle proof for staged heterogeneous deep search.

Uses CPU audit builds under runtime/cpu-audit; no Slurm or neural queries.
"""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    start=time.monotonic();folder=ROOT/'results/search-v1/runtime/cpu-audit';sys.path.insert(0,str(folder))
    import _allie_growforest as forest
    import _allie_scaled_count as backup
    sys.path.insert(0,'/data/group_data/dei-group/yimingz3/allie/results/recipe10x/data-v1-round2/source-ours')
    from chess_vocab import MOVES
    forest.initialize(MOVES)
    rows=json.loads((ROOT/'results/search-v1/aug-deep-scale-v1/sample.json').read_text())['positions'][:8]
    prefixes=[r['prefix'] for r in rows];budgets=[128,256,512,1000,4000,16000,0,4000];n=len(rows)
    def oracle(ps):
        return np.array([np.random.default_rng(int(hashlib.sha256(np.array(p,np.int16).tobytes()).hexdigest()[:8],16)).normal(size=2432).astype(np.float32) for p in ps])
    def advance(tree,known):
        while not tree.done:
            h=tree.select();queries=[]
            for node,parent,move,length in h:
                p=known[int(parent)]+[int(move)];assert len(p)==length;known[int(node)]=p;queries.append(p)
            if queries:tree.update(oracle(queries))
    z=oracle(prefixes)
    full=forest.Tree(prefixes,z,[16000]*n,[2.5]*n,1);advance(full,{i:p for i,p in enumerate(prefixes)});reference=full.compact();del full
    mixed=forest.Tree(prefixes,z,[min(b,128) for b in budgets],[2.5]*n,1);known={i:p for i,p in enumerate(prefixes)}
    advance(mixed,known);before=np.array(mixed.evals);mixed.grow(budgets);advance(mixed,known);actual=mixed.compact()
    k=max(len(r['legal']) for r in rows);ids=np.zeros((n,k),np.int32)
    for i,r in enumerate(rows):ids[i,:len(r['legal'])]=np.array(r['legal'])-378
    parent=reference['parent'];owner=np.full(len(parent),-1,np.int32)
    owner[reference['roots']]=np.arange(n)
    for i,p in enumerate(parent):
        if p>=0:owner[i]=owner[p]
    expected=[]
    for i,b in enumerate(budgets):
        scale=16*max(1,b/1000)
        qref=backup.Backup(reference,b,scale).reduce(np.log(.2),-.5,ids)[0,i]
        qactual=backup.Backup(actual,16000,scale).reduce(np.log(.2),-.5,ids)[0,i]
        np.testing.assert_array_equal(qref,qactual)
        expected.append(int(((parent>=0)&(owner==i)&(reference['born']<=b)&(reference['terminal']<0)).sum()))
    np.testing.assert_array_equal(mixed.evals,expected)
    assert (np.array(mixed.evals)>=before).all()
    # A no-op continuation may not repeat any model evaluation.
    mixed.grow(budgets);assert mixed.done;np.testing.assert_array_equal(mixed.evals,expected)
    result=dict(budgets=budgets,reference_budget=16000,nn_evals=expected,seconds=time.monotonic()-start,
        checks=['staged mixed128..16000 root Q bit-identical to matching full-tree prefixes under each budget-normalized temperature','actual NN counts equal retained nonterminal prefix nodes','initial128 work included exactly once','zero-budget root and no-op grow'],
        caveat='Exact deterministic oracle. GPU batching differences still require the separate live-model audit.',
        sources={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['search/test_mixed_depth.py','search/engine/growforest.cpp','search/engine/scaled_count.cpp','search/engine/diff_backup.cpp']})
    (folder/'mixed-depth-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
