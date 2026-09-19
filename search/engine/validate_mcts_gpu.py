"""Real-logit native-tree replay against the reference, without extra GPU work."""
import json
import numpy as np
from .benchmark import ROOT
from .native_mcts import run as native
from .mcts import reference


def run(oracle,spec):
    rows=json.loads((ROOT/'dev.json').read_text())['positions'][:128];report={}
    for adaptive in (False,True):
        oracle.reset();root=oracle([r['prefix'] for r in rows]);tape=[]
        def record(prefixes):
            z=oracle(prefixes);tape.append(([list(p) for p in prefixes],z.copy()));return z
        expected,stats=reference.run(rows,root,record,adaptive=adaptive,n_sims=52)
        cursor=0
        def replay(prefixes):
            nonlocal cursor
            paths,z=tape[cursor];cursor+=1
            assert prefixes==paths,f'Selection differs at inference batch {cursor}'
            return z
        actual,native_stats=native(rows,root,replay,adaptive=adaptive,n_sims=52)
        assert cursor==len(tape)
        for key in expected:np.testing.assert_allclose(actual[key],expected[key],rtol=0,atol=1e-7)
        for key in stats:
            if key!='seconds':assert stats[key]==native_stats[key],key
        report['adaptive' if adaptive else 'fixed']=dict(requests=cursor,evaluated_leaves=stats['evaluated_leaves'],
            max_policy_delta=float(np.max(np.abs(actual['policy']-expected['policy']))),
            native_cpu_seconds=native_stats['seconds'])
    return dict(stage='Real-model replay: identical selected paths and visit counts; output tolerance1e-7',methods=report)
