"""No-math-change fast-path gate: exact outputs, slot tables, then wall-time repeats."""
import cProfile
import io
import json
import pstats
import time
import numpy as np
import torch
from .direct import DirectOracle
from .direct_packed import PackedOracle
from .benchmark import ROOT
from .tree import batch
from .native_mcts import run as mcts


def run(oracle,spec):
    rows=json.loads((ROOT/'dev.json').read_text())['positions'];reports=[]
    original=oracle.__class__
    def stages():
        seq=[r['prefix'] for r in rows[:128]]
        return [seq,[s+[r['legal'][0]] for s,r in zip(seq,rows[:128])],seq,
                [s[:-2] for s in seq],[[s[0]] for s in seq]]
    try:
        predictions=[];tables=[]
        for cls in (DirectOracle,PackedOracle):
            oracle.reset();oracle.__class__=cls;zs=[];maps=[]
            for seq in stages():
                zs.append(oracle(seq))
                # Only positions in each live sequence are meaningful in the table.
                maps.extend(oracle.runner.req_to_token_pool.req_to_token[oracle.nodes[tuple(s)],:len(s)].cpu().numpy().copy() for s in seq)
            predictions.append(zs);tables.append(maps)
        for a,b in zip(predictions[0],predictions[1]):np.testing.assert_array_equal(a,b)
        for a,b in zip(tables[0],tables[1]):np.testing.assert_array_equal(a,b)
        for repeat in range(3):
            order=(DirectOracle,PackedOracle) if repeat%2==0 else (PackedOracle,DirectOracle)
            for cls in order:
                for kind,n in [('four_ply',64),('mcts',1024)]:
                    oracle.reset();oracle.__class__=cls;part=rows[:n];start=time.monotonic()
                    if kind=='four_ply':_,_,stats=batch(part,oracle,batch_size=1024)
                    else:
                        z=oracle([r['prefix'] for r in part]);_,stats=mcts(part,z,oracle,n_sims=61)
                    stats.update(mode=cls.__name__,kind=kind,repeat=repeat,
                                 end_to_end_seconds=time.monotonic()-start,forward_seconds=oracle.forward_seconds)
                    reports.append(stats);print('packed benchmark',stats,flush=True)
        oracle.__class__=DirectOracle;oracle.reset();part=rows[:1024]
        z=oracle([r['prefix'] for r in part]);profile=cProfile.Profile();profile.enable();mcts(part,z,oracle,n_sims=61);profile.disable()
        stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats('cumulative').print_stats(40)
        (ROOT/'logs/native-mcts-profile-current.txt').write_text(stream.getvalue())
        return dict(stage='Exact logits and valid KV slot maps match across prefill/forks/repeated roots/reset; no quality tuning',runs=reports)
    finally:
        oracle.__class__=original;oracle.reset()
