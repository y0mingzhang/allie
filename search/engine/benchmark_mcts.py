"""Measure released MCTS with identical algorithms on old and new engines."""
import json
import time
import numpy as np
from .benchmark import ROOT
from .mcts import run,reference


def run_benchmark(oracle):
    from client import Oracle
    rows=json.loads((ROOT/'dev.json').read_text())['positions'];results=[]
    legacy=Oracle()
    for n in (32,128):
        block=rows[:n]
        for label,engine,algorithm in [('original',legacy,reference.run),('cached_python',oracle,reference.run),('cached_native',oracle,run)]:
            for adaptive in (False,True):
                if engine is oracle:oracle.reset()
                start=time.monotonic();root=engine([r['prefix'] for r in block])
                out,stats=algorithm(block,root,engine,adaptive=adaptive,n_sims=52)
                stats.update(engine=label,roots=n,adaptive=adaptive,end_to_end_seconds=time.monotonic()-start)
                if engine is oracle:stats.update(new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds)
                results.append(stats);print('MCTS benchmark',stats,flush=True)
    return dict(stage='Development-only cost benchmark, same fixed52/adaptive50 algorithms',runs=results)
