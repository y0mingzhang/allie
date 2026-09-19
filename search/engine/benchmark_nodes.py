"""Equal-node-work throughput, including large batches of independent MCTS trees."""
import json
import time
import numpy as np
from .benchmark import ROOT
from .native_mcts import run as mcts
from .tree import batch as four_ply


def run(oracle,spec):
    rows=json.loads((ROOT/'dev.json').read_text())['positions'];records=[]
    for repeat in range(2):
        oracle.reset();q,z,stats=four_ply(rows[:64],oracle,batch_size=1024)
        stats.update(method='four_ply',roots=64,repeat=repeat,nodes=sum(stats['leaves_by_depth']),
            new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds)
        stats['nodes_per_second']=stats['nodes']/stats['seconds'];records.append(stats)
        print('equal-node benchmark',stats,flush=True)
    budget=sum(stats['leaves_by_depth'])
    for n in (128,512,1024):
        for repeat in range(2):
            oracle.reset();start=time.monotonic();z=oracle([r['prefix'] for r in rows[:n]])
            prefill_seconds=time.monotonic()-start;prefill_tokens=oracle.new_tokens;prefill_gpu=oracle.forward_seconds
            sims=int(round(budget/n));out,stats=mcts(rows[:n],z,oracle,n_sims=sims)
            total_seconds=time.monotonic()-start
            stats.update(method='native_mcts',roots=n,repeat=repeat,simulations_per_root=sims,
                nodes=stats['evaluated_leaves'],prefill_seconds=prefill_seconds,
                end_to_end_seconds=total_seconds,new_tokens=oracle.new_tokens,
                prefill_tokens=prefill_tokens,forward_seconds=oracle.forward_seconds,
                search_forward_seconds=oracle.forward_seconds-prefill_gpu)
            stats['nodes_per_second']=stats['nodes']/stats['seconds'];records.append(stats)
            print('equal-node benchmark',stats,flush=True)
    return dict(stage='Infrastructure throughput; different numbers of roots, roughly equal evaluated leaves, no quality comparison',
        runs=records)
