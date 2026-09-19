"""End-to-end CPU profile of a large native MCTS batch."""
import cProfile
import gc
import io
import json
import pstats
import time
from .benchmark import ROOT
from .native_mcts import run as mcts


def run(oracle,spec):
    rows=json.loads((ROOT/'dev.json').read_text())['positions'][:1024];report=[]
    for label in ('profile','normal','gc_disabled'):
        oracle.reset();gc.collect();z=oracle([r['prefix'] for r in rows]);p=cProfile.Profile()
        if label=='profile':p.enable()
        if label=='gc_disabled':gc.disable()
        try:out,stats=mcts(rows,z,oracle,n_sims=61)
        finally:
            if label=='gc_disabled':gc.enable()
        if label=='profile':
            p.disable();stream=io.StringIO();pstats.Stats(p,stream=stream).sort_stats('cumulative').print_stats(35)
            (ROOT/'logs/native-mcts-profile.txt').write_text(stream.getvalue())
        cleanup=time.monotonic();gc.collect();stats['cleanup_seconds']=time.monotonic()-cleanup
        stats['mode']=label;report.append(stats);print('profile',stats,flush=True)
    return dict(runs=report)
