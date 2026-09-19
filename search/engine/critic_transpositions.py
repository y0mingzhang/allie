"""Same-board legal history averaging of WDL critics, actual human prior unchanged."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .service import ROOT, GLOBAL_STOP, atomic
from .balanced_eval import digest
from .native_board import from_prefix
from .transpose_history import variants, test as variant_test
from .sample_data import read
from .residual_screen import analyze


def run(oracle, spec):
    variant_test()
    start = time.monotonic(); study = 'aug-critic-transpositions-v1'
    out = ROOT / study; out.mkdir(exist_ok=True)
    d = read('aug-tune-expanded-v1'); rows, games, mask = (d[k] for k in ('rows','games','mask'))
    n, k = mask.shape
    p = dict(sample_sha256=digest(ROOT / 'aug-tune-expanded-v1/sample.json'), max_variants=2, lookback=24,
             sources={f: digest(Path(__file__).with_name(f)) for f in ('critic_transpositions.py','transpose_history.py','residual_screen.py','direct.py')},
             semantics='Only reorder known moves before a later unchanged irreversible pawn move/capture. Exact intermediate and final FEN, suffix legality/termination checked. No clock inputs in this fixed checkpoint. No root target or future move in variant generation. Present and future repetition-relevant suffix unchanged.',
             method='Evaluate actual and up to2 equivalent histories after every legal root action. Equal mean or median value minus actual-history value becomes an additive log-policy correction. Frozen1000-node policy stays unchanged at coefficient0. Original oneply value is a control.',
             validation='Independent python-chess repetition-claim checks on up to32 changed roots. All variant child outcomes agree with original. Inference cost includes every new prefix and nonterminal child query; no training/external value engine.')
    pp = out / 'plan.json'
    if pp.exists(): assert json.loads(pp.read_text()) == p
    else: atomic(pp, p)
    for lo in range(0,n,256):
        path = out / f'{lo:06d}.npz'
        if path.exists(): continue
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists(): raise RuntimeError('STOP')
        before=time.monotonic(); part=rows[lo:lo+256]; m=len(part); width=max(len(r['legal']) for r in part)
        q=np.zeros((3,m,width)); counts=np.ones(m,int); nodes=np.zeros(m,int); prefills=np.zeros(m,int)
        prefixes=[]; children=[]; owners=[]; checked=0
        for i,r in enumerate(part):
            original=r['prefix']; b=from_prefix(original)
            vv=variants(original,maximum=2); contexts=[original]+[x['prefix'] for x in vv]
            counts[i]=len(contexts); prefills[i]=len(contexts)
            if lo==0 and vv and checked<32:
                import chess
                from .native_board import MOVE_ID
                inverse={v:u for u,v in MOVE_ID.items()}
                states=[]
                for context in contexts:
                    cb=chess.Board()
                    for token in context[11:]: cb.push_uci(inverse[token])
                    states.append((cb.fen(en_passant='fen'),cb.can_claim_threefold_repetition(),cb.is_fivefold_repetition(),cb.can_claim_fifty_moves()))
                assert all(s==states[0] for s in states),states
                checked+=1
            for v,context in enumerate(contexts):
                board=from_prefix(context)
                assert board.fen()==b.fen() and board.outcome()==b.outcome()
                assert sorted(board.legal())==sorted(r['legal'])
                prefixes.append(context)
                for j,move in enumerate(r['legal']):
                    child=board.child(move); terminal=child.outcome()
                    assert terminal==b.child(move).outcome()
                    if terminal>=0:
                        q[v,i,j]=0. if terminal==.5 else 1.
                    else:
                        children.append(context+[move]);owners.append((v,i,j));nodes[i]+=1
        oracle.reset();oracle(prefixes)
        prefill_tokens=oracle.new_tokens
        z=oracle(children) if children else np.empty((0,2432))
        wdl=softmax(z[:,2413:2416].astype(float),axis=1)
        for a,(v,i,j) in enumerate(owners): q[v,i,j]=wdl[a,2]-wdl[a,0]
        assert np.isfinite(q).all() and np.max(np.abs(q))<=1.
        stats=dict(seconds=time.monotonic()-before,forward_seconds=oracle.forward_seconds,prefill_tokens=prefill_tokens,
                   child_model_tokens=oracle.new_tokens-prefill_tokens,
                   total_model_tokens=oracle.new_tokens,root_prefix_queries=len(prefixes),child_queries=len(children),repetition_claim_roots_checked=checked)
        tmp=path.with_suffix('.partial')
        with tmp.open('wb') as f: np.savez_compressed(f,q=q,counts=counts,nodes=nodes,prefills=prefills,game=[r['game'] for r in part],ply=[r['ply'] for r in part],stats=json.dumps(stats))
        tmp.replace(path);print('critic transpose',lo+m,n,stats['seconds'],flush=True)
    q=np.zeros((3,n,k));counts=np.zeros(n,int);nodes=np.zeros(n,int);prefills=np.zeros(n,int);stats=[]
    for path in sorted(out.glob('[0-9]*.npz')):
        with np.load(path) as f:
            lo=int(path.stem);hi=lo+len(f['game']);np.testing.assert_array_equal(f['game'],games[lo:hi])
            q[:,lo:hi,:f['q'].shape[-1]]=f['q'];counts[lo:hi]=f['counts'];nodes[lo:hi]=f['nodes'];prefills[lo:hi]=f['prefills'];stats.append(json.loads(str(f['stats'])))
    assert (counts>=1).all()
    mean=q.sum(0)/counts[:,None]
    active=np.arange(3)[:,None,None]<counts[None,:,None]
    median=np.nanmedian(np.where(active,q,np.nan),axis=0)
    features=dict(original_oneply=q[0],mean_delta=mean-q[0],median_delta=median-q[0])
    for name in ('mean_delta','median_delta'): assert np.array_equal(features[name][counts==1],np.zeros_like(features[name][counts==1]))
    coverage=[float(np.mean(counts[d['cells']==c]>1)) for c in range(16)]
    worker=dict(positions=n,seconds=time.monotonic()-start,blocks=stats,coverage_by_cell=coverage,plan_sha256=digest(pp))
    atomic(out/'worker.json',worker)
    costs={name:dict(extra_nodes=nodes+prefills,summary=dict(mean_extra_children=float(nodes.mean()),mean_extra_prefixes=float(prefills.mean()),
                 summed_wall_seconds=sum(s['seconds'] for s in stats),prefill_tokens=sum(s['prefill_tokens'] for s in stats))) for name in features}
    # Original-only control can omit all variant work when served alone.
    costs['original_oneply']=dict(extra_nodes=nodes/counts+1,summary=dict(mean_extra_children=float(np.mean(nodes/counts)),mean_extra_prefixes=1))
    report=analyze(study,features,costs,p['method'])
    return dict(worker=worker,fit_cv_selected=report['fit_cv_selected'])
