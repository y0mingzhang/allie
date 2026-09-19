"""Exact-board frequencies from the same temporally clean June pilot bank."""
from collections import defaultdict,Counter
import hashlib,json,pickle,time
from pathlib import Path
import chess
import numpy as np
from scipy.optimize import minimize_scalar
from .engine.service import ROOT,atomic
from .engine.balanced_eval import digest
from .engine.native_board import MOVES
from .engine.retrieval_common import data
from .engine.analyze_august import means

def key(board):
    # Includes pieces, side, castling and only legally meaningful en-passant.
    # Halfmove clock matters for the 50-move rule. Full repetition history is
    # not a key field; this is an empirical policy prior, not a value oracle.
    return board._transposition_key()+(board.halfmove_clock,)

def main():
    start=time.monotonic();out=ROOT/'aug-boardbook-v1';assert not out.exists();out.mkdir()
    bank=ROOT/'retrieval-bank-v2';book=defaultdict(Counter);move=[chess.Move.from_uci(m) for m in MOVES]
    with np.load(bank/'bank.npz') as z:tokens=z['tokens'];labels=z['labels'];offsets=z['offsets'];bank_games=z['game']
    build=time.monotonic()
    for i in range(len(offsets)-1):
        ts=tokens[offsets[i]:offsets[i+1]];ls=labels[offsets[i]:offsets[i+1]];b=chess.Board()
        for j,t in enumerate(ts[11:],11):
            if not 378<=t<2346:
                assert j==len(ts)-1 and t in (2346,2347);break
            if ls[j]>=0:book[(int(ls[j]),key(b))][int(t)-378]+=1
            b.push(move[int(t)-378])
        if i%2048==0:print('Board book',i,'games',flush=True)
    build_seconds=time.monotonic()-build
    d=data();rows=d['rows'];cells=d['cells'];games=d['games'];fm=d['fit'];cv=d['cv'];target=d['target'];n=len(rows);ar=np.arange(n)
    assert not set(bank_games).intersection(games)
    p=np.zeros_like(d['ids'],float);counts=np.zeros(n,int);begin=time.monotonic()
    for i,r in enumerate(rows):
        b=chess.Board()
        for t in r['prefix'][11:]:b.push(move[t-378])
        lookup=book.get((r['cell'],key(b)),{})
        legal=set(np.asarray(r['legal'])-378)
        assert set(lookup).issubset(legal)
        for j,t in enumerate(d['ids'][i]):
            if d['mask'][i,j]:p[i,j]=lookup.get(int(t),0)
        counts[i]=p[i].sum()
        if counts[i]:p[i]/=counts[i]
    query_seconds=time.monotonic()-begin
    with (out/'book.pkl').open('wb') as f:pickle.dump(dict(book),f,protocol=5)
    with (out/'counts.npz').open('wb') as f:np.savez_compressed(f,p=p,counts=counts,ids=d['ids'],mask=d['mask'],game=games,ply=[r['ply'] for r in rows])
    plan=dict(bank_sha256=digest(bank/'bank.npz'),sample_sha256=digest(ROOT/'aug-tune-v1/sample.json'),source_sha256=digest(Path(__file__)),
        key='same16cell + piece placement,side,castling,legal_ep,halfmove_clock; full repetition history not included',
        variants=['constant','dirichlet'],fit='Global lambda or log pseudocount, separately for direct/search baseline, fit-game CV. Missing boards retain baseline exactly.',
        semantics='Additional June training-corpus memory; exact-state retrieval, no Stockfish and no root human-target input into lookup.')
    atomic(out/'plan.json',plan)
    def policy(base,a,kind):
        lam=np.full(n,a) if kind=='constant' else counts/(counts+np.exp(a))
        lam=np.where(counts>0,lam,0.)
        return (1-lam[:,None])*base+lam[:,None]*p
    def fit(base,train,kind):
        bounds=(0.,.95) if kind=='constant' else (np.log(.1),np.log(1e6))
        count=np.bincount(cells[train],minlength=16);w=1/count[cells[train]];w/=w.sum()
        obj=lambda a:float(w@(-np.log(policy(base,a,kind)[ar[train],target[train]])))
        r=minimize_scalar(obj,bounds=bounds,method='bounded',options=dict(xatol=1e-8));assert r.success
        zero=0. if kind=='constant' else np.log(1e6)
        return float(min([zero,r.x,bounds[0],bounds[1]],key=obj))
    records={};losses={}
    def record(name,prob,oof,params,base,kind):
        loss=-np.log(prob[ar,target]);conf=means(loss[~fm],cells[~fm]);cvce=means(oof[fm],cells[fm])
        records[name]=dict(parameters=params,base=base,kind=kind,training_equivalent_cm=None,
            mean_nodes=float(d['cost'].mean()) if base=='search' else 0.,
            fit_game_cv=dict(macro_ce=float(cvce.mean()),expert_ce=float(cvce[3::4].mean())),
            confirmation=dict(macro_ce=float(conf.mean()),expert_ce=float(conf[3::4].mean()),cells=conf.tolist()))
        losses[name]=loss
    for base,versions in d['controls'].items():
        bp,params=versions[0];oof=np.full(n,np.nan)
        for f in range(3):
            val=fm&(cv==f);pp=versions[f+1][0];oof[val]=-np.log(pp[ar[val],target[val]])
        record(base,bp,oof,dict(base=params),base,None)
        for kind in plan['variants']:
            a=fit(bp,fm,kind);prob=policy(bp,a,kind);oof=np.full(n,np.nan)
            for f in range(3):
                val=fm&(cv==f);pp=versions[f+1][0];af=fit(pp,fm&(cv!=f),kind);cp=policy(pp,af,kind)
                oof[val]=-np.log(cp[ar[val],target[val]])
            record(base+'_'+kind,prob,oof,dict(base=params,mixing=a),base,kind)
    selected={metric:min(records,key=lambda x:records[x]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;ct=np.zeros((ng,16));np.add.at(ct,(ix,cells[~fm]),1)
    w=np.random.default_rng(98318).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=w@ct;assert (den>0).all()
    for name,rec in records.items():
        delta=losses[name]-losses[rec['base']];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);draw=w@sums/den
        rec['confirmation_delta_vs_base']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),macro_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draw[:,3::4].mean(1),[.025,.975]).tolist())
    report=dict(stage='August game-CV / disjoint confirmation; potentially model-training-seen, no golden CM.',results=records,fit_cv_selected=selected,
        cells_coverage=means(counts>0,cells).tolist(),mean_examples_per_query=float(counts.mean()),
        max_examples=int(counts.max()),bank_keys=len(book),bank_games=len(bank_games),bank_positions=int(labels[labels>=0].size),
        build_seconds=build_seconds,query_seconds=query_seconds,invocation_seconds=time.monotonic()-start,
        book_bytes=(out/'book.pkl').stat().st_size,plan_sha256=digest(out/'plan.json'))
    atomic(out/'results.json',report)
    with (out/'scores.npz').open('wb') as f:np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    for name,rec in records.items():print(name,rec['confirmation']['macro_ce'],rec['confirmation']['expert_ce'],rec['parameters'].get('mixing'),flush=True)
    print('SELECTED',selected,'coverage',report['cells_coverage'],'seconds',report['invocation_seconds'],flush=True)

if __name__=='__main__':main()
