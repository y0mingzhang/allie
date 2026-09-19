"""Frozen representative baseline transports with unchanged search semantics."""
import importlib,json,subprocess,sys,sysconfig,time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from search.engine.service import ROOT,atomic,GLOBAL_STOP
from search.engine.balanced_eval import digest
from search.engine.native_board import MOVES,from_prefix
from search.engine.policy import solve
from search.engine.adaptive_policy import output
from .collect import OUT,inventory,freeze
from .handles import ShipHandles
from .context import predicted_seconds


def native():
    import pybind11
    folder=OUT/'native';folder.mkdir(exist_ok=True)
    source=Path(__file__).with_name('allie_handles.cpp');target=folder/('_ship_allie_handles'+sysconfig.get_config_var('EXT_SUFFIX'))
    key=dict(source=digest(source),engine={n:digest(source.parent.parent/'engine'/n) for n in ('board.cpp','mcts_native.hpp')},python=sys.version)
    stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        tmp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(source),'-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),'-I'+str(source.parents[2]/'vendor/chess-library/include'),'-o',str(tmp)],check=True)
        tmp.replace(target);atomic(stamp,key)
    sys.path.insert(0,str(folder));m=importlib.import_module('_ship_allie_handles');m.initialize(MOVES);return m


def shallow(rows,bridge):
    """All legal root moves, then top4/2/2 policy branches; unchanged critic fallback."""
    n=len(rows);white=[from_prefix(r['prefix']).white for r in rows]
    q=np.full((4,n,1968),np.nan);active=[];next_id=n;counts=np.zeros((4,n),int)
    for i,r in enumerate(rows):
        board=from_prefix(r['prefix'])
        for token in r['legal']:
            b=board.child(token);v=b.outcome()
            if v>=0:q[0,i,token-378]=v if white[i] else 1-v
            else:active.append((i,token-378,next_id,i,token,len(r['prefix'])+1,b,1.,0.));next_id+=1
    terminal_delta=np.zeros((n,1968))
    for depth in range(4):
        if depth:q[depth]=q[depth-1]+terminal_delta
        next_nodes=[];terminal_delta=np.zeros((n,1968))
        for lo in range(0,len(active),2048):
            part=active[lo:lo+2048]
            h=np.array([[a[2],a[3],a[4],a[5]] for a in part],np.int32);pred=bridge(h)
            value=softmax(pred[:,2413:2416].astype(float),axis=1)@np.array([1.,.5,0.])
            if depth%2==0:value=1-value
            for a,z,v in zip(part,pred,value):
                owner,action,node,parent,token,length,board,mass,pv=a;counts[depth,owner]+=1
                if depth==0:q[depth,owner,action]=v
                else:q[depth,owner,action]+=mass*(v-pv)
                if depth==3 or length>=1025:continue
                legal=board.legal();p=softmax(z[legal].astype(float))
                for j in np.argsort(p)[::-1][: (4,2,2)[depth]]:
                    move=legal[j];b=board.child(move);t=b.outcome();w=mass*p[j]
                    if t>=0:terminal_delta[owner,action]+=w*((t if white[owner] else 1-t)-v)
                    else:next_nodes.append((owner,action,next_id,node,move,length+1,b,w,v));next_id+=1
        active=next_nodes
    for i,r in enumerate(rows):assert np.isfinite(q[:,i,np.array(r['legal'])-378]).all()
    assert counts.sum()==bridge.queries
    return q,counts.cumsum(0)


def mcts(rows,bridge,mode):
    n=len(rows);z=bridge.root_logits
    ns=np.clip(np.rint(predicted_seconds(z)*50/4.64001),0,200).astype(int) if mode=='released' else np.full(n,50,int)
    cp=1.25*np.sqrt(50/np.maximum(ns,1)) if mode=='released' else np.full(n,1.25)
    tree=native().Tree([r['prefix'] for r in rows],z,ns.tolist(),cp.tolist())
    tree.first_prior=tree.preserve_depth=(mode=='repaired')
    cache={};owner={i:i for i in range(n)};count=np.zeros(n,int)
    while not tree.done:
        h=tree.select()
        if not len(h):continue
        for node,parent,_,_ in h:owner[int(node)]=owner[int(parent)]
        take=np.array([int(a[0]) not in cache for a in h])
        if take.any():
            zz=bridge(h[take])
            for a,v in zip(h[take],zz):cache[int(a[0])]=v;count[owner[int(a[0])]]+=1
        tree.update(np.array([cache[int(a[0])] for a in h]))
    summaries=tree.summaries();q=np.zeros((n,1968));mask=np.zeros_like(q,bool)
    for i,(ids,visits,v,prior) in enumerate(summaries):q[i,ids]=v;mask[i,ids]=True
    prob=solve(summaries,ns,cp) if mode=='released' else output(z[:,378:2346],q,mask,.9,2.,'reverse')
    assert int(count.sum())==bridge.queries
    return q,prob,count,tree.stats()


def run(oracle,spec):
    plan=freeze();size=spec['model'];split=spec['split'];assert (OUT/('parity-'+size)/'results.json').exists()
    rows,features,hashes=inventory(split);folder=OUT/f'{size}-{split}-baselines';folder.mkdir(exist_ok=True)
    config=dict(plan=digest(OUT/'plan.json'),inventory=hashes,spec=spec,source=digest(Path(__file__)),native=digest(Path(__file__).with_name('allie_handles.cpp')),
        arms=['shallow','released','repaired'],roots_per_block=128,shallow_widths=[4,2,2],released_mean=50,repaired_budget=50,
        semantics='Released time allocation and reverse-KL output unchanged. Repaired = first-prior/depth fixes and calibrated reverse KL. Query transport owns board and clocks per tree. Cached repeats incur no new neural calls.')
    fp=folder/'plan.json'
    if fp.exists():assert json.loads(fp.read_text())==config
    else:atomic(fp,config)
    start=time.monotonic()
    for mode in config['arms']:
        for lo in range(0,len(rows),128):
            path=folder/f'{mode}-{lo:06d}.npz'
            if path.exists():continue
            if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
            part=rows[lo:lo+128];oracle.reset();tick=time.monotonic()
            bridge=ShipHandles(oracle,[r['prefix'] for r in part],features[lo:lo+len(part)])
            prefill=oracle.new_tokens
            if mode=='shallow':q,nodes=shallow(part,bridge);prob=np.empty(0);stat={}
            else:q,prob,nodes,stat=mcts(part,bridge,mode)
            stat.update(seconds=time.monotonic()-tick,prefill_tokens=prefill,forward_seconds=oracle.forward_seconds,root_queries=len(part))
            tmp=path.with_suffix('.partial')
            with tmp.open('wb') as f:np.savez_compressed(f,root=bridge.root_logits,q=q,policy=prob,nodes=nodes,stats=json.dumps(stat),game=[r['game'] for r in part],ply=[r['ply'] for r in part])
            tmp.replace(path)
            if lo%512==0:print('TRANSFER BASELINE',size,split,mode,lo+len(part),len(rows),round(stat['seconds'],3),flush=True)
    return dict(folder=str(folder),seconds=time.monotonic()-start)
