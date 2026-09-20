"""Immutable budget grids; no move label enters tree or budget decisions."""
import hashlib,importlib,json,os,subprocess,sys,sysconfig,time
from pathlib import Path
import numpy as np
from search.engine.service import ROOT,GLOBAL_STOP,atomic
from search.engine.native_board import MOVES
from search.engine.growforest_native import load as coverage
from search.engine.scaled_count_native import load as backup
from search.transfer.collect import inventory,freeze
from search.transfer.handles import ShipHandles

OUT=ROOT/'time-router-v1'
BUDGETS=[64,128,256,512,1000]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def native():
 import pybind11
 folder=OUT/'native';folder.mkdir(parents=True,exist_ok=True)
 source=Path(__file__).with_name('native.cpp');dest=folder/('_allie_time_router'+sysconfig.get_config_var('EXT_SUFFIX'))
 sources=[source,*[source.parent.parent/'engine'/x for x in ('board.cpp','mcts_native.hpp')]]
 signature={str(p):sha(p) for p in sources}|{'python':sys.version}
 stamp=folder/('build-'+sysconfig.get_config_var('SOABI')+'.json')
 if not dest.exists() or not stamp.exists() or json.loads(stamp.read_text())!=signature:
  temp=dest.with_suffix('.new')
  subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(source),'-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),'-I'+str(source.parents[2]/'vendor/chess-library/include'),'-o',str(temp)],check=True)
  temp.replace(dest);atomic(stamp,signature)
 sys.path.insert(0,str(folder));m=importlib.import_module('_allie_time_router');m.initialize(MOVES);return m

def run(oracle,spec):
 rows,features,hashes=inventory(spec['split']);n=len(rows);bs=int(spec.get('batch',256));kind=spec['kind'];cp=float(spec.get('cpuct',2.5))
 assert spec['model']=='large' and kind in ('coverage','allie')
 tag='coverage' if kind=='coverage' else 'allie-'+str(cp)
 folder=OUT/(spec['split']+'-'+tag);folder.mkdir(parents=True,exist_ok=True)
 plan=dict(spec=spec,budgets=BUDGETS,inventory=hashes,source=sha(__file__),native=sha(Path(__file__).with_name('native.cpp')),transfer_plan=sha(ROOT/'transfer-v1/plan.json'),hardware=os.environ.get('SLURMD_NODENAME'))
 pp=folder/'plan.json'
 if pp.exists():
  old=json.loads(pp.read_text());assert {k:v for k,v in old.items() if k!='hardware'}=={k:v for k,v in plan.items() if k!='hardware'},'Changed source/config'
 else:atomic(pp,plan)
 mod=coverage() if kind=='coverage' else native();reducer=backup() if kind=='coverage' else None
 start=time.monotonic()
 for lo in range(0,n,bs):
  path=folder/f'{lo:06d}.npz'
  if path.exists():continue
  if (OUT/'STOP').exists() or GLOBAL_STOP.exists():raise RuntimeError('STOP')
  part=rows[lo:lo+bs];nr=len(part);width=max(len(r['legal']) for r in part);ids=np.zeros((nr,width),np.int32);mask=np.zeros_like(ids,bool)
  for i,r in enumerate(part):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True
  forced=np.array([len(r['legal'])==1 for r in part]);oracle.reset();tick=time.monotonic();bridge=ShipHandles(oracle,[r['prefix'] for r in part],features[lo:lo+nr])
  z=bridge.root_logits.copy();args=([r['prefix'] for r in part],z,np.where(forced,0,BUDGETS[0]).tolist(),[cp]*nr)
  tree=mod.Tree(*args,4) if kind=='coverage' else mod.Tree(*args)
  if kind=='allie':tree.first_prior=tree.preserve_depth=True
  cache={};owners={i:i for i in range(nr)};counts=np.zeros(nr,int);qs=[];nodes=[]
  for budget in BUDGETS:
   if budget!=BUDGETS[0]:tree.grow(np.where(forced,0,budget).tolist())
   while not tree.done:
    h=tree.select()
    if not len(h):continue
    if kind=='coverage':tree.update(bridge(h))
    else:
     for node,parent,_,_ in h:owners[int(node)]=owners[int(parent)]
     take=np.array([int(a[0]) not in cache for a in h])
     if take.any():
      zz=bridge(h[take])
      for a,value in zip(h[take],zz):cache[int(a[0])]=value;counts[owners[int(a[0])]]+=1
     tree.update(np.array([cache[int(a[0])] for a in h]))
   if kind=='coverage':q=reducer.Backup(tree.compact(),budget,16.).reduce(np.log(.2),-.5,ids)[0];count=np.array(tree.evals)
   else:
    q=np.zeros(ids.shape)
    for i,(legal,visits,value,prior) in enumerate(tree.summaries()):
     lookup=dict(zip(legal,value));q[i,:mask[i].sum()]=[lookup[int(t)] for t in ids[i,mask[i]]]
    count=counts.copy()
   qs.append(q);nodes.append(count)
  assert int(nodes[-1].sum())==bridge.queries
  stats=tree.stats();stats.update(seconds=time.monotonic()-tick,job=os.environ.get('SLURM_JOB_ID'),host=os.environ.get('SLURMD_NODENAME'),prefill_tokens=sum(len(r['prefix']) for r in part),forward_seconds=oracle.forward_seconds)
  del tree
  tmp=path.with_suffix('.partial')
  with tmp.open('wb') as f:np.savez_compressed(f,root=z,q=np.array(qs),nodes=np.array(nodes),ids=ids,mask=mask,game=[r['game'] for r in part],ply=[r['ply'] for r in part],stats=json.dumps(stats))
  tmp.replace(path);print('ROUTER GRID',tag,spec['split'],lo+nr,n,round(stats['seconds'],2),flush=True)
 return dict(folder=str(folder),seconds=time.monotonic()-start)
