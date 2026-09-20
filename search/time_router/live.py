"""Actual mixed-budget execution, using frozen root-only allocations."""
import json,os,time
import numpy as np
from .collect import *
from .analysis import ROUTER_BUDGETS,frozen_selection

def run(oracle,spec):
 alloc=json.loads((OUT/'gold-allocations.json').read_text());frozen=frozen_selection();name=spec['method'];base=name.replace('-within-cell','');r=frozen['methods'][base];tag=r['tag'];kind='coverage' if tag=='coverage' else 'allie';cp=2.5 if kind=='coverage' else float(tag.split('-')[1])
 rows,features,hashes=inventory('gold');choices=np.array(alloc['choice'][name]);budgets=ROUTER_BUDGETS[choices];bs=256;n=len(rows)
 folder=OUT/('live-'+name);folder.mkdir(exist_ok=True);plan=dict(spec=spec,allocation_sha256=sha(OUT/'gold-allocations.json'),frozen_sha256=sha(OUT/'frozen.json'),source=sha(__file__),inventory=hashes,batch=bs)
 pp=folder/'plan.json'
 if pp.exists():assert json.loads(pp.read_text())==plan
 else:atomic(pp,plan)
 mod=coverage() if kind=='coverage' else native();reducer=backup() if kind=='coverage' else None
 for lo in range(0,n,bs):
  dst=folder/f'{lo:06d}.npz'
  if dst.exists():continue
  if (OUT/'STOP').exists() or GLOBAL_STOP.exists():raise RuntimeError('STOP')
  part=rows[lo:lo+bs];nr=len(part);width=max(len(r['legal']) for r in part);ids=np.zeros((nr,width),np.int32);mask=np.zeros_like(ids,bool)
  for i,row in enumerate(part):ids[i,:len(row['legal'])]=np.array(row['legal'])-378;mask[i,:len(row['legal'])]=True
  forced=mask.sum(1)==1;ns=np.where(forced,0,budgets[lo:lo+nr]);oracle.reset();tick=time.monotonic();bridge=ShipHandles(oracle,[r['prefix'] for r in part],features[lo:lo+nr]);z=bridge.root_logits.copy()
  args=([r['prefix'] for r in part],z,ns.tolist(),[cp]*nr);tree=mod.Tree(*args,4) if kind=='coverage' else mod.Tree(*args)
  if kind=='allie':tree.first_prior=tree.preserve_depth=True
  cache={};owner={i:i for i in range(nr)};count=np.zeros(nr,int)
  while not tree.done:
   h=tree.select()
   if not len(h):continue
   if kind=='coverage':tree.update(bridge(h))
   else:
    for node,parent,_,_ in h:owner[int(node)]=owner[int(parent)]
    take=np.array([int(a[0]) not in cache for a in h])
    if take.any():
     zz=bridge(h[take])
     for a,value in zip(h[take],zz):cache[int(a[0])]=value;count[owner[int(a[0])]]+=1
    tree.update(np.array([cache[int(a[0])] for a in h]))
  if kind=='coverage':q=reducer.Backup(tree.compact(),1000,16.).reduce(np.log(.2),-.5,ids)[0];count=np.array(tree.evals)
  else:
   q=np.zeros(ids.shape)
   for i,(legal,visits,value,prior) in enumerate(tree.summaries()):
    lookup=dict(zip(legal,value));q[i,:mask[i].sum()]=[lookup[int(t)] for t in ids[i,mask[i]]]
  assert int(count.sum())==bridge.queries
  stat=tree.stats();stat.update(seconds=time.monotonic()-tick,forward_seconds=oracle.forward_seconds,prefill_tokens=sum(len(r['prefix']) for r in part),job=os.environ.get('SLURM_JOB_ID'),host=os.environ.get('SLURMD_NODENAME'))
  del tree
  tmp=dst.with_suffix('.partial')
  with tmp.open('wb') as f:np.savez_compressed(f,root=z,q=q,nodes=count,ids=ids,mask=mask,allocated=ns,game=[r['game'] for r in part],ply=[r['ply'] for r in part],stats=json.dumps(stat))
  tmp.replace(dst);print('ROUTER LIVE',name,lo+nr,n,round(stat['seconds'],2),flush=True)
 return dict(folder=str(folder))
