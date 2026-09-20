"""Authoritative live scores and the numerical gap to cached stops."""
import json,sys
import numpy as np
from .analysis import *
from .evaluate import report

def readout(d,q,root,r,choice):
 p=np.zeros_like(q);plan=json.loads((ROOT/'transfer-v1/plan.json').read_text());z=np.where(d['mask'],root[:,378:2346][np.arange(len(root))[:,None],d['ids']],0.)
 for j,b in enumerate(ROUTER_BUDGETS):
  take=choice==j
  if not take.any():continue
  if r['tag']=='coverage':
   p[take]=policy([d['rows'][i] for i in np.flatnonzero(take)],root[take],q[take],d['ids'][take],d['mask'][take],d['seconds'][take],plan['budget_policies'][str(b)])
  else:
   for g in range(4):
    t=take&(d['cells']%4==g)
    if not t.any():continue
    par=r['head'][str(b)][str(g)];p[t]=loss_gradient([par['alpha'],par['beta']],z[t],q[t],d['mask'][t],d['target'][t],'reverse',return_policy=True)[0]
 assert np.allclose(p.sum(1),1,atol=1e-9) and (p[np.arange(len(p)),d['target']]>0).all()
 return p

def main():
 frozen=frozen_selection();alloc=json.loads((OUT/'gold-allocations.json').read_text());d=meta('gold');n=len(d['rows']);scores={};nodes={};difference={};dm={};timing={}
 with np.load(OUT/'cached-scores.npz') as f:
  for name in ['port-raw','legal']:
   i=list(f['names']).index(name);scores[name]=f['loss'][i];nodes[name]=f['nodes'][i]
 for name,choice in alloc['choice'].items():
  r=frozen['methods'][name.replace('-within-cell','')];folder=OUT/('live-'+name)
  if '--completed-only' in sys.argv and not (folder/f'{n-256:06d}.npz').exists():continue
  root=np.zeros((n,2432));q=np.zeros(d['ids'].shape);count=np.zeros(n);times=[]
  for lo in range(0,n,256):
   with np.load(folder/f'{lo:06d}.npz') as f:
    nr=len(f['game']);hi=lo+nr;k=f['ids'].shape[1];np.testing.assert_array_equal(f['game'],d['games'][lo:hi]);np.testing.assert_array_equal(f['ids'],d['ids'][lo:hi,:k]);q[lo:hi,:k]=f['q'];root[lo:hi]=f['root'];count[lo:hi]=f['nodes'];times.append(json.loads(str(f['stats'])))
  choice=np.array(choice);p=readout(d,q,root,r,choice);scores[name]=-np.log(p[np.arange(n),d['target']]);nodes[name]=count
  if r['tag'] not in dm:dm[r['tag']]=load('gold',r['tag'])
  cached=dm[r['tag']];cq=cached['q'][choice+1,np.arange(n)];cp=readout(d,cq,cached['root'],r,choice)
  cd=-np.log(cp[np.arange(n),d['target']]);gap=scores[name]-cd;cm=cellmean(gap,d['cells']);boot=bootstrap_deltas(gap[:,None],d['cells'],d['games'])[:,:,0]
  kl=(np.where(p>0,p,0)*np.log(np.maximum(p,1e-300)/np.maximum(cp,1e-300))).sum(1)
  difference[name]=dict(max_policy_diff=float(np.abs(p-cp).max()),max_kl=float(kl.max()),macro_delta=float(cm.mean()),expert_delta=float(cm[3::4].mean()),macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist(),nodes_delta=float(cellmean(count-cached['nodes'][choice+1,np.arange(n)],d['cells']).mean()))
  timing[name]=dict(seconds=sum(t['seconds'] for t in times),forward_seconds=sum(t['forward_seconds'] for t in times),prefill_tokens=sum(t['prefill_tokens'] for t in times),host=times[0]['host'],job=times[0]['job'])
  print('LIVE CHECK',name,difference[name],flush=True)
 tag='live-partial' if '--completed-only' in sys.argv else 'live'
 np.savez_compressed(OUT/(tag+'-scores.npz'),names=list(scores),loss=np.stack(list(scores.values())),nodes=np.stack(list(nodes.values())),cells=d['cells'],games=d['games'])
 report(scores,nodes,d,tag,dict(status='PARKED by user; only completed arms scored' if '--completed-only' in sys.argv else 'complete',frozen_sha256=sha(OUT/'frozen.json'),allocation_sha256=sha(OUT/'gold-allocations.json'),cached_vs_live=difference,timing=timing,authoritative='Actual mixed-budget reruns; cached-stop estimates are diagnostic only.'))

if __name__=='__main__':main()
