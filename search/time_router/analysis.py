"""Development-only selection, followed by frozen, actual-cost reporting."""
import argparse,functools,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.special import softmax,logsumexp
from search.engine.service import ROOT,atomic
from search.engine.analyze_balanced import cellmean,bootstrap_deltas
from search.engine.budget_policy import policy,route
from search.engine.fit_policy import fit,loss_gradient
from search.training_cm import multiplier
from search.transfer.collect import inventory
from .collect import OUT,BUDGETS,sha

ROUTER_BUDGETS=np.array([128,256,512,1000]);TARGET=460.

@functools.lru_cache(None)
def meta(split):
 rows,features,hashes=inventory(split);n=len(rows);k=max(len(r['legal']) for r in rows)
 ids=np.zeros((n,k),np.int32);mask=np.zeros((n,k),bool);target=np.zeros(n,int)
 for i,r in enumerate(rows):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True;target[i]=r['legal'].index(r['target'])
 games=np.array([r['game'] for r in rows]);cells=np.array([r['cell'] for r in rows]);fitmask=np.array([r.get('fold',-1)==0 for r in rows]);cv=np.array([int(hashlib.sha256(('cv:'+g).encode()).hexdigest(),16)%3 for g in games])
 # Query-owned reproducible ties; this does not depend on the move being predicted.
 jitter=np.array([[int.from_bytes(hashlib.sha256(np.asarray(r['prefix'],np.int16).tobytes()+bytes([j])).digest()[:8],'little')/2**64 for j in range(4)] for r in rows])*1e-8
 seconds_table=np.r_[0,15,30,45,60,90,np.arange(2,181)*60,-1]
 base=np.array([seconds_table[r['prefix'][1]-192] for r in rows]);inc=np.array([r['prefix'][2]-10 if r['prefix'][2]<191 else -1 for r in rows])
 feat=np.array([f[-1] for f in features]);clock=feat[:,0];known=clock>=0
 raw=np.column_stack([np.log1p(np.maximum(base,0)),np.log1p(np.maximum(inc,0)),np.log1p(np.maximum(clock,0)),
  np.log1p(np.maximum(feat[:,1],0)),np.log1p(np.maximum(feat[:,2],0)),np.where(known,np.clip(clock/np.maximum(base,1),0,3),0),
  known&(clock<=15),known&(clock<=5),~known,feat[:,2]<0,base<0,inc<0])
 return dict(rows=rows,ids=ids,mask=mask,target=target,cells=cells,games=games,fit=fitmask,cv=cv,jitter=jitter,time=raw,seconds=clock,base=base,increment=inc,forced=mask.sum(1)==1,hashes=hashes)

def load(split,tag):
 d=meta(split);n=len(d['rows']);q=np.zeros((5,n,d['ids'].shape[1]));root=np.zeros((n,2432));nodes=np.zeros((5,n));seconds=0.
 folder=OUT/(split+'-'+tag);plan=json.loads((folder/'plan.json').read_text());bs=plan['spec']['batch']
 for lo in range(0,n,bs):
  with np.load(folder/f'{lo:06d}.npz') as f:
   nr=len(f['game']);hi=lo+nr;k=f['ids'].shape[1];np.testing.assert_array_equal(f['game'],d['games'][lo:hi]);np.testing.assert_array_equal(f['ids'],d['ids'][lo:hi,:k])
   root[lo:hi]=f['root'];q[:,lo:hi,:k]=f['q'];nodes[:,lo:hi]=f['nodes'];seconds+=json.loads(str(f['stats']))['seconds']
 z=np.where(d['mask'],root[:,378:2346][np.arange(n)[:,None],d['ids']],0.)
 return dict(**d,q=q,root=root,z=z,nodes=nodes,seconds_collected=seconds)

def weights(cells):
 count=np.bincount(cells,minlength=16);assert (count>0).all();return 1/(16*count[cells])

def design(d,kind,train=None,normalizer=None):
 elo=np.eye(4)[d['cells']%4]
 if kind=='elo':raw=elo[:,1:]
 else:
  raw=np.eye(16)[d['cells']][:,1:]
  if kind=='time':raw=np.c_[raw,(elo[:,:,None]*d['time'][:,None,:]).reshape(len(elo),-1)]
 if normalizer is None:normalizer=dict(mean=raw[train].mean(0).tolist(),scale=np.maximum(raw[train].std(0),1e-6).tolist())
 x=np.c_[np.ones(len(raw)),np.clip((raw-normalizer['mean'])/normalizer['scale'],-4,4)]
 return x,normalizer

def cost_model(d,train):
 # Available before a query: development mean costs by budget and public cell.
 active=train&~d['forced'];cost=d['nodes'][1:].T;global_mean=np.average(cost[active],axis=0,weights=weights(d['cells'][active]))
 table=[]
 for c in range(16):
  take=active&(d['cells']==c);table.append(((cost[take].sum(0)+64*global_mean)/(take.sum()+64)).tolist())
 return table

def costs(d,table):return np.asarray(table)[d['cells']]*~d['forced'][:,None]

def allocate(pred,cost,jitter,cells,cap=TARGET,w=None):
 w=weights(cells) if w is None else np.asarray(w)/np.sum(w);floor=float(w@cost.min(1));ceiling=float(w@cost.max(1));cap=float(np.clip(cap,floor,ceiling));choose=lambda penalty:np.argmin(pred+jitter+penalty*cost/1000,axis=1)
 def measure(p):a=choose(p);return float(w@cost[np.arange(len(a)),a])
 if cap<=floor+1e-9:
  a=np.argmin(cost,axis=1);return a,float('inf'),floor
 if cap>=ceiling-1e-9:
  a=np.argmax(cost,axis=1);return a,float('-inf'),ceiling
 low,high=-1.,1.
 while measure(low)<cap:low*=2
 while measure(high)>cap:high*=2
 for _ in range(64):
  mid=(low+high)/2
  if measure(mid)>cap:low=mid
  else:high=mid
 a=choose(high);return a,float(high),float(w@cost[np.arange(len(a)),a])

def head_losses(d,tag,train,params=None):
 losses=np.zeros((len(d['rows']),4));output={};plan=json.loads((ROOT/'transfer-v1/plan.json').read_text())
 for j,b in enumerate(ROUTER_BUDGETS):
  q=d['q'][j+1]
  if tag=='coverage':
   p=policy(d['rows'],d['root'],q,d['ids'],d['mask'],d['seconds'],plan['budget_policies'][str(b)])
   losses[:,j]=-np.log(p[np.arange(len(p)),d['target']])
  else:
   output[str(b)]={}
   for g in range(4):
    a=train&(d['cells']%4==g);take=d['cells']%4==g
    if params is None:
     ct=np.bincount(d['cells'][a],minlength=16);par=fit(d['z'][a],q[a],d['mask'][a],d['target'][a],'reverse',1/ct[d['cells'][a]])
     assert par['converged'],par
    else:par=params[str(b)][str(g)]
    _,nll=loss_gradient([par['alpha'],par['beta']],d['z'][take],q[take],d['mask'][take],d['target'][take],'reverse',return_policy=True)
    losses[take,j]=nll;output[str(b)][str(g)]=par
 return losses,output

def regression(x,y,cells,ridge):
 w=weights(cells);reg=np.eye(x.shape[1])*ridge;reg[0,0]=1e-10
 return np.linalg.solve(x.T@(w[:,None]*x)+reg,x.T@(w[:,None]*(y-y[:,:1])))

def train(tag):
 d=load('aug',tag);fm=d['fit'];cfg=json.loads((OUT/'plan.json').read_text());candidates={};heads={};lls={}
 assert not set(d['games'][fm])&set(d['games'][~fm])
 for fold in [0,1,2,'full']:
  tr=fm if fold=='full' else fm&(d['cv']!=fold)
  ll,head=head_losses(d,tag,tr);heads[str(fold)]=head;lls[fold]=ll
  print('HEAD',tag,fold,flush=True)
 for kind in cfg['routers']:
  for ridge in cfg['ridge']:
   key=kind+'-'+str(ridge);oof=np.full(len(fm),np.nan);nc=np.full(len(fm),np.nan);allparams=[]
   for fold in [0,1,2,'full']:
    tr=fm if fold=='full' else fm&(d['cv']!=fold);val=~fm if fold=='full' else fm&(d['cv']==fold)
    x,norm=design(d,kind,tr);coef=regression(x[tr],lls[fold][tr],d['cells'][tr],ridge);table=cost_model(d,tr)
    selected,penalty,expected=allocate((x@coef)[val],costs(d,table)[val],d['jitter'][val],d['cells'][val])
    ids=np.flatnonzero(val);loss=lls[fold][ids,selected];cost=d['nodes'][1:][selected,ids]
    pars=dict(kind=kind,ridge=ridge,normalizer=norm,coef=coef.tolist(),cost_table=table,penalty=penalty,expected_nodes=expected)
    if fold=='full':full=dict(parameters=pars,head=heads['full'],confirmation_cells=cellmean(loss,d['cells'][val]).tolist(),confirmation_nodes=float(cellmean(cost,d['cells'][val]).mean()),confirmation_choice=selected.tolist())
    else:oof[val]=loss;nc[val]=cost
   cv_cells=cellmean(oof[fm],d['cells'][fm]);record=dict(**full,cv_macro=float(cv_cells.mean()),cv_expert=float(cv_cells[3::4].mean()),cv_nodes=float(cellmean(nc[fm],d['cells'][fm]).mean()))
   candidates[key]=record
   print('ROUTER',tag,key,'CV',record['cv_macro'],record['cv_expert'],'confirm',np.mean(record['confirmation_cells']),np.mean(record['confirmation_cells'][3::4]),'nodes',record['confirmation_nodes'],flush=True)
 # Save full losses for paired confirmation; selection itself only uses CV.
 np.savez_compressed(OUT/(tag+'-dev-losses.npz'),loss=lls['full'],cells=d['cells'],games=d['games'],nodes=d['nodes'][1:],fit=fm)
 result=dict(tag=tag,candidates=candidates,source_sha256=sha(__file__),study_plan_sha256=sha(OUT/'plan.json'),collection_sha256=sha(OUT/('aug-'+tag)/'plan.json'))
 atomic(OUT/(tag+'-development.json'),result)
 return result

def select():
 tags=['coverage','allie-0.5','allie-1.25','allie-2.5'];reports={t:json.loads((OUT/(t+'-development.json')).read_text()) for t in tags};selected={}
 best_allie=min((r['cv_macro'],t) for t,s in reports.items() if t.startswith('allie') for r in s['candidates'].values() if r['parameters']['kind']=='elo')[1]
 for family in ['coverage','allie']:
  for kind in ['elo','format','time']:
   possibilities=[(r['cv_macro'],t,k,r) for t,s in reports.items() if t==('coverage' if family=='coverage' else best_allie) for k,r in s['candidates'].items() if r['parameters']['kind']==kind]
   _,tag,key,r=min(possibilities,key=lambda a:(a[0],a[1],a[2]));selected[family+'-'+kind]=dict(tag=tag,candidate=key,**r)
 selected['allie-predicted-time']=dict(selected['allie-elo']);selected['allie-predicted-time']['parameters']=dict(selected['allie-elo']['parameters'],kind='think')
 record=dict(methods=selected,selection='Allie cpuct selected on Elo-router fit-game CV, then held fixed across router families. Ridge selected by fit-game CV within router family. Confirmation/July losses never choose parameters.',source_sha256=sha(__file__),development_hashes={t:sha(OUT/(t+'-development.json')) for t in tags})
 p=OUT/'frozen.json';assert not p.exists(),'Do not overwrite a frozen selection';atomic(p,record)
 for key,r in selected.items():print('FROZEN',key,r['tag'],r['candidate'],r['cv_macro'],np.mean(r['confirmation_cells']),np.mean(r['confirmation_cells'][3::4]),r['confirmation_nodes'],flush=True)

def frozen_selection():
 record=json.loads((OUT/'frozen.json').read_text())
 if 'allie-predicted-time' not in record['methods']:
  r=dict(record['methods']['allie-elo']);r['parameters']=dict(r['parameters'],kind='think');record['methods']['allie-predicted-time']=r
 record['think_time_addendum_sha256']=sha(OUT/'think-time-addendum.json')
 return record

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['train','select']);ap.add_argument('tag',nargs='?');a=ap.parse_args();train(a.tag) if a.action=='train' else select()
