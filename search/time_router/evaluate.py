"""Frozen routers, query-only cost calibration, and paired July estimation."""
import argparse,json
import numpy as np
from scipy.special import logsumexp
from .analysis import *
from search.transfer.context import predicted_seconds

def allocation(d,r,cap=TARGET,per_cell=None,aggregate_actual=False):
 pars=r['parameters'];cost=costs(d,pars['cost_table'])
 if pars['kind']=='think':
  # Nearest log-budget to scale times root predicted thinking seconds; one
  # unlabeled scale sets the mean query cost. No move targets are inspected.
  seconds=np.maximum(predicted_seconds(d['root']),.01);actual=d['nodes'][1:].T
  def choose(scale):return np.argmin((np.log(ROUTER_BUDGETS)[None,:]-np.log(seconds*scale)[:,None])**2+d['jitter'],axis=1)
  w=weights(d['cells']);cc=actual if aggregate_actual else cost;lo,hi=1e-9,1e9
  for _ in range(80):
   mid=np.sqrt(lo*hi);a=choose(mid)
   if w@cc[np.arange(len(a)),a]>cap:hi=mid
   else:lo=mid
  return choose(lo),dict(scale=lo,aggregate_actual_calibration=aggregate_actual,rule='Budget proportional to root predicted think seconds, clipped/discretized to128/256/512/1000. Same development-selected cpuct and reverse-KL heads as Allie Elo.')
 x,_=design(d,pars['kind'],normalizer=pars['normalizer']);pred=x@np.asarray(pars['coef'])
 def run(take,limit):
  p,c,j=pred[take],cost[take],d['jitter'][take];cc=d['cells'][take];w=np.ones(len(cc)) if per_cell is not None else weights(cc)
  choice,penalty,expected=allocate(p,c,j,cc,limit,w=w)
  # OPTIONAL unlabeled aggregate calibration. The routing rule still sees only
  # root features and the development cost table, never per-position future cost.
  if aggregate_actual:
   actual=d['nodes'][1:].T[take];w=w/w.sum();choose=lambda a:np.argmin(p+j+a*c/1000,axis=1)
   def measure(a):s=choose(a);return float(w@actual[np.arange(len(s)),s])
   lo,hi=-1.,1.
   for _ in range(50):
    if measure(lo)>=limit:break
    lo*=2
   for _ in range(50):
    if measure(hi)<=limit:break
    hi*=2
   for _ in range(64):
    mid=(lo+hi)/2
    if measure(mid)>limit:lo=mid
    else:hi=mid
   choice=choose(hi);penalty=hi
  return choice,penalty
 if per_cell is None:
  choice,penalty=run(np.ones(len(pred),bool),cap);return choice,dict(penalty=penalty,aggregate_actual_calibration=aggregate_actual)
 choice=np.zeros(len(pred),int);penalty=[]
 for c in range(16):
  take=d['cells']==c;choice[take],p=run(take,float(per_cell[c]));penalty.append(p)
 return choice,dict(per_cell_penalty=penalty,aggregate_actual_calibration=aggregate_actual)

def main():
 frozen=frozen_selection();dm={t:load('gold',t) for t in set(r['tag'] for r in frozen['methods'].values())};d=dm['coverage'];n=len(d['rows']);ix=np.arange(n)
 methods={};choices={};lls={};allnodes={};source_hashes={}
 for name,r in frozen['methods'].items():
  dd=dm[r['tag']];ll,_=head_losses(dd,r['tag'],np.zeros(n,bool),r['head']);lls[name]=ll
  a,info=allocation(dd,r);nodes=dd['nodes'][1:][a,ix];choices[name]=a
  methods[name]=dict(tag=r['tag'],**info,initial_mean_nodes=float(cellmean(nodes,d['cells']).mean()))
 # Apply the same extra label-free calibration to every arm if any drift >2%.
 recalibrate=any(abs(r['initial_mean_nodes']/TARGET-1)>.02 for r in methods.values())
 for name,r in frozen['methods'].items():
  if recalibrate:choices[name],info=allocation(dm[r['tag']],r,aggregate_actual=True);methods[name].update(info)
  a=choices[name];allnodes[name]=dm[r['tag']]['nodes'][1:][a,ix]
 # Time vs Elo at matched PER-CELL cost separates reallocation across cells.
 for family in ('coverage','allie'):
  name=family+'-time-within-cell';r=frozen['methods'][family+'-time'];target=cellmean(allnodes[family+'-elo'],d['cells'])
  choices[name],info=allocation(dm[r['tag']],r,per_cell=target,aggregate_actual=True)
  methods[name]=dict(tag=r['tag'],reference_cell_nodes=target.tolist(),**info);lls[name]=lls[family+'-time'];allnodes[name]=dm[r['tag']]['nodes'][1:][choices[name],ix]
 # Freeze choices BEFORE computing golden losses. None of allocation reads target.
 record=dict(methods=methods,frozen_sha256=sha(OUT/'frozen.json'),inventory=d['hashes'],choice={k:v.tolist() for k,v in choices.items()},selection='Frozen development models; optional aggregate query-cost calibration is label-free. No golden-quality selection.')
 pp=OUT/'gold-allocations.json'
 if pp.exists():assert json.loads(pp.read_text())==record
 else:atomic(pp,record)
 scores={name:lls[name][ix,a] for name,a in choices.items()}
 for family in ('coverage','allie'):
  name=family+'-elo';dd=dm[frozen['methods'][name]['tag']]
  for j,b in enumerate(ROUTER_BUDGETS):scores[f'{family}-fixed{b}']=lls[name][:,j];allnodes[f'{family}-fixed{b}']=dd['nodes'][j+1]
 legal=logsumexp(np.where(d['mask'],d['z'],-np.inf),1)-d['z'][ix,d['target']]
 raw=logsumexp(d['root'][:,378:2346],1)-d['root'][ix,np.array([r['target'] for r in d['rows']])]
 scores={'port-raw':raw,'legal':legal,**scores};allnodes={'port-raw':np.zeros(n),'legal':np.zeros(n),**allnodes}
 report(scores,allnodes,d,'cached',record)
 np.savez_compressed(OUT/'cached-scores.npz',names=list(scores),loss=np.stack(list(scores.values())),nodes=np.stack(list(allnodes.values())),cells=d['cells'],games=d['games'])

def report(scores,nodes,d,tag,provenance):
 canonical=json.loads((ROOT/'transfer-v1/canonical-large/results.json').read_text());full=np.array(canonical['cells']);n=len(d['rows']);cells=d['cells'];games=d['games'];names=list(scores)
 with np.load(ROOT/'transfer-v1/canonical-large/sample.npz') as f:
  np.testing.assert_array_equal(f['game'],games);z=f['logits'][:,378:2346].astype(float)
 y=np.array([r['target']-378 for r in d['rows']]);ref=logsumexp(z,1)-z[np.arange(n),y]
 delta=np.stack(list(scores.values()),1)-ref[:,None];boot=bootstrap_deltas(delta,cells,games);draws=full[None,:,None]+boot;points=full[:,None]+np.stack([cellmean(delta[:,i],cells) for i in range(len(names))],1)
 laws=json.loads((ROOT/'training-cm-laws.json').read_text());results={};ci=lambda x:np.quantile(x,[.025,.975]).tolist();legal=names.index('legal')
 for i,name in enumerate(names):
  cost=cellmean(nodes[name],cells);row=dict(mean_nodes=float(cost.mean()),expert_nodes=float(cost[3::4].mean()),cell_nodes=cost.tolist(),cells=points[:,i].tolist())
  for metric,sel in [('macro',np.arange(16)),('expert_macro',np.arange(3,16,4))]:
   law=laws['metrics'][metric]['law'];cm=lambda x:multiplier(law,3e17,canonical[metric],float(x));value=float(points[sel,i].mean());lb,ub=ci(draws[:,sel,i].mean(1))
   row[metric]=value;row[metric+'_ci95']=[lb,ub];row[metric+'_cm']=cm(value);row[metric+'_cm_ci95']=[cm(ub),cm(lb)]
   row[metric+'_cm_vs_legal']=cm(value)/cm(points[sel,legal].mean())
   for other in ['legal','coverage-elo','allie-elo','coverage-time','allie-time']:
    if other in names:
     j=names.index(other);row[metric+'_delta_vs_'+other]=float((points[sel,i]-points[sel,j]).mean());row[metric+'_delta_vs_'+other+'_ci95']=ci((draws[:,sel,i]-draws[:,sel,j]).mean(1))
  # Allocation diagnostics. Clock groups pooled within each available format;
  # primary scores above always weight all sixteen cells equally.
  row['by_format']={str(g):dict(nodes=float(cost[4*g:4*g+4].mean()),ce=float(points[4*g:4*g+4,i].mean())) for g in range(4)}
  clock=d['seconds'];bins=np.where(clock<0,0,np.where(clock<=15,1,np.where(clock<=60,2,3)))
  row['clock_buckets']={str(b):dict(positions=int(sum(bins==b)),nodes=float(np.mean(nodes[name][bins==b])),sample_ce=float(np.mean(scores[name][bins==b]))) for b in np.unique(bins)}
  results[name]=row
 record=dict(methods=results,provenance=provenance,canonical=canonical,positions=n,games=len(set(games)),note='Full canonical cell means plus paired July8192 differences. Reused golden, no new confirmation. CIs resample whole games; exclude law-fit and search numerical uncertainty. CM conditional on prior law shape at rung3e17; search-only CM vs legal separately. Cached stops require live validation.')
 atomic(OUT/(tag+'-results.json'),record)
 for name,r in results.items():print(tag,name,*(round(r[k],6) for k in ['macro','expert_macro','macro_cm_vs_legal','expert_macro_cm_vs_legal','mean_nodes']),flush=True)

if __name__=='__main__':main()
