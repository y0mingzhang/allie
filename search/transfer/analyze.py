"""Calibration-only development fit and paired transfer estimates. No golden fitting."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.special import softmax,logsumexp
from scipy.optimize import minimize_scalar
from search.engine.service import ROOT,atomic
from search.engine.balanced_eval import digest
from search.engine.analyze_balanced import cellmean,bootstrap_deltas
from search.engine.fit_policy import fit,loss_gradient
from search.engine.budget_policy import policy
from search.training_cm import multiplier,excess
from .collect import OUT,inventory,freeze,BUDGETS
from .canonical import BASE,MODELS


def data(size,split):
    rows,features,_=inventory(split);n=len(rows);width=max(len(r['legal']) for r in rows)
    ids=np.zeros((n,width),int);mask=np.zeros((n,width),bool)
    for i,r in enumerate(rows):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True
    target=np.array([r['legal'].index(r['target']) for r in rows]);root=np.zeros((n,2432));q={str(b):np.zeros((n,width)) for b in BUDGETS};q['projected']=np.zeros((n,width))
    nodes={k:np.zeros(n) for k in q};timings=[];folder=OUT/f'{size}-{split}-fixed-predicted'
    for lo in range(0,n,128):
        with np.load(folder/f'{lo:06d}.npz') as f:
            hi=lo+len(f['game']);k=f['ids'].shape[1]
            np.testing.assert_array_equal(f['game'],[r['game'] for r in rows[lo:hi]])
            np.testing.assert_array_equal(f['ply'],[r['ply'] for r in rows[lo:hi]])
            np.testing.assert_array_equal(f['ids'],ids[lo:hi,:k]);root[lo:hi]=f['root']
            for j,b in enumerate(BUDGETS):q[str(b)][lo:hi,:k]=f['q'][j];nodes[str(b)][lo:hi]=f['nodes'][j]
            q['projected'][lo:hi,:k]=f['projected_q'];nodes['projected'][lo:hi]=f['nodes'][-1]
            timings.append(json.loads(str(f['stats'])))
    z=np.where(mask,root[:,378:2346][np.arange(n)[:,None],ids],0.)
    return dict(rows=rows,root=root,z=z,q=q,nodes=nodes,ids=ids,mask=mask,target=target,
        seconds=np.array([f[-1,0] for f in features]),cells=np.array([r['cell'] for r in rows]),timings=timings)


def fitted(z,q,mask,cells,params):
    a=np.array([params[str(g)]['alpha'] for g in cells%4]);b=np.array([params[str(g)]['beta'] for g in cells%4])
    return softmax(np.where(mask,a[:,None]*z+b[:,None]*q,-np.inf),axis=1)


def score(p,ids,target):
    chosen=np.where(p==p.max(1)[:,None],ids,1968).min(1)
    assert np.isfinite(p).all() and (p[np.arange(len(p)),target]>0).all()
    return np.stack([-np.log(p[np.arange(len(p)),target]),chosen==ids[np.arange(len(p)),target],p.max(1)],1)


def policies(d,calibration):
    plan=freeze();n=len(d['rows']);z=d['z'];mask=d['mask'];ids=d['ids'];target=d['target'];cells=d['cells'];root=d['root']
    scores={};nodes={};p0=softmax(np.where(mask,z,-np.inf),1)
    scores['legal']=score(p0,ids,target);nodes['legal']=np.zeros(n)
    raw=softmax(root[:,378:2346],1);y=np.array([r['target']-378 for r in d['rows']])
    scores['port_raw']=np.stack([-np.log(raw[np.arange(n),y]),raw.argmax(1)==y,raw.max(1)],1);nodes['port_raw']=np.zeros(n)
    direct=fitted(z,z*0,mask,cells,calibration['direct'])
    scores['calibrated_direct']=score(direct,ids,target);nodes['calibrated_direct']=np.zeros(n)
    for k,q in d['q'].items():
        old=plan['old_parameters']['lambda3' if k=='projected' else 'unchanged'] if k in ('1000','projected','64') else plan['budget_policies'][k]
        p=policy(d['rows'],root,q,ids,mask,d['seconds'],old)
        scores['frozen_'+k]=score(p,ids,target);nodes['frozen_'+k]=d['nodes'][k]
        p=fitted(z,q,mask,cells,calibration[k])
        scores['refit_'+k]=score(p,ids,target);nodes['refit_'+k]=d['nodes'][k]
    return scores,nodes


def calibrate(size):
    d=data(size,'aug');rows=d['rows'];cells=d['cells'];train=np.array([r['fold']==0 for r in rows]);params={}
    for k,q in {'direct':d['z']*0,**d['q']}.items():
        params[k]={}
        for g in range(4):
            take=train&(cells%4==g);count=np.bincount(cells[take],minlength=16);w=1/count[cells[take]]
            if k=='direct':
                result=minimize_scalar(lambda a:loss_gradient([a,0.],d['z'][take],q[take],d['mask'][take],d['target'][take],'forward',w)[0],bounds=(.4,1.6),method='bounded')
                p=dict(alpha=float(result.x),beta=0.,converged=bool(result.success))
            else:p=fit(d['z'][take],q[take],d['mask'][take],d['target'][take],'forward',w)
            assert p['converged'];params[k][str(g)]=p
    # Parameters are committed before confirmation losses are inspected.
    out=OUT/f'calibration-{size}.json'
    record=dict(parameters=params,source_sha256=digest(Path(__file__)),collection_sha256=digest(OUT/f'{size}-aug-fixed-predicted/plan.json'),
        specification='Per-Elo alpha/beta only, fit on2048 August fold0 positions (128/cell); direct has beta0. All arms reported on disjoint2048 fold1; no confirmation/golden selection. No algorithm/NN-weight updates.')
    if out.exists():
        original=json.loads(out.read_text())
        for k in ('parameters','collection_sha256','specification'):assert original[k]==record[k],k
        atomic(OUT/f'calibration-reproduction-{size}.json',dict(original_sha256=digest(out),current_source_sha256=record['source_sha256'],parameters_exactly_equal=True,
            note='Recomputed on the identical August fit fold to verify source refactoring did not alter frozen parameters; no data/parameter/selection change.'))
    else:atomic(out,record)
    scores,nodes=policies(d,params);report={}
    for name,s in scores.items():
        ce=cellmean(s[~train,0],cells[~train]);cost=cellmean(nodes[name][~train],cells[~train]);report[name]=dict(macro=float(ce.mean()),expert_macro=float(ce[3::4].mean()),mean_nodes=float(cost.mean()),training_eq_cm=None)
    atomic(OUT/f'development-{size}.json',dict(methods=report,calibration_sha256=digest(out),note='August calibration confirmation, potentially training-seen; golden CM does not apply.'))
    print('DEVELOPMENT',size,json.dumps(report),flush=True)


def golden(size):
    d=data(size,'gold');cal=json.loads((OUT/f'calibration-{size}.json').read_text())
    scores,nodes=policies(d,cal['parameters']);n=len(d['rows']);cells=d['cells'];games=np.array([r['game'] for r in d['rows']]);extra_times={}
    plan=freeze()
    ad=OUT/f'{size}-gold-adaptive-predicted'
    if (ad/f'{n-128:06d}.npz').exists():
        for key in ('adaptive_frozen','adaptive_refit'):scores[key]=[];nodes[key]=[]
        total_time=0.
        for lo in range(0,n,128):
            with np.load(ad/f'{lo:06d}.npz') as f:
                part=d['rows'][lo:lo+len(f['game'])];nn=len(part);ids=f['ids'];mask=f['mask'];root=f['root'].astype(float);q=f['q'][0]
                np.testing.assert_array_equal(f['game'],games[lo:lo+nn]);target=d['target'][lo:lo+nn];cc=cells[lo:lo+nn]
                z=np.where(mask,root[:,378:2346][np.arange(nn)[:,None],ids],0.)
                pp=np.zeros(mask.shape);new=np.zeros(mask.shape)
                for b in (128,256,512,1000):
                    take=(f['allocated']==b)|((f['allocated']==0)&(b==128))
                    if not take.any():continue
                    pp[take]=policy([part[i] for i in np.flatnonzero(take)],root[take],q[take],ids[take],mask[take],d['seconds'][lo:lo+nn][take],plan['budget_policies'][str(b)])
                    new[take]=fitted(z[take],q[take],mask[take],cc[take],cal['parameters'][str(b)])
                for key,p in [('adaptive_frozen',pp),('adaptive_refit',new)]:scores[key].append(score(p,ids,target));nodes[key].append(f['nodes'][0])
                total_time+=json.loads(str(f['stats']))['seconds']
        for key in ('adaptive_frozen','adaptive_refit'):scores[key]=np.concatenate(scores[key]);nodes[key]=np.concatenate(nodes[key])
        extra_times['adaptive']=total_time
    basefolder=OUT/f'{size}-gold-baselines';bp=json.loads((ROOT/'golden-balanced-v1/plan.json').read_text())['methods']
    for mode in ('shallow','released','repaired'):
        if not (basefolder/f'{mode}-{n-128:06d}.npz').exists():continue
        keys=['two_ply','four_ply'] if mode=='shallow' else ['allie_'+mode]
        for key in keys:scores[key]=[];nodes[key]=[]
        total_time=0.
        for lo in range(0,n,128):
            with np.load(basefolder/f'{mode}-{lo:06d}.npz') as f:
                nn=len(f['game']);np.testing.assert_array_equal(f['game'],games[lo:lo+nn]);part=d['rows'][lo:lo+nn]
                ids=d['ids'][lo:lo+nn];mask=d['mask'][lo:lo+nn];target=d['target'][lo:lo+nn];ar=np.arange(nn)
                z=f['root'][:,378:2346][ar[:,None],ids].astype(float)
                for key in keys:
                    if mode=='shallow':
                        setting=bp['calibrated_'+key];depth=setting['depth']-1
                        q=f['q'][depth][ar[:,None],ids];prob=softmax(np.where(mask,setting['alpha']*z+setting['beta']*np.nan_to_num(q),-np.inf),1);count=f['nodes'][depth]
                    else:prob=f['policy'][ar[:,None],ids];count=f['nodes']
                    scores[key].append(score(prob,ids,target));nodes[key].append(count)
                total_time+=json.loads(str(f['stats']))['seconds']
        for key in keys:scores[key]=np.concatenate(scores[key]);nodes[key]=np.concatenate(nodes[key])
        extra_times[mode]=total_time
    names=list(scores)
    canonical=json.loads((OUT/f'canonical-{size}/results.json').read_text());full=np.array(canonical['cells'])
    with np.load(OUT/f'canonical-{size}/sample.npz') as f:
        np.testing.assert_array_equal(f['game'],games);cz=f['logits'][:,378:2346].astype(float)
    y=np.array([r['target']-378 for r in d['rows']]);ref=logsumexp(cz,1)-cz[np.arange(n),y]
    losses=np.stack([scores[k][:,0] for k in names],1);delta=losses-ref[:,None]
    boot=bootstrap_deltas(delta,cells,games);draws=full[None,:,None]+boot
    points=full[:,None]+np.stack([cellmean(delta[:,i],cells) for i in range(len(names))],1)
    laws=json.loads((ROOT/'training-cm-laws.json').read_text());run,study=MODELS[size]
    training=json.loads((BASE/'results/recipe10x'/study/'results'/(run+'.json')).read_text());c0=float(training['budget']);c6=training['useful_training_flops']/6
    results={};ci=lambda x:np.quantile(x,[.025,.975]).tolist();legal=names.index('legal')
    for i,name in enumerate(names):
        r=dict(mean_nodes=float(cellmean(nodes[name],cells).mean()),expert_mean_nodes=float(cellmean(nodes[name],cells)[3::4].mean()),
            cells=points[:,i].tolist(),sample_macro_accuracy=float(cellmean(scores[name][:,1],cells).mean()),sample_expert_accuracy=float(cellmean(scores[name][:,1],cells)[3::4].mean()))
        for metric,ix in [('macro',np.arange(16)),('expert_macro',np.arange(3,16,4))]:
            law=laws['metrics'][metric]['law'];value=float(points[ix,i].mean());lb,ub=ci(draws[:,ix,i].mean(1))
            cm=lambda x:multiplier(law,c0,canonical[metric],float(x))
            r[metric]=value;r[metric+'_ci95']=[lb,ub];r[metric+'_training_eq_cm']=cm(value);r[metric+'_cm_ci95']=[cm(ub),cm(lb)]
            r[metric+'_cm_flops_div6_sensitivity']=multiplier(law,c6,canonical[metric],value)
            r[metric+'_local_dL_dlnC']=-law['alpha']*law['beta']/(law['alpha']+law['beta'])*excess(law,c0)
            for baseline in ('legal','calibrated_direct','frozen_1000'):
                j=names.index(baseline);r[metric+'_delta_vs_'+baseline]=float((points[ix,i]-points[ix,j]).mean())
                r[metric+'_delta_vs_'+baseline+'_ci95']=ci((draws[:,ix,i]-draws[:,ix,j]).mean(1))
                r[metric+'_cm_vs_'+baseline]=cm(value)/cm(points[ix,j].mean())
        results[name]=r
    record=dict(model=run,canonical=canonical,methods=results,training_useful_flops=training['useful_training_flops'],law_coordinate_rung=c0,alternative_coordinate_flops_div6=c6,
        law_shape_sha256=digest(ROOT/'training-cm-laws.json'),calibration_sha256=digest(OUT/f'calibration-{size}.json'),positions=n,games=len(set(games)),
        caveat='CM uses prior golden law shape vertically anchored at this checkpoint, C0=study rung label exactly, matching training ledgers. Useful training FLOPs/6 is separately reported sensitivity (the initial provisional convention). Shape transfer is unverified for the new recipe; CIs include game sampling only, not law uncertainty. Search-only CM is relative to legal policy; all-in CM separately includes port and legality. Golden sample reused, every frozen/refit method reported without selecting.',
        timing=dict(shared_fixed_grid_seconds=sum(t['seconds'] for t in d['timings']),prefill_tokens=sum(t['prefill_tokens'] for t in d['timings']),**extra_times))
    atomic(OUT/f'results-{size}.json',record)
    np.savez_compressed(OUT/f'scores-{size}.npz',names=names,scores=np.stack(list(scores.values())),nodes=np.stack(list(nodes.values())),canonical_nll=ref,cells=cells,games=games)
    print('GOLDEN',size,{k:{m:v[m] for m in ('macro','expert_macro','macro_training_eq_cm','expert_macro_training_eq_cm','mean_nodes')} for k,v in results.items()},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['calibrate','golden']);ap.add_argument('size',choices=['small','large']);a=ap.parse_args()
    (calibrate if a.action=='calibrate' else golden)(a.size)
