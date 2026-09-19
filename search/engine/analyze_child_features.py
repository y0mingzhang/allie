"""Does only the value information added by deeper search improve the prior?"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .balanced_eval import ROOT,atomic,digest
from .analyze_august import means


from .innovation import fit


def main():
    start=time.monotonic();rows=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions'];n=len(rows);ar=np.arange(n)
    cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows]);fm=np.array([r['fold']==0 for r in rows])
    cv=np.array([int(hashlib.sha256(('cv:'+g).encode()).hexdigest(),16)%3 for g in games])
    k=max(len(r['legal']) for r in rows);mask=np.zeros((n,k),bool);ids=np.zeros((n,k),int);target=np.zeros(n,int)
    for i,r in enumerate(rows):mask[i,:len(r['legal'])]=True;ids[i,:len(r['legal'])]=np.array(r['legal'])-378;target[i]=r['legal'].index(r['target'])
    root=np.zeros((n,2432));q=np.zeros((n,k));f=np.zeros((n,k,6));extra=np.zeros(n);search_nodes=np.zeros(n)
    for lo in range(0,n,512):
        with np.load(ROOT/f'aug-selection-v1/zero_cp25/{lo:06d}.npz') as z:
            hi=lo+len(z['game']);assert list(z['game'])==list(games[lo:hi]);kk=z['q'].shape[-1]
            root[lo:hi]=z['z'];q[lo:hi,:kk]=z['q'][-1];search_nodes[lo:hi]=z['evaluated_nodes'][-1]
        with np.load(ROOT/f'aug-child-features-v1/{lo:06d}.npz') as z:
            assert list(z['game'])==list(games[lo:hi]);kk=z['features'].shape[1];f[lo:hi,:kk]=z['features'];extra[lo:hi]=z['nodes']
    logits=np.where(mask,root[:,378:2346][ar[:,None],ids],0.)
    # Existing root logits and deep Q are unchanged; adding these predictions
    # is separately charged. No credit for their root batch's numerical drift.
    menus=[('baseline',[q],0.),('opponent_time',[q,f[:,:,0]],.001),
        ('reply_entropy',[q,f[:,:,2]],.001),('difficulty',[q,f[:,:,0],f[:,:,2]],.001),
        ('difficulty_time_entropy',[q,f[:,:,0],f[:,:,2],f[:,:,1]],.001),
        ('cheap_oneply',[f[:,:,4]],0.),('cheap_difficulty',[f[:,:,4],f[:,:,0],f[:,:,2]],.001)]
    records={};losses={}
    for name,values,ridge in menus:
        features=np.stack([logits,*values],1);p,params=fit(features,mask,target,cells,fm,ridge);loss=-p[ar,target];oof=np.full(n,np.nan);converged=[]
        for fold in range(3):
            val=fm&(cv==fold);v,info=fit(features,mask,target,cells,fm&(cv!=fold),ridge);oof[val]=-v[ar[val],target[val]];converged.append(info['converged'])
        a=means(loss[~fm],cells[~fm]);b=means(oof[fm],cells[fm]);losses[name]=loss
        cost=search_nodes if name=='baseline' else extra if name.startswith('cheap') else search_nodes+extra
        records[name]=dict(parameters=params,cv_converged=converged,training_equivalent_cm=None,
            mean_nodes=float(means(cost,cells).mean()),expert_mean_nodes=float(means(cost,cells)[3::4].mean()),
            confirmation=dict(macro_ce=float(a.mean()),expert_ce=float(a[3::4].mean()),cells=a.tolist()),
            fit_game_cv=dict(macro_ce=float(b.mean()),expert_ce=float(b[3::4].mean())))
        print(name,records[name]['fit_game_cv'],records[name]['confirmation']['macro_ce'],records[name]['confirmation']['expert_ce'],flush=True)
    prior=json.loads((ROOT/'aug-selection-v1/results.json').read_text())['results']['zero_cp25_1000_elo']['confirmation']
    for key in ('macro_ce','expert_ce'):np.testing.assert_allclose(records['baseline']['confirmation'][key],prior[key],atol=2e-6,rtol=0)
    selected={metric:min(records,key=lambda k:records[k]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);g=ix.max()+1;count=np.zeros((g,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(8317).multinomial(g,np.full(g,1/g),size=2000).astype(float);den=w@count
    for name in records:
        sums=np.zeros((g,16));np.add.at(sums,(ix,cells[~fm]),(losses[name]-losses['baseline'])[~fm]);draws=w@sums/den
        records[name]['confirmation_delta_vs_baseline_ci95']=dict(macro=np.quantile(draws.mean(1),[.025,.975]).tolist(),expert=np.quantile(draws[:,3::4].mean(1),[.025,.975]).tolist())
    atomic(ROOT/'aug-child-features-v1/results.json',dict(results=records,fit_cv_selected=selected,analysis_seconds=time.monotonic()-start,source_sha256=digest(Path(__file__)),
        stage='August potentially training-seen, fit-game CV selection. Extra all-legal-child queries explicitly charged; no future observed time/reply/outcome. cp2.5 root/deep caches unchanged. All seven arms reported; CM pending golden.'))
    print('SELECTED',selected,flush=True)


if __name__=='__main__':main()
