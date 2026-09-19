"""Equal expected-node comparison with an independently randomized base budget."""
import json
import time
from pathlib import Path
import numpy as np
from .service import ROOT, atomic
from .balanced_eval import digest
from .analyze_august import means


def run(oracle, spec):
    start=time.monotonic(); out=ROOT/'aug-tactical-budget-v1'
    inp=out/'results.json'; scores=out/'scores.npz'
    data=json.loads(inp.read_text()); records=data['results']
    plan=dict(source_sha256=digest(Path(__file__)),results_sha256=digest(inp),scores_sha256=digest(scores),
        control='Independently of board and labels, choose one of the two bracketing ordinary-search budgets with a fixed Bernoulli probability. Choose that probability from mean NN cost alone, matching the tactical arm mean cost. Expected CE is the convex combination of the losses, not CE of a policy ensemble. No both-budget execution and no extra model evaluation is assumed.',
        uncertainty='Whole-game paired bootstrap of the expected loss difference. Random budget-choice noise is analytically integrated out. This is a cached-prefix development comparison; live batching/timing remains necessary for a deployment claim.')
    pp=out/'cost-comparison-plan.json'
    if pp.exists(): assert json.loads(pp.read_text())==plan
    else: atomic(pp,plan)
    with np.load(scores) as f:
        names=f['names'].tolist(); losses=dict(zip(names,f['loss']))
        cells,games,fm=f['cells'],f['games'],f['fit']
    bases=sorted([n for n in records if n.endswith('_base')],key=lambda n:records[n]['mean_nodes'])
    _,ix=np.unique(games[~fm],return_inverse=True); ng=ix.max()+1
    count=np.zeros((ng,16)); np.add.at(count,(ix,cells[~fm]),1)
    draws=np.random.default_rng(91214).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float)
    den=draws@count; assert (den>0).all()
    result={}
    for name,rec in records.items():
        if '_tactical_' not in name: continue
        cost=rec['mean_nodes']
        lo=max([b for b in bases if records[b]['mean_nodes']<=cost],key=lambda b:records[b]['mean_nodes'])
        hi=min([b for b in bases if records[b]['mean_nodes']>=cost],key=lambda b:records[b]['mean_nodes'])
        prob=(cost-records[lo]['mean_nodes'])/(records[hi]['mean_nodes']-records[lo]['mean_nodes'])
        assert 0<=prob<=1
        mixed=(1-prob)*losses[lo]+prob*losses[hi]
        delta=losses[name]-mixed; point=means(delta[~fm],cells[~fm])
        ce=means(mixed[~fm],cells[~fm]); sums=np.zeros((ng,16))
        np.add.at(sums,(ix,cells[~fm]),delta[~fm]); boot=draws@sums/den
        result[name]=dict(mean_nodes=cost,control_low=lo,control_high=hi,probability_high=float(prob),
            control_macro_ce=float(ce.mean()),control_expert_ce=float(ce[3::4].mean()),
            delta_macro=float(point.mean()),delta_expert=float(point[3::4].mean()),
            macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),
            expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist(),
            expert_mean_nodes_tactical=rec['expert_mean_nodes'],
            expert_mean_nodes_control=(1-prob)*records[lo]['expert_mean_nodes']+prob*records[hi]['expert_mean_nodes'])
    atomic(out/'cost-comparison.json',dict(results=result,plan_sha256=digest(pp),seconds=time.monotonic()-start))
    print(json.dumps(result,indent=2),flush=True)
    return dict(study=out.name,comparisons=len(result))
