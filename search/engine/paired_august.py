"""Paired game bootstrap against a specified current control, across study caches."""
import json
from pathlib import Path
import sys
import numpy as np
from .balanced_eval import ROOT, atomic, digest
from .analyze_august import means


def main(study,control_study,reference):
    out=ROOT/study;control=ROOT/control_study
    assert out.resolve().parent==ROOT.resolve() and control.resolve().parent==ROOT.resolve()
    pa=json.loads((out/'plan.json').read_text());pb=json.loads((control/'plan.json').read_text())
    assert pa['sample_sha256']==pb['sample_sha256']
    with np.load(out/'scores.npz') as z:a={k:z[k] for k in z.files}
    with np.load(control/'scores.npz') as z:b={k:z[k] for k in z.files}
    for key in ('cells','games','fit'):np.testing.assert_array_equal(a[key],b[key])
    base=b['loss'][list(b['names']).index(reference)];fm=a['fit'];cells=a['cells'];games=a['games']
    _,ix=np.unique(games[~fm],return_inverse=True);g=ix.max()+1;count=np.zeros((g,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(98318).multinomial(g,np.full(g,1/g),size=2000).astype(float);den=w@count;assert (den>0).all()
    result={}
    for name,loss in zip(a['names'],a['loss']):
        delta=(loss-base)[~fm];point=means(delta,cells[~fm]);sums=np.zeros((g,16));np.add.at(sums,(ix,cells[~fm]),delta);draws=w@sums/den
        result[str(name)]=dict(macro_delta=float(point.mean()),expert_delta=float(point[3::4].mean()),
            macro_ci95=np.quantile(draws.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draws[:,3::4].mean(1),[.025,.975]).tolist(),
            cell_delta=point.tolist(),cell_ci95=np.quantile(draws,[.025,.975],axis=0).T.tolist())
    report=dict(reference_study=control_study,reference_method=reference,comparisons=result,
        study_scores_sha256=digest(out/'scores.npz'),control_scores_sha256=digest(control/'scores.npz'),
        source_sha256=digest(Path(__file__)),stage='Game-disjoint August confirmation only, potentially model-training-seen. Paired whole-game bootstrap, no training-CM conversion or correction for cumulative research reuse.')
    atomic(out/'paired-vs-reference.json',report)
    selected=json.loads((out/'results.json').read_text())['fit_cv_selected']
    for name in set(selected.values()):print(name,result[name],flush=True)


if __name__=='__main__':main(*sys.argv[1:])
