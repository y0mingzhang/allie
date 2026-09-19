"""Freeze a uniform per-cell sample of the EXISTING golden scored positions.

Selection uses labels/coordinates and a fixed RNG, never scores. Existing full-set
canonical baseline losses provide the per-position difference-estimator anchor.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from .native_board import from_prefix
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import golden_data

ROOT=Path(__file__).resolve().parents[2]/'results/search-v1'
OUT=ROOT/'golden-balanced-v1'


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'sample.json').exists(),'Sample is immutable; do not resample'
    # Freeze all method choices before selecting sample positions or reading their losses.
    two=json.loads((ROOT/'reply-pilot/calibrated-selection.json').read_text())
    four=json.loads((ROOT/'deeper-pilot/results.json').read_text())
    repair=json.loads((ROOT/'adaptive-repairs-pilot/results.json').read_text())
    # The expanded two-ply confirmation records the selected constants explicitly.
    two=json.loads((ROOT/'reply-confirmation/calibrated-results.json').read_text())['parameters']
    plan=dict(seed=1926731,per_cell=512,methods=dict(
        legal=dict(alpha=1.,beta=0.),
        calibrated_two_ply=dict(alpha=two['alpha'],beta=two['beta'],depth=2,widths=[4,2,2]),
        calibrated_four_ply=dict(zip(('alpha','beta'),four['results']['depth4']['coefficients'])) | dict(depth=4,widths=[4,2,2]),
        released_adaptive=dict(mean_sims=50),
        repaired_fixed=dict(n_sims=50,**repair['calibration']['reverse'],direction='reverse')),
        estimator='Exact full canonical-raw cell CE + uniform-sample paired (method CE - canonical-raw CE), then unweighted cell mean. Also report ordinary sample CE.',
        uncertainty='Whole-game paired bootstrap shared across cells/methods; scaling-law shape uncertainty reported separately.',
        decision='Report every preregistered method; no parameter or method selection using golden scores. No test split opened.',
        model=json.loads((ROOT/'serving-export/provenance.json').read_text()),
        canonical_baseline_plan=json.loads((ROOT/'golden-baseline/plan.json').read_text()),
        dev_sources={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'reply-confirmation/calibrated-results.json',ROOT/'deeper-pilot/results.json',ROOT/'adaptive-repairs-pilot/results.json')},
        law_sha256=digest(ROOT/'training-cm-laws.json'))
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:p.write_text(json.dumps(plan,indent=2)+'\n')
    data,labels,manifest=golden_data.load()
    assert manifest['sha256']==plan['canonical_baseline_plan']['strat_sha256']
    rng=np.random.default_rng(plan['seed']);selected=set()
    for cell in range(16):
        ri,ci=np.nonzero(labels[:,1:]==cell);ci+=1
        assert len(ri)==manifest['scored_moves'][cell]
        selected.update((int(ri[k]),int(ci[k])) for k in rng.choice(len(ri),plan['per_cell'],replace=False))
    rows=[];by_row={};baseline_block={}
    for ri,ci in selected:by_row.setdefault(ri,[]).append(ci)
    for di,doc in enumerate(golden_data.games(data,labels)):
        start=doc['start'];end=start+len(doc['prefix'])
        chosen=[col for col in by_row.get(doc['row'],[]) if start<=col<end]
        if not chosen:continue
        for col in sorted(chosen):
            j=col-start;prefix=doc['prefix'][:j];board=from_prefix(prefix);legal=board.legal();target=doc['prefix'][j]
            assert target in legal
            record=dict(row=doc['row'],column=col,game=doc['game'],ply=j-11,cell=int(labels[doc['row'],col]),
                        prefix=prefix,target=target,legal=legal,baseline_block=di//32)
            rows.append(record)
            baseline_block.setdefault(di//32,[]).append(len(rows)-1)
    assert len(rows)==16*plan['per_cell'] and len({(r['row'],r['column']) for r in rows})==len(rows)
    for block,indices in baseline_block.items():
        with np.load(ROOT/f'golden-baseline/{block:05d}.npz') as z:
            lookup={(str(g),int(p)):i for i,(g,p) in enumerate(zip(z['game'],z['ply']))}
            for i in indices:
                r=rows[i];j=lookup[(r['game'],r['ply'])];assert z['cell'][j]==r['cell']
                r['canonical_raw_nll']=float(z['nll'][j,0]);r['canonical_legal_nll']=float(z['nll'][j,1])
                r['canonical_raw_correct']=bool(z['correct'][j,0]);r['canonical_legal_correct']=bool(z['correct'][j,1])
    # Grouping by existing document improves cache locality; it is independent of scores.
    counts=np.bincount([r['cell'] for r in rows],minlength=16);assert (counts==plan['per_cell']).all()
    sample=dict(positions=rows,counts=counts.tolist(),strat_sha256=manifest['sha256'],plan_sha256=digest(p),
                source_sha256=digest(__file__),side_channels='Checkpoint has no clock/Elo/continuous-feature inputs; no sidecar is consumed. Strength headers are unchanged.')
    tmp=OUT/'sample.partial';tmp.write_text(json.dumps(sample)+'\n');tmp.replace(OUT/'sample.json')
    print(json.dumps(dict(positions=len(rows),games=len({r['game'] for r in rows}),counts=counts.tolist(),plan_sha256=digest(p),sample_sha256=digest(OUT/'sample.json')),indent=2))


if __name__=='__main__':main()
