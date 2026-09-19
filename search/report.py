"""Compact cumulative experiment table; never convert dev losses with golden laws."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'


def read(name):
    p=ROOT/name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    tables=[]
    def table(title,rows):
        if not rows:return
        lines=[title,'','| Idea | CE ↓ | Expert CE ↓ | Macro CM | Expert CM |','|---|---:|---:|---:|---:|']
        for name,ce,ex,cm,ecm in rows:
            fmt=lambda x:'Pending' if x is None else f'{x:.3f}×'
            lines.append(f'| {name} | {ce:.5f} | {ex:.5f} | {fmt(cm)} | {fmt(ecm)} |')
        tables.append('\n'.join(lines))
    for path,title in [('aug-search-v1/results.json','August balanced tuning: may be training-seen; confirmation is game-disjoint from parameter fitting, not from model training.'),
                       ('aug-search-v1/distributional-results.json','August same-node outcome-distribution experiments: no golden CM conversion.'),
                       ('aug-search-v1/unvisited-results.json','August unvisited-action fallback and visit shrinkage: no additional model nodes.'),
                       ('aug-compact-v1/asymmetric-results.json','August own/opponent soft-backup temperature scan: CPU-only analysis of the same cached trees.'),
                       ('aug-conditioning-v1/results.json','August counterfactual-strength guidance, direct and stacked with search: extra full-prefix queries charged separately.'),
                       ('aug-search-v1/budget-router.json','August budget allocation by known Elo group, selected on fit CV under a node cap.'),
                       ('aug-selection-v1/results.json','August first-play urgency and exploration-strength selection ablations.'),
                       ('aug-selection-wide-v1/results.json','August broader exploration and FPU combination.'),
                       ('aug-child-features-v1/results.json','August predicted opponent difficulty from candidate-child time and policy heads; extra query cost charged.'),
                       ('aug-deep-v1/results.json','August extension to4000 simulations, with own1000snapshot control; higher node cost, no dominance claim.'),
                       ('aug-coverage-deep-v1/results.json','August root-coverage extension to4000 simulations, with its own matched-batch1000 control.'),
                       ('aug-coverage-v1/results.json','August root coverage quotas, with ordinary PUCT below root.'),
                       ('aug-coverage-v1/behavior.json','August expected outcome under the value-tilted continuation policy; unchanged soft backup selected.'),
                       ('aug-coverage-v1/format-calibration.json','August format-by-rating calibration, partially pooled toward Elo-only coefficients; no confirmation gain.'),
                       ('aug-player-search-v1/results.json','August Bayesian search-strength mixture from past same-player choices, versus cheaper static-mixture controls.'),
                       ('aug-adaptive-deep-v1/results.json','August depth transfer of the count-dependent backup: matched trees at256/1000/4000 simulations.'),
                       ('aug-diff-backup-v2/results.json','August differentiated backup calibration: native gradients checked against independent torch autograd; fit-CV selects unchanged control.'),
                       ('aug-adaptive-temperature-v1/results.json','August count-dependent soft backup on identical cached trees; subtree count is an exploration proxy, not independent samples.'),
                       ('aug-moment-tail-v1/results.json','August critic-moment correction for unvisited actions; no confirmation win.'),
                       ('aug-history-residual-v1/results.json','August within-game residual correction from past moves only; frozen model, no parameter updates.'),
                       ('aug-clock-calibration-v1/results.json','August pre-move remaining-clock calibration, direct and search; added legal information, no future thinking time.'),
                       ('aug-outcome-consistency-v1/results.json','August root/child outcome-consistency projection, all model predictions; no true outcomes used.'),
                       ('aug-rollout-v1/results.json','August human-policy Monte Carlo leaf values at1/4/8/16ply, four trajectories per action.'),
                       ('aug-rollout-v1/stack.json','August rollout/search combinations and empirical noise shrinkage, additional rollout nodes charged.'),
                       ('aug-retrieval-residual-v2/results.json','August residual retrieval: empirical human neighbors minus their frozen-model expectations; same June memory.'),
                       ('aug-boardbook-v1/results.json','August exact-state June boardbook: same-cell frequencies, full support from neural prior; extra-data inference.'),
                       ('aug-retrieval-v1/results.json','August June-datastore retrieval, alone and stacked with search; additional corpus memory and query costs, not pure search.'),
                       ('aug-adaptive-root-v1/results.json','August updated-policy CE-curvature root allocation against a matched static control.'),
                       ('aug-influence-v1/results.json','August internal value-influence allocation, with the same root-coverage quota; compare to the root-coverage control.'),
                       ('aug-transpositions-v1/results.json','August legal transposition-history averaging: fixed original/variant-mean weights; full-prefix cost charged; no confirmation win.'),
                       ('aug-selection-v1/utilities.json','August nonlinear value utilities and policy-weighted standardization, same cached trees.'),
                       ('aug-selection-v1/discount.json','August recursively regularized critic backups; all depths use the same tree, with exact terminal values.'),
                       ('aug-selection-v1/innovation.json','August search-innovation and shallow/deep value combination; no confirmation win.'),
                       ('aug-selection-v1/state-calibration.json','August state-dependent search correction; fit-game CV selection, fixed cached trees and node cost.'),
                       ('aug-soft-selection-v1/results.json','August soft Bellman values used for branch selection as well as final output; matched simulation budget.'),
                       ('aug-search-v1/latent-budget.json','August mixtures of latent search budgets, with nested game CV. Existing fixed budget selected; no promotion.')]:
        d=read(path)
        if d:
            rows=[]
            for k,x in d['results'].items():
                score=x['confirmation'];selected=' [fit-CV selected]' if k in d.get('fit_cv_selected',{}).values() else ''
                rows.append((k+selected,score['macro_ce'],score['expert_ce'],None,None))
            table(title,rows)
    for path,title in [
        ('aug-expanded-search-v1/results.json','Expanded August: 16384 moves, same game folds; backup and search-strength mixtures at256/1000 simulations.'),
        ('aug-expanded-backup-v1/results.json','Expanded August differentiated backup: output-only two-scalar or joint ten-scalar fit; same1000 trees.'),
        ('aug-expanded-stack-v1/results.json','Expanded August stacked fitted backups and search-strength mixtures; parameters refitted inside game folds.'),
        ('aug-consideration-v1/results.json','Expanded August sampled consideration-set model; rejected on confirmation.'),
        ('aug-retrieval-large-v1/results.json','Expanded August larger June retrieval datastore; additional data memory, shared-player exclusion ablation, all16 cells reported.')]:
        d=read(path)
        if d:
            rows=[]
            for k,x in d['results'].items():
                score=x['confirmation'];selected=' [fit-CV selected]' if k in d.get('fit_cv_selected',{}).values() else ''
                rows.append((k+selected,score['macro_ce'],score['expert_ce'],None,None))
            table(title,rows)
    rows=[]
    d=read('confirmation/results.json')
    if d:
        for k,label in [('canonical_raw','Canonical direct policy'),('legal','Legal policy'),('shallow','Shallow value correction')]:
            x=d['methods'][k];rows.append((label,x['dev_ce'],x['expert_dev_ce'],None,None))
    d=read('mcts-confirmation/results.json')
    if d:
        for k,label in [('fixed52','Released fixed52 MCTS'),('adaptive50','Released adaptive50 MCTS')]:
            x=d['methods'][k];rows.append((label,x['ce'],x['expert_ce'],None,None))
    d=read('reply-confirmation/results.json')
    if d:
        for k,x in d['methods'].items():rows.append((k,x['ce'],x['expert_ce'],None,None))
    d=read('reply-confirmation/calibrated-results.json')
    if d:rows.append(('Calibrated two-ply',d['ce'],d['expert_ce'],None,None))
    d=read('reply-confirmation/utility-results.json')
    if d:rows.append(('State-scaled two-ply',d['ce'],d['expert_ce'],None,None))
    table('Expanded development check: 64,366 positions, 11,396 expert moves; these are not macro metrics.',rows)
    rows=[];d=read('pilot-results.json')
    if d:
        for k,label in [('raw','Canonical direct policy'),('legal','Legal policy')]:
            x=d['methods'][k];rows.append((label,x['dev_ce'],x['expert_dev_ce'],None,None))
    d=read('prompt-pilot/selected.json')
    if d:
        x=d['selected']['confirmation'];rows.append(('Strength-prompt sweep (selected unchanged policy)',x['ce'],x['expert_ce'],None,None))
    d=read('player-pilot/selected.json')
    if d:
        x=d['selected'];rows.append(('Past-move Bayesian current form',x['confirmation_ce'],x['confirmation_expert_ce'],None,None))
    d=read('reply-pilot/selected.json')
    if d:
        for k,label in [('best_one_ply','All-legal one-ply values'),('selected','Two-ply human-reply expectation')]:
            x=d[k]['fold1'];rows.append((label,x['ce'],x['expert_ce'],None,None))
    d=read('reply-pilot/calibrated-selection.json')
    if d:
        x=d['parameters']['all'];rows.append(('Calibrated two-ply',x['confirmation_ce'],x['confirmation_expert_ce'],None,None))
    d=read('reply-pilot/utility-selection.json')
    if d:
        for k,x in d['results'].items():
            if k=='linear':continue
            x=x['metrics']['confirmation'];rows.append((f'Two-ply utility: {k}',x['ce'],x['expert_ce'],None,None))
    d=read('branch-pilot/results.json')
    if d:
        for k,label in [('exact_future_legality','Exact future-legality conditioning'),('reply_entropy','Two-ply plus response entropy')]:
            x=d['results'][k];x=x.get('metrics',x)['confirmation'];rows.append((label,x['ce'],x['expert_ce'],None,None))
    d=read('mcts-pilot/output-ablation.json')
    if d:
        for k,label in [('prior_to_search','Fixed52 tree + fixed reverse-KL coefficient'),('search_to_prior','Fixed52 tree + fixed forward-KL coefficient')]:
            x=d['results'][k]['results']['fixed_matched']['confirmation'];rows.append((label,x['ce'],x['expert_ce'],None,None))
    d=read('deeper-pilot/results.json')
    if d:
        for k,x in d['results'].items():
            x=x['metrics']['confirmation'];rows.append((f'Continuation expectation: {k}',x['ce'],x['expert_ce'],None,None))
    d=read('reply-pilot/adaptive-selection.json')
    if d:
        for k,x in d['results'].items():
            if k=='two_ply':continue
            x=x['metrics']['confirmation'];rows.append((f'Adaptive calibration: {k}',x['ce'],x['expert_ce'],None,None))
    table('Small development check: 1,050 positions, 558 expert moves. Same pilot reused for research; not a fresh final holdout.',rows)
    d=read('adaptive-repairs-pilot/results.json')
    if d:
        keep=('legal','released_fixed','fixed_repairs','released_time','released_time_repairs',
              'fixed_repairs_reverse','released_time_repairs_reverse','decoupled_time_reverse',
              'time_repairs_reverse','shuffled_time_reverse','entropy_reverse')
        table('Adaptive-MCTS repair pilot: same 1,050-position development confirmation; calibration fit on fixed-tree fold0, then shared across allocations.',
              [(k,d['results'][k]['metrics']['confirmation']['ce'],d['results'][k]['metrics']['confirmation']['expert_ce'],None,None) for k in keep])
    d=read('fast-deeper-pilot/results.json')
    if d:
        table('Cached engine verification on the same small development check; coefficients frozen before the port.',
              [(name,x['ce'],x['expert_ce'],None,None) for name,x in d['methods'].items()])
    d=read('mcts1000-pilot/results.json')
    if d:
        table('MCTS budget extension: 1,000 simulations per position; fit on fold0, report on the same small dev fold1. Higher node cost than the original 50-simulation baseline.',
              [(k,x['metrics']['confirmation']['ce'],x['metrics']['confirmation']['expert_ce'],None,None) for k,x in d['results'].items()])
    d=read('depth6-pilot/results.json')
    if d:
        table('Six-ply continuation extension: same development folds, every horizon and the predeclared deep/shallow-disagreement correction reported.',
              [(k,x['metrics']['confirmation']['ce'],x['metrics']['confirmation']['expert_ce'],None,None) for k,x in d['results'].items()])
    for priority in ('mass','policy','variance'):
        d=read(f'expectation-{priority}-pilot/results.json')
        if d:
            table(f'Adaptive continuation expectation ({priority} priority, 1000-node cap): development folds only.',
                  [(k,x['metrics']['confirmation']['ce'],x['metrics']['confirmation']['expert_ce'],None,None) for k,x in d['results'].items()])
    d=read('mcts1000-postprocess/results.json')
    if d:
        table('CPU-only MCTS output ablations: same cached 1000-simulation trees; family selection uses game CV within fit fold, all development confirmation arms reported.',
              [(k,x['metrics']['confirmation']['ce'],x['metrics']['confirmation']['expert_ce'],None,None) for k,x in d['results'].items()])
    d=read('mcts4000-pilot/results.json')
    if d:
        table('MCTS budget extension to 4000 simulations: existing development folds only.',
              [(k,x['metrics']['confirmation']['ce'],x['metrics']['confirmation']['expert_ce'],None,None) for k,x in d['results'].items()])
    d=read('residual-policy-dev/results.json')
    if d:
        for budget,run in d['results'].items():
            table(f'Cached {budget}-simulation value corrections and expectation combinations: development only; combined-tree cost is additional.',
                  [(k,x['confirmation']['ce'],x['confirmation']['expert_ce'],None,None) for k,x in run['results'].items()])
    d=read('allie-allocation-pilot/results.json')
    if d:
        table('Allie allocation retry: exactly 1000 simulations/position on average; fixed controls reused, calibration shared across allocation methods.',
              [(k,x['metrics']['confirmation']['ce'],x['metrics']['confirmation']['expert_ce'],None,None) for k,x in d['results'].items()])
    d=read('backup-ladder-dev/results.json')
    if d:
        table('Same-tree backup comparison at 16/64/256/1000 simulations: fixed output calibration, identical node cost within each budget; blitz development only.',
              [(k,x['metrics']['confirmation']['ce'],x['metrics']['confirmation']['expert_ce'],None,None) for k,x in d['results'].items()])
    for study in ('mcts1000-pilot', 'allie-allocation-pilot'):
        d=read(study+'/regularization-results.json')
        if d:
            table(f'Grill reverse-KL regularization formula versus its dev-calibrated common scale ({study}); development only.',
                  [(k,x['metrics']['confirmation']['ce'],x['metrics']['confirmation']['expert_ce'],None,None) for k,x in d['results'].items()])
    d=read('golden-v1/results.json') or read('golden-baseline/results.json')
    if d:
        table('Exact golden evaluation: 16-cell macro CE and four-cell expert macro CE; 1,553,058 scored moves.',
              [(k,x['macro'],x['expert_macro'],x['macro_training_eq_cm'],x['expert_macro_training_eq_cm']) for k,x in d['methods'].items()])
    balanced=read('golden-balanced-v1/results.json')
    if balanced:
        table('Preregistered balanced golden confirmation: 512 positions per cell, 8,192 total. Full-set canonical means plus paired sample differences; every method reported.',
              [(k,x['macro'],x['expert_macro'],x['macro_training_eq_cm'],x['expert_macro_training_eq_cm']) for k,x in balanced['methods'].items()])
        lines=['Paired whole-game bootstrap, 95% intervals. CM uncertainty is conditional on the frozen training-law shape.',
               '', '| Method | Macro CE interval | Expert CE interval | Macro CM interval | Expert CM interval |',
               '|---|---:|---:|---:|---:|']
        for k,x in balanced['methods'].items():
            interval=lambda key:'–'.join(f'{v:.4f}' for v in x[key])
            lines.append(f'| {k} | {interval("macro_ci95")} | {interval("expert_macro_ci95")} | {interval("macro_cm_ci95")} | {interval("expert_macro_cm_ci95")} |')
        tables.append('\n'.join(lines))
    d=read('golden-mcts1000-v1/results.json')
    if d:
        table('Frozen 1000-simulation MCTS confirmation: existing balanced golden sample, 512 positions/cell. All dev-frozen arms reported; training-law-conditional CM.',
              [(k,x['macro'],x['expert_macro'],x['macro_training_eq_cm'],x['expert_macro_training_eq_cm']) for k,x in d['methods'].items()])
    d=read('golden-augcal-v1/results.json')
    if d:
        table('August-fit-CV-selected golden budget frontier: unchanged July sample, all frozen arms reported. Conditional training-equivalent CM.',
              [(k,x['macro'],x['expert_macro'],x['macro_training_eq_cm'],x['expert_macro_training_eq_cm']) for k,x in d['methods'].items()])
        lines=['| Method | Avg nodes | Expert avg nodes | Macro CE delta vs four-ply (95% CI) | Expert CE delta vs four-ply (95% CI) |',
               '|---|---:|---:|---:|---:|']
        for k,x in d['methods'].items():
            fmt=lambda key:' to '.join(f'{v:+.4f}' for v in x[key])
            ec='unavailable' if x['expert_mean_nodes'] is None else f'{x["expert_mean_nodes"]:.1f}'
            lines.append(f'| {k} | {x["mean_nodes"]:.1f} | {ec} | {fmt("macro_delta_vs_previous_four_ply_ci95")} | {fmt("expert_macro_delta_vs_previous_four_ply_ci95")} |')
        tables.append('\n'.join(lines))
    for path,title in [('golden-router-v1/results.json','Frozen budget allocation on cached golden trees; dynamic execution is reported separately.'),
                       ('golden-dynamic-router-v1/results.json','Actual mixed-budget golden inference; same frozen policy, no retuning.'),
                       ('golden-explore-v1/results.json','August-selected broader-exploration golden check. Routed variant uses cached stopping; actual mixed-budget execution is separate.'),
                       ('golden-coverage-v1/results.json','August-selected root-coverage golden check. Routed variant is cached stopping, not yet an actual mixed-budget runtime measurement.'),
                       ('golden-dynamic-coverage-v1/results.json','Actual mixed-budget root-coverage search, with cached versus live numerical differences reported separately.'),
                       ('golden-fast-coverage-v1/results.json','Faster root-coverage search: actual4000 simulations and logical1000 prefix, frozen August coefficients; higher cost alone is not dominance.'),
                       ('golden-player-search-v1/results.json','Frozen static and past-choice-adapted search-strength mixtures; existing root trees plus new past-only queries. Total node cost includes both; reused golden.'),
                       ('golden-adaptive-temperature-v1/results.json','August-selected subtree-dependent soft backup, identical live1000 trees and nodes as its constant control; reused golden sample.'),
                       ('golden-utilities-v1/results.json','August-selected utility normalization on identical golden trees; zero additional model queries.'),
                       ('golden-permutation-v1/results.json','Numerical sensitivity audit: same frozen router, different fixed position order. Neither order is selected by quality.')]:
        d=read(path)
        if d:
            table(title,[(k,x['macro'],x['expert_macro'],x['macro_training_eq_cm'],x['expert_macro_training_eq_cm']) for k,x in d['methods'].items()])
    d=read('golden-expanded-stack-v2/results.json')
    if d:
        table('Live golden check of expanded-August CV-selected search stacks. Same1000 trees and nodes; reused golden, conditional training-equivalent CM.',
              [(k,d['methods'][k]['macro'],d['methods'][k]['expert_macro'],d['methods'][k]['macro_training_eq_cm'],d['methods'][k]['expert_macro_training_eq_cm'])
               for k in ('constant_single','subtree_sigma10','joint_sigma05','old_prior_s05','old4000')])
    audit=read('golden-permutation-v1/results.json')
    if audit:
        tables.append('Fixed position-permutation audit of the frozen live router: macro CE shift '+f"{audit['order_delta_ce']['macro']:+.6f}"+', expert shift '+f"{audit['order_delta_ce']['expert_macro']:+.6f}"+'. This measured order effect is separate from paired sampling intervals; one permutation does not estimate its full variance. Neither order is selected by quality.')
    text='# Inference research results\n\nTarget: both ≥10× macro and ≥10× expert training-equivalent CM, with an improved average-search-nodes versus quality frontier. Not achieved.\n\n'
    text+='CM is computed only from matching golden metrics, using a frozen training-law shape anchored to the official raw checkpoint. Development-only rows stay pending. Search reports additionally separate gains beyond legal normalization.\n\n'
    text+='\n\n'.join(tables)+'\n\nOne persistent GPU allocation at a time; previous preempt RTX6000Ada usage and its general L40S replacement remain charged in status.json, including idle time. Serving implementation changes and the original MCTS audit are documented in search/PLAN.md and search/ALLIE_REVIEW.md.\n'
    if balanced:
        t=balanced['timing']
        text+=f'\nBalanced confirmation timing: four-ply traversal {t["summed_four_ply_block_seconds"]:.1f} s (also supplies two-ply and cheap controls); both MCTS variants together {t["summed_two_MCTS_block_seconds"]:.1f} s. These are warm scoring times; cold startup is recorded separately. Full per-cell deltas, paired comparisons against calibration, accuracy, calibration and alternative-law sensitivity are in golden-balanced-v1/results.json.\n'
    d=read('engine-queue/002-mcts.result.json')
    if d:
        text+='\nWarm MCTS cost on 128 development roots (includes root inference and tree work):\n\n| Algorithm | Previous stack | Cached native-board stack | Speedup |\n|---|---:|---:|---:|\n'
        for adaptive,name in [(False,'Fixed52'),(True,'Adaptive50')]:
            pick=lambda engine:next(x['end_to_end_seconds'] for x in d['runs'] if x['roots']==128 and x['engine']==engine and x['adaptive']==adaptive)
            a,b=pick('original'),pick('cached_native');text+=f'| {name} | {a:.3f} s | {b:.3f} s | {a/b:.2f}× |\n'
        text+='\nBF16 kernel differences can change a few branches and predicted time budgets; the algorithm is unchanged. See search/engine/README.md for numerical drift and correctness tests.\n'
    d=read('engine-queue/009-nvme-final.result.json')
    if d:
        text+='\nEqual-node throughput after native tree and batched output-solver work (two repeats, mean wall time):\n\n| Method | Roots batched | Evaluated leaves | Seconds including root prefill |\n|---|---:|---:|---:|\n'
        for kind,n in [('four_ply',64),('native_mcts',128),('native_mcts',512),('native_mcts',1024)]:
            r=[x for x in d['benchmark']['runs'] if x['method']==kind and x['roots']==n]
            seconds=sum(x.get('end_to_end_seconds',x['seconds']) for x in r)/len(r)
            text+=f'| {kind} | {n} | {r[0]["nodes"]:,} | {seconds:.3f} |\n'
        text+='\nThese throughput runs use different numbers of roots, so they are not a quality comparison. These final measurements include native tree destruction and the complete call. The local runtime started in 30.7 s including Python imports; it stays resident between experiments.\n'
    d=read('handle-benchmark-v1/results.json')
    if d:
        text+='\nNode-handle inference bridge timing (same root-action forest, two repeats):\n\n| Positions × simulations | Full-history bridge | Node-handle bridge | Speedup |\n|---|---:|---:|---:|\n'
        for n,b in [(512,1000),(160,4000)]:
            avg=lambda kind:sum(d['measurements'][f'{n}-{b}-{kind}-{rep}']['seconds'] for rep in range(2))/2
            a,c=avg('prefix'),avg('handle');text+=f'| {n} × {b} | {a:.2f} s | {c:.2f} s | {a/c:.2f}× |\n'
        text+='\nThis is an implementation benchmark, not a new golden CE/CM result. Exact-oracle tests preserve search; measured GPU batching differences are retained in handle-benchmark-v1/results.json.\n'
    d=read('thread-benchmark-v1/results.json')
    if d:
        text+='\nParallel leaf expansion passes bit-exact live-model output checks at 1/2/4 threads. The job reserves eight CPUs; only leaf expansion is parallel, and value backups retain their original serial order. Two threads give nearly the same wall time as four and leave more CPU headroom.\n'
    tmp=ROOT/'REPORT.partial';tmp.write_text(text);tmp.replace(ROOT/'REPORT.md')
    print(text)


if __name__=='__main__':main()
