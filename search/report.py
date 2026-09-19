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
    d=read('golden-v1/results.json') or read('golden-baseline/results.json')
    if d:
        table('Exact golden evaluation: 16-cell macro CE and four-cell expert macro CE; 1,553,058 scored moves.',
              [(k,x['macro'],x['expert_macro'],x['macro_training_eq_cm'],x['expert_macro_training_eq_cm']) for k,x in d['methods'].items()])
    text='# Inference research results\n\nTarget: both ≥2× macro and ≥10× expert training-equivalent CM. Not achieved.\n\n'
    text+='CM is computed only from matching golden metrics, using a frozen training-law shape anchored to the official raw checkpoint. Development-only rows stay pending. Search reports additionally separate gains beyond legal normalization.\n\n'
    text+='\n\n'.join(tables)+'\n\nOne preempt RTX6000Ada allocation; all reserved time, including idle time, is recorded in status.json. Serving implementation changes and the original MCTS audit are documented in search/PLAN.md and search/ALLIE_REVIEW.md.\n'
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
    tmp=ROOT/'REPORT.partial';tmp.write_text(text);tmp.replace(ROOT/'REPORT.md')
    print(text)


if __name__=='__main__':main()
