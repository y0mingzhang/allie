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
    d=read('branch-pilot/results.json')
    if d:
        for k,label in [('exact_future_legality','Exact future-legality conditioning'),('reply_entropy','Two-ply plus response entropy')]:
            x=d['results'][k];x=x.get('metrics',x)['confirmation'];rows.append((label,x['ce'],x['expert_ce'],None,None))
    d=read('mcts-pilot/output-ablation.json')
    if d:
        for k,label in [('prior_to_search','Fixed52 tree + fixed reverse-KL coefficient'),('search_to_prior','Fixed52 tree + fixed forward-KL coefficient')]:
            x=d['results'][k]['results']['fixed_matched']['confirmation'];rows.append((label,x['ce'],x['expert_ce'],None,None))
    table('Small development check: 1,050 positions, 558 expert moves. Same pilot reused for research; not a fresh final holdout.',rows)
    d=read('golden-v1/results.json') or read('golden-baseline/results.json')
    if d:
        table('Exact golden evaluation: 16-cell macro CE and four-cell expert macro CE; 1,553,058 scored moves.',
              [(k,x['macro'],x['expert_macro'],x['macro_training_eq_cm'],x['expert_macro_training_eq_cm']) for k,x in d['methods'].items()])
    text='# Inference research results\n\nTarget: both ≥2× macro and ≥10× expert training-equivalent CM. Not achieved.\n\n'
    text+='CM is computed only from matching golden metrics, using a frozen training-law shape anchored to the official raw checkpoint. Development-only rows stay pending. Search reports additionally separate gains beyond legal normalization.\n\n'
    text+='\n\n'.join(tables)+'\n\nOne preempt RTX6000Ada allocation; all reserved time, including idle time, is recorded in status.json. Serving implementation changes and the original MCTS audit are documented in search/PLAN.md and search/ALLIE_REVIEW.md.\n'
    tmp=ROOT/'REPORT.partial';tmp.write_text(text);tmp.replace(ROOT/'REPORT.md')
    print(text)


if __name__=='__main__':main()
