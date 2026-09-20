"""Render the finite study; no parameter selection or training."""
import json
from .analysis import OUT,ROOT

LABELS={'coverage-elo':'Coverage · Elo','coverage-format':'Coverage · format × Elo','coverage-time':'Coverage · time-aware','allie-elo':'Repaired Allie · Elo','allie-format':'Repaired Allie · format × Elo','allie-time':'Repaired Allie · time-aware','allie-predicted-time':'Repaired Allie · predicted thinking time','coverage-time-within-cell':'Coverage · time-aware, cell costs fixed','allie-time-within-cell':'Repaired Allie · time-aware, cell costs fixed','legal':'Legal policy'}

def main():
 live=json.loads((OUT/'live-results.json').read_text());cached=json.loads((OUT/'cached-results.json').read_text());r=live['methods'];p=live['provenance'];state=json.loads((ROOT/'status.json').read_text())
 text=['# Time-aware budgets and repaired Allie','', '129M ship checkpoint, same L40S, nine frozen allocation arms. Lower CE is better. The table uses actual mixed-budget execution, with the same full-evaluation anchors and July positions for every arm.','', '| Method | Nodes | Macro CE | Expert CE | Macro CM | Expert CM |','|---|---:|---:|---:|---:|---:|']
 for name in ['legal',*[k for k in LABELS if k!='legal']]:
  if name not in r:continue
  a=r[name];text.append(f"| {LABELS[name]} | {a['mean_nodes']:.1f} | {a['macro']:.4f} | {a['expert_macro']:.4f} | {a['macro_cm_vs_legal']:.2f}× | {a['expert_macro_cm_vs_legal']:.2f}× |")
 text+=['','CM is search-only, relative to this checkpoint’s legal policy. It is a training-equivalent estimate conditional on the old scaling-law shape at rung 3e17, not a measured training gain or a serving speedup. Raw-to-legal improvements are excluded.','', '## Paired comparisons','', 'CE difference, new minus reference; negative helps. Intervals are 95% whole-game bootstrap intervals, not corrected for the full family of tested methods.','', '| Comparison | Macro Δ [95% CI] | Expert Δ [95% CI] |','|---|---:|---:|']
 pairs=[('coverage-format','coverage-elo'),('coverage-time','coverage-elo'),('allie-format','allie-elo'),('allie-time','allie-elo'),('allie-predicted-time','allie-elo'),('allie-elo','coverage-elo'),('allie-time','coverage-time'),('coverage-time-within-cell','coverage-elo'),('allie-time-within-cell','allie-elo')]
 for a,b in pairs:
  row=[]
  for m in ['macro','expert_macro']:
   delta=r[a][m+'_delta_vs_'+b];lo,hi=r[a][m+'_delta_vs_'+b+'_ci95'];row.append(f'{delta:+.4f} [{lo:+.4f}, {hi:+.4f}]')
  text.append(f'| {LABELS[a]} versus {LABELS[b]} | '+ ' | '.join(row)+' |')
 text+=['','## Where the compute goes','', 'Mean actual neural evaluations per position, equally weighted over Elo within each format.','', '| Method | Bullet | Blitz | Rapid | Classical |','|---|---:|---:|---:|---:|']
 for name in list(LABELS)[:7]:text.append('| '+LABELS[name]+' | '+' | '.join(f"{r[name]['by_format'][str(g)]['nodes']:.0f}" for g in range(4))+' |')
 text+=['','The two cell-cost diagnostics keep each format × Elo cell at its Elo-router reference cost. They test allocation within cells rather than the benefit of transferring nodes between cells. Detailed clock buckets and all 16 cell values are in `live-results.json`.','', '## Fixed-budget reference curves','', 'These use the cached common-prefix grids; the adaptive table above uses live reruns. Each fixed point has its own predeclared output calibration.','', '| Method | Nodes | Macro CE | Expert CE | Macro CM | Expert CM |','|---|---:|---:|---:|---:|---:|']
 for name,a in cached['methods'].items():
  if '-fixed' in name:text.append(f"| {name} | {a['mean_nodes']:.1f} | {a['macro']:.4f} | {a['expert_macro']:.4f} | {a['macro_cm_vs_legal']:.2f}× | {a['expert_macro_cm_vs_legal']:.2f}× |")
 text+=['','## Allie development sweep','', 'For each cpuct, the Elo-router ridge is chosen by fit-game CV. Confirmation is reported without selecting on it. These August CE values have no golden CM conversion.','', '| cpuct | Fit-game CV macro | Confirmation macro | Confirmation expert |','|---|---:|---:|---:|']
 for cp in [.5,1.25,2.5]:
  dev=json.loads((OUT/f'allie-{cp}-development.json').read_text());a=min((v for v in dev['candidates'].values() if v['parameters']['kind']=='elo'),key=lambda v:v['cv_macro']);cc=a['confirmation_cells'];text.append(f"| {cp} | {a['cv_macro']:.4f} | {sum(cc)/16:.4f} | {sum(cc[3::4])/4:.4f} |")
 text+=['','## Runtime and numerical check','', '| Live method | Evaluation seconds | Live−cached macro | Live−cached expert | Max policy change |','|---|---:|---:|---:|---:|']
 for name,a in p['cached_vs_live'].items():text.append(f"| {LABELS[name]} | {p['timing'][name]['seconds']:.1f} | {a['macro_delta']:+.5f} | {a['expert_delta']:+.5f} | {a['max_policy_diff']:.4f} |")
 jobs=[j for j in state['jobs'] if j['id'] in ['10506788',json.loads((OUT/'live-job.json').read_text())['id']]]
 text+=['','Times include root prefills and search on 8,192 positions; runtime staging, model startup, file writes and CPU fitting/bootstrap are additional. This study used one GPU at a time. Allocation usage: '+f"{sum(j.get('gpu_hours',0) for j in jobs):.3f} GPU-hours"+'; all earlier charges remain in the cumulative ledger.','', '## Method and limits','', '- Repaired Allie received cpuct 0.5/1.25/2.5 development sweeps, separate per-Elo reverse-KL calibration at every budget, and the same mean-node target as coverage. Its selected cpuct is held fixed across its routers. This is a bounded comparison, not proof that every Allie configuration is dominated.', '- The predicted-thinking-time arm keeps the repaired tree and calibrated output, but discretizes its proportional budget to 128/256/512/1000. It is not the exact released-paper budget/exploration schedule.', '- Time features use only information available before the move. Hypothetical child clocks use the model’s predicted thinking time.', '- July estimates are full canonical cell means plus paired differences on 512 positions per cell. The sample has been reused across research; this is not a fresh final confirmation. Final test games remain unopened.', '- Coverage’s output policy was previously frozen; Allie uses new development-fitted reverse-KL heads. The package comparison does not isolate backup alone.', '- CM intervals in the JSON cover game sampling, not scaling-law uncertainty. CE deltas are the primary evidence.', '', 'Files: `plan.json`, `selection-addendum.json`, `think-time-addendum.json`, `frozen.json`, `gold-allocations.json`, `cached-results.json`, `live-results.json`, and the source’s `METHOD.md`.','']
 (OUT/'REPORT.md').write_text('\n'.join(text))
 print(OUT/'REPORT.md')

if __name__=='__main__':main()
