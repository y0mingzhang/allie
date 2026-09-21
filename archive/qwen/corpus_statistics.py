"""Estimate move-weighted rating mixture without changing the original corpus."""
import json
from pathlib import Path
import numpy as np
from lm_data import Packed
from lm_train import rating_at_targets
root=Path(__file__).resolve().parents[1]
bins=[0,1000,1400,1800,2000,2200,2400,2600,3000,10000]
report={}
for split in ['train','val']:
 data=Packed('/data/group_data/dei-group/yimingz3/allie/lichess_tokens_v2',split)
 ids=np.random.default_rng(617).choice(int(data.ends[-1]),min(20000,int(data.ends[-1])),replace=False)
 counts=np.zeros(len(bins)-1,np.int64);expert_by_row=[];move_by_row=[]
 for lo in range(0,len(ids),256):
  rows=data.rows(ids[lo:lo+256]);y=rows[:,1:];mask=(y>=378)&(y<=2345);elo=rating_at_targets(rows)
  counts+=np.histogram(elo[mask],bins=bins)[0]
  expert_by_row.extend((mask&(elo>=2400)).sum(1).tolist());move_by_row.extend(mask.sum(1).tolist())
 expert=np.array(expert_by_row);moves=np.array(move_by_row);rate=expert.sum()/moves.sum()
 # Row-cluster ratio standard error; rows may contain multiple games.
 se=np.std(expert-rate*moves,ddof=1)/np.sqrt(len(ids))/moves.mean()
 report[split]=dict(sampled_rows=len(ids),total_rows=int(data.ends[-1]),move_count=int(counts.sum()),
  rating_bin_edges=bins,move_counts=counts.tolist(),expert2400_fraction=rate,
  expert2400_fraction_row_cluster_95ci=[rate-1.96*se,rate+1.96*se],
  note='Uniform sample of original packed rows; validation uses all rows; no dataset changes.')
report['val'].pop('expert2400_fraction_row_cluster_95ci')
report['val']['fraction_is_full_validation_census']=True
(root/'results/corpus-statistics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
