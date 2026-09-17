"""Conservative GPU bound for Slurm jobs linked by AND afterok dependencies.

Ancestors and descendants cannot overlap. Unrecognized dependency syntax is
ignored (counted as independent), never used to lower the bound.
"""
from functools import lru_cache
import re

def parse_afterok(value):
 part=r'\d+(?:\((?:unfulfilled|fulfilled)\))?'
 if not re.fullmatch('afterok:'+part+'(?::'+part+')*',value):return []
 return [int(re.match(r'\d+',p).group()) for p in value[len('afterok:'):].split(':')]

def peak_bound(weights,dependencies):
 ancestors={}
 def visit(j,stack):
  if j in ancestors:return ancestors[j]
  if j in stack:raise ValueError('Cyclic dependency graph')
  result=set()
  for parent in dependencies.get(j,[]):
   result.add(parent)
   if parent in weights:result.update(visit(parent,stack|{j}))
  ancestors[j]=result;return result
 for j in weights:visit(j,set())
 ids=[j for j,w in weights.items() if w>0];w=[int(weights[j]) for j in ids]
 assert all(v>=0 for v in weights.values())
 # Large unfamiliar queues fall back to the old conservative sum.
 if len(ids)>40:return sum(w)
 conflict=[]
 for i,j in enumerate(ids):conflict.append(sum(1<<k for k,p in enumerate(ids) if p in ancestors[j] or j in ancestors[p]))
 @lru_cache(None)
 def best(mask):
  if not mask:return 0
  members=[i for i in range(len(ids)) if mask>>i&1]
  i=max(members,key=lambda k:(conflict[k]&mask).bit_count())
  if not conflict[i]&mask:return sum(w[k] for k in members)
  return max(best(mask&~(1<<i)),w[i]+best(mask&~((1<<i)|conflict[i])))
 return best((1<<len(ids))-1)

def qos_peak_bound(weights, dependencies, job_qos, per_user_caps):
 """Upper bound using live scheduler limits; unknown jobs remain uncapped.

 Sum of per-QoS upper bounds is conservative even with cross-QoS dependencies.
 Intersect with the original DAG bound to retain those dependencies as well.
 """
 groups={}
 for job,weight in weights.items():
  qos=job_qos.get(job)
  groups.setdefault(qos,{})[job]=weight
 bound=0
 for qos,subset in groups.items():
  demand=peak_bound(subset,dependencies)
  cap=per_user_caps.get(qos)
  bound+=min(demand,cap) if cap is not None else demand
 return min(peak_bound(weights,dependencies),bound)

def check_math():
 import random
 assert peak_bound({1:4,2:4,3:4},{2:[1],3:[1]})==8
 assert peak_bound({1:8,2:4,3:4,4:1},{2:[1],3:[1],4:[2,3]})==8
 assert peak_bound({1:4,2:0,3:8},{2:[1],3:[2]})==8
 assert parse_afterok('afterok:12(unfulfilled):13(fulfilled)')==[12,13]
 assert parse_afterok('afterok:12?afterok:13')==[]
 rng=random.Random(42)
 for n in range(1,9):
  for _ in range(12):
   w={j:rng.randrange(1,5) for j in range(n)};deps={j:[k for k in range(j) if rng.random()<.3] for j in range(n)}
   reach=[[False]*n for _ in range(n)]
   for j,parents in deps.items():
    for k in parents:reach[j][k]=True
   for k in range(n):
    for i in range(n):
     for j in range(n):reach[i][j]|=reach[i][k] and reach[k][j]
   brute=max(sum(w[j] for j in range(n) if mask>>j&1) for mask in range(1<<n) if all(not(reach[i][j] or reach[j][i]) for i in range(n) for j in range(i) if mask>>i&1 and mask>>j&1))
   assert peak_bound(w,deps)==brute
 return dict(passed=True,random_dags_checked=96,fork_join_and_zero_gpu_chains=True,unknown_syntax_conservative=True)
if __name__=='__main__':
 import json
 print(json.dumps(check_math()))
