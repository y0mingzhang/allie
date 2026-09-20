"""Finite two-stage driver: frozen grids, stop GPU, then CPU report."""
import json,time,sys
from .analysis import OUT,frozen_selection
from search.engine.service import atomic

def wait(p):
 for _ in range(720):
  if p.exists():return
  errors=list((OUT/'queue').glob('*.error.json'))
  if errors:raise RuntimeError(errors[0].read_text())
  time.sleep(10)
 raise TimeoutError(p)

def main():
 wait(OUT/'frozen.json');frozen=frozen_selection();tag=frozen['methods']['allie-elo']['tag'];cp=float(tag.split('-')[1]);q=OUT/'queue';requests=[]
 for i,kind,c in [(5,'coverage',2.5),(6,'allie',cp)]:
  p=q/f'{i:03d}-gold-{kind}-{c}.request.json';spec=dict(model='large',split='gold',kind=kind,cpuct=c,batch=256)
  if p.exists():assert json.loads(p.read_text())==spec
  else:atomic(p,spec)
  requests.append(p.with_name(p.name.replace('.request.','.result.')))
 for p in requests:wait(p)
 # No more GPU work until the actual routed phase; release this allocation.
 if '--live' not in sys.argv:(OUT/'STOP').write_text('Grid collection complete. Safe restart for live phase after this job exits.\n')
 from .evaluate import main as evaluate
 evaluate()
 if '--live' in sys.argv:
  allocations=json.loads((OUT/'gold-allocations.json').read_text())
  for i,name in enumerate(allocations['choice'],7):
   p=q/f'{i:03d}-live-{name}.request.json';spec=dict(action='live',method=name)
   if p.exists():assert json.loads(p.read_text())==spec
   else:atomic(p,spec)
  from .finish_live import main as finish
  finish()

if __name__=='__main__':main()
