"""Stop the resident GPU after the finite live queue, then analyze once."""
import json,time
from .analysis import OUT

def main():
 requests=sorted((OUT/'queue').glob('*-live-*.request.json'));assert len(requests)==9,len(requests)
 for p in requests:
  result=p.with_name(p.name.replace('.request.','.result.'));error=p.with_name(p.name.replace('.request.','.error.'))
  for _ in range(360):
   if result.exists():break
   if error.exists():raise RuntimeError(error.read_text())
   time.sleep(10)
  else:raise TimeoutError(str(p))
 (OUT/'STOP').write_text('All frozen mixed-budget evaluations complete. GPU released; CPU report remaining.\n')
 from .score_live import main as score
 score()

if __name__=='__main__':main()
