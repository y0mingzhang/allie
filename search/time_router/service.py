"""One resident GPU, finite study queue, immutable receipts."""
import fcntl,json,os,time,traceback
from .collect import OUT,GLOBAL_STOP,atomic,run
from search.transfer.oracle import ShipOracle
from search.engine.service import ROOT

def main():
 q=OUT/'queue';q.mkdir(exist_ok=True)
 with (q/'lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  oracle=ShipOracle(ROOT/'transfer-v1/ship-large-export',capacity=350000)
  atomic(q/'ready.json',dict(pid=os.getpid(),job=os.environ.get('SLURM_JOB_ID'),startup_seconds=oracle.startup_seconds))
  while not (OUT/'STOP').exists() and not GLOBAL_STOP.exists():
   for p in sorted(q.glob('*.request.json')):
    result=p.with_name(p.name.replace('.request.','.result.'));error=p.with_name(p.name.replace('.request.','.error.'))
    if result.exists() or error.exists():continue
    try:
     spec=json.loads(p.read_text())
     if spec.get('action')=='live':
      from .live import run as live_run
      atomic(result,live_run(oracle,spec))
     else:atomic(result,run(oracle,spec))
    except Exception:traceback.print_exc();atomic(error,dict(traceback=traceback.format_exc()))
    oracle.reset()
   time.sleep(2)

if __name__=='__main__':main()
