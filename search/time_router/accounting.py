"""Use the existing cumulative accountant, but retain this study's identity."""
import fcntl,json,time
from search.advance import tick,OUT as ROOT
from search.engine.service import atomic
OUT=ROOT/'time-router-v1'

def main():
 with (ROOT/'accounting.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   _,done=tick();state=json.loads((ROOT/'status.json').read_text())
   state['phase']='Time routing / matched-cost Allie study: GPU ended' if done else 'Time routing / matched-cost Allie study active'
   state['active_followup']='time-router-v1';atomic(ROOT/'status.json',state)
   if done:break
   time.sleep(60)

if __name__=='__main__':main()
