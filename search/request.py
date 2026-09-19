"""Submit a persistent-oracle request. Inputs and outputs must be under own results."""
import argparse,json,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'
p=argparse.ArgumentParser();p.add_argument('op',choices=('ping','score'));p.add_argument('--input');p.add_argument('--output');p.add_argument('--timeout',type=float,default=600);a=p.parse_args()
queue=ROOT/'queue';queue.mkdir(exist_ok=True);key=uuid.uuid4().hex
r={'op':a.op}
if a.op=='score':
    assert a.input and a.output
    r|={'input':str(Path(a.input).resolve()),'output':str(Path(a.output).resolve())}
f=queue/(key+'.request.json');tmp=f.with_suffix('.partial');tmp.write_text(json.dumps(r));tmp.replace(f)
print('request',key,flush=True);start=time.monotonic()
while time.monotonic()-start<a.timeout:
    done=queue/(key+'.done.json');error=queue/(key+'.error.txt')
    if done.exists():print(done.read_text());break
    if error.exists():raise RuntimeError(error.read_text())
    time.sleep(.2)
else:raise TimeoutError('Request retained; inspect queue before retrying: '+key)
