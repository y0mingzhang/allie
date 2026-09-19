"""Fit a cheap temperature control on existing development FIT games only."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[2]/'results/search-v1'
rows=json.loads((ROOT/'dev.json').read_text())['positions']
fit=np.array([r['fold']==0 for r in rows]);y=np.array([r['target']-378 for r in rows])[fit]
chunks=[]
for lo in range(0,len(rows),256):
 with np.load(ROOT/f'adaptive-repairs-pilot/released_fixed-{lo:06d}.npz') as z:chunks.append(z['root'][:,378:2346])
z=np.concatenate(chunks).astype(float);legal=np.zeros_like(z,bool)
for i,r in enumerate(rows):legal[i,np.array(r['legal'])-378]=True
x=z[fit];mask=legal[fit];ar=np.arange(len(y))
def f(alpha):
 t=np.where(mask,alpha*x,-np.inf);return float(np.mean(logsumexp(t,axis=1)-t[ar,y]))
opt=minimize_scalar(f,bounds=(.5,2.),method='bounded',options=dict(xatol=1e-10));assert opt.success
record=dict(stage='Added cheap control before any new golden method scoring; fit only on existing dev fold0',
            alpha=float(opt.x),fit_ce=float(opt.fun),fit_positions=int(fit.sum()),
            dev_sha256=hashlib.sha256((ROOT/'dev.json').read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
p=ROOT/'golden-balanced-v1/direct-control.json'
if p.exists():assert json.loads(p.read_text())==record
else:p.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
