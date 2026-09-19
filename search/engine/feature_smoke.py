"""Capture parity and causality checks; root logits must stay unchanged."""
import json
from pathlib import Path
import time
import numpy as np
from .service import ROOT,atomic
from .balanced_eval import digest
from .features import last,full

def run(oracle,spec):
    out=ROOT/'feature-smoke-v1';out.mkdir(exist_ok=True)
    rows=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions']
    prefixes=[r['prefix'] for r in rows[:16]];start=time.monotonic()
    oracle.reset();reference=oracle(prefixes)
    oracle.reset();z,h=last(oracle,prefixes)
    np.testing.assert_array_equal(z,reference)
    oracle.reset();zz,hh=full(oracle,prefixes)
    np.testing.assert_array_equal(zz,reference)
    np.testing.assert_array_equal(hh[np.cumsum(list(map(len,prefixes)))-1],h)
    # Same shape with arbitrary different future tokens: earlier states must be
    # bit-exact, so bank targets cannot leak backward into stored hidden states.
    cuts=[max(11,len(p)//2) for p in prefixes]
    changed=[p[:cut]+list(reversed(p[cut:])) for p,cut in zip(prefixes,cuts)]
    oracle.reset();_,changed_h=full(oracle,changed);offset=0
    for p,cut in zip(prefixes,cuts):
        np.testing.assert_array_equal(hh[offset:offset+cut],changed_h[offset:offset+cut])
        offset+=len(p)
    result=dict(logit_identity=True,last_full_identity=True,future_mask_identity=True,
        positions=len(prefixes),seconds=time.monotonic()-start,
        sources={p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name('features.py'))})
    atomic(out/'results.json',result);print('Feature smoke',result,flush=True);return result
