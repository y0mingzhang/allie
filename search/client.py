"""No-proxy authenticated client for the resident one-GPU oracle."""
import io
import json
import urllib.request
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'results/search-v1'


class Oracle:
    def __init__(self):
        self.ready = json.loads((OUT / 'server-ready.json').read_text())
        self.token = (OUT / 'rpc-token').read_text().strip()
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def __call__(self, prefixes, columns=None, *, all_tokens=False, compact=False):
        body = dict(prefixes=prefixes)
        if all_tokens:body['all_tokens']=True
        if compact:body['dtype']='float16'
        if columns is not None:
            body['columns'] = columns
        req = urllib.request.Request(self.ready['url'], data=json.dumps(body).encode(),
            headers={'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json'})
        with self.opener.open(req, timeout=120) as response:
            return np.load(io.BytesIO(response.read()), allow_pickle=False)

    def stats(self):
        req=urllib.request.Request(self.ready['url']+'/stats',headers={'Authorization':'Bearer '+self.token})
        with self.opener.open(req,timeout=30) as r:return json.loads(r.read())
