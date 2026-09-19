"""Local token-only SGLang client. Returned scores share a constant per row."""
import json
import base64
import time
import urllib.request
import numpy as np


class SGLangOracle:
    def __init__(self,url='http://127.0.0.1:43552',binary=False):
        self.url=url
        self.binary=binary
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.last_metadata=[]

    def request(self,path,body=None):
        data=None if body is None else json.dumps(body).encode()
        req=urllib.request.Request(self.url+path,data=data,headers={'Content-Type':'application/json'})
        with self.opener.open(req,timeout=600) as r:
            text=r.read().decode()
            return json.loads(text) if 'json' in r.headers.get('Content-Type','') else text

    def flush(self):return self.request('/flush_cache',{})

    def __call__(self,prefixes,columns=None):
        columns=list(range(2432)) if columns is None else list(columns)
        body=dict(input_ids=prefixes,sampling_params=dict(temperature=0,max_new_tokens=1,ignore_eos=True))
        if not self.binary:body.update(return_logprob=True,logprob_start_len=[len(p) for p in prefixes],token_ids_logprob=columns)
        response=self.request('/generate',body)
        if isinstance(response,dict):response=[response]
        self.last_metadata=[r['meta_info'] for r in response]
        rows=[]
        for meta in self.last_metadata:
            if self.binary:
                raw=base64.b64decode(meta['allie_logits_fp16'][0])
                rows.append(np.frombuffer(raw,dtype=np.float16)[columns]);continue
            entries=meta['output_token_ids_logprobs'][0]
            assert [x[1] for x in entries]==columns
            rows.append([x[0] for x in entries])
        return np.asarray(rows,dtype=np.float32)


def main():
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]/'results/search-v1'
    rows=json.loads((root/'dev.json').read_text())['positions'][:32]
    oracle=SGLangOracle(binary='--binary' in sys.argv);prefixes=[r['prefix'] for r in rows]
    outputs={}
    for name in ('cold','repeat','fork'):
        if name=='cold':oracle.flush()
        seq=[p+[r['legal'][0]] for p,r in zip(prefixes,rows)] if name=='fork' else prefixes
        start=time.monotonic();z=oracle(seq)
        outputs[name]=dict(seconds=time.monotonic()-start,
            cached_tokens=sum(m['cached_tokens'] for m in oracle.last_metadata))
        np.save(root/f'sglang-{name}.npy',z)
        print(name,outputs[name],flush=True)
    oracle.flush();z=oracle([p+[r['legal'][0]] for p,r in zip(prefixes,rows)])
    fork=np.load(root/'sglang-fork.npy');delta=z-fork;delta-=delta.mean(1,keepdims=True)
    outputs['fork_fresh_vs_cached_max_centered_logprob_difference']=float(abs(delta).max())
    (root/'sglang-cache-check.json').write_text(json.dumps(outputs,indent=2)+'\n')
    print(outputs,flush=True)


if __name__=='__main__':main()
