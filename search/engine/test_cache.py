"""Validate side-state, forks and slot reuse independently of BF16 kernels."""
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from .model import ChessLM,DenseBackend
from .sglang_models.allie import RadixBackend


class Harness:
    def __init__(self,model):
        c=model.config;self.c=c;self.k={};self.v={}
        self.current_slots=torch.zeros(128,dtype=torch.long)
        self.previous_slots=torch.zeros(128,dtype=torch.long)
        self.smear_cache=torch.zeros(512,c.width,dtype=torch.float64)
        for i in model.long_layers:setattr(self,f'raw_keys_{i}',torch.zeros(512,c.width//c.head_dim,c.head_dim//2,dtype=torch.float64))
        self.layers=[SimpleNamespace(attn=self.attention(i)) for i in range(c.layers)]

    def attention(self,i):
        c=self.c;shape=(512,c.width//c.head_dim,c.head_dim)
        self.k[i]=torch.zeros(shape,dtype=torch.float64);self.v[i]=torch.zeros(shape,dtype=torch.float64)
        def call(q,k,v,batch):
            n=len(q);slots=self.current_slots[:n];self.k[i][slots]=k;self.v[i][slots]=v
            kk=self.k[i][batch.path];vv=self.v[i][batch.path]
            mask=torch.arange(len(batch.path))[None]<=batch.positions[:,None]
            out=F.scaled_dot_product_attention(q.view(n,-1,c.head_dim).transpose(0,1)[None],
                kk.transpose(0,1)[None],vv.transpose(0,1)[None],attn_mask=mask,scale=c.attn_scale)
            return out[0].transpose(0,1).reshape(n,-1)
        return call

    def run(self,model,ids,path,prefix):
        positions=torch.arange(prefix,len(path));n=len(positions)
        self.current_slots[:n]=torch.tensor(path[prefix:])
        self.previous_slots[:n]=torch.tensor([path[max(0,i-1)] for i in positions])
        batch=SimpleNamespace(path=path,positions=positions)
        return model(torch.tensor(ids[prefix:]),positions,RadixBackend(self,batch,positions))


def main():
    torch.set_num_threads(2);torch.manual_seed(123)
    config=dict(width=32,layers=8,head_dim=16,vocab_size=64,scalars_size=29,
        rotary_length=128,split_embed=True,ws_short=11,ws_long=23,clock=False,elo=False,
        norm_eps=1.1920928955078125e-7,attn_scale=.1)
    m=ChessLM(config).double().eval()
    with torch.no_grad():
        for p in m.parameters():p.normal_(std=.1)
        m.cos.fill_(1);m.sin.zero_()
        h=Harness(m);root=[2,3,8,19,23,18];ids=root+[4,5,9]
        # Chunked prefill, then a continuation; branch off an interior prefix.
        h.run(m,root[:3],[1,2,3],0)
        a=h.run(m,root,[1,2,3,4,5,6],3)
        b=m(torch.tensor(root),torch.arange(6),DenseBackend())[3:]
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        a=h.run(m,ids,list(range(1,10)),6)
        b=m(torch.tensor(ids),torch.arange(9),DenseBackend())[6:]
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        fork=root[:4]+[12,13,14];path=[1,2,3,4,20,21,22]
        a=h.run(m,fork,path,4)
        b=m(torch.tensor(fork),torch.arange(7),DenseBackend())[4:]
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        # Reuse slots for a completely different root; stale state cannot leak.
        fresh=[31,29,27,25];a=h.run(m,fresh,[1,2,3,4],0)
        b=m(torch.tensor(fresh),torch.arange(4),DenseBackend())
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    print('PASS: chunked prefill, continuation, prefix fork, slot reuse; FP64 tolerance 1e-12')


if __name__=='__main__':main()
