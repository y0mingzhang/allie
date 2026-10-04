"""Inference-only port of the dense transformer (board CNN, SwiGLU, three clock features), used for FLOP
counts and as a CPU reference backend. Attention is a backend callback. Derived from the project's MIT-licensed
model.nanogpt implementation.
"""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from allie.model.board import BoardConv


def linear(x,w):return F.linear(x,w.to(x.dtype))


class AttentionWeights(nn.Module):
    def __init__(self,width,heads,value_gate=True):
        super().__init__()
        self.qkvo_w=nn.Parameter(torch.empty(4*width,width,dtype=torch.float32))
        self.attn_gate=nn.Linear(16,heads,bias=False,dtype=torch.bfloat16)
        if value_gate:self.value_embed_gate=nn.Linear(16,heads,bias=False,dtype=torch.bfloat16)


class BlockWeights(nn.Module):
    def __init__(self,width,heads,value_gate,hidden):
        super().__init__();self.attn=AttentionWeights(width,heads,value_gate)
        self.mlp=nn.Module()
        self.mlp.register_parameter('c_fc',nn.Parameter(torch.empty(2*hidden,width,dtype=torch.float32)))
        self.mlp.register_parameter('c_proj',nn.Parameter(torch.empty(hidden,width,dtype=torch.float32)))


class ChessLM(nn.Module):
    def __init__(self,config):
        super().__init__();c=self.config=SimpleNamespace(**dict(config))
        w,l,h,v=c.width,c.layers,c.width//c.head_dim,c.vocab_size
        self.smear_gate=nn.Linear(16,1,bias=False,dtype=torch.bfloat16)
        self.skip_gates=nn.ModuleList([nn.Linear(16,1,bias=False,dtype=torch.bfloat16) for _ in range(3)])
        ne=min(5,l//2)
        self.value_embeds=nn.ModuleList([nn.Embedding(v,w,dtype=torch.bfloat16) for _ in range(ne)])
        self.blocks=nn.ModuleList([BlockWeights(w,h,i<ne or i>=l-ne,c.mlp_hidden) for i in range(l)])
        self.lm_head=nn.Linear(w,v,bias=False,dtype=torch.bfloat16)
        self.embed=nn.Embedding(v,w,dtype=torch.bfloat16)
        self.embed2=nn.Embedding(v,w,dtype=torch.bfloat16)
        self.clock_embed=nn.Embedding(64,w,dtype=torch.bfloat16)  # in the checkpoints, unused at feats=3
        self.elo_embed=nn.Embedding(64,w,dtype=torch.bfloat16)
        self.feat_embed=nn.Embedding(64,w,dtype=torch.bfloat16)
        self.board=BoardConv(w)
        self.x0_lambdas=nn.Parameter(torch.empty(2*l,dtype=torch.float32))
        self.scalars=nn.Parameter(torch.empty(c.scalars_size,dtype=torch.float32))
        self.register_buffer('cos',torch.empty(c.rotary_length,c.head_dim//2,dtype=torch.bfloat16))
        self.register_buffer('sin',torch.empty(c.rotary_length,c.head_dim//2,dtype=torch.bfloat16))
        self.long_layers={round(i*(l-1)/15) for i in (0,4,11,15)}
        self.skip_in=[i*l//16 for i in (2,4,6)]
        self.skip_out=[9*l//16+i for i in range(3)]

    def norm(self,x):
        # Native PyTorch RMSNorm uses the accumulation dtype's epsilon (FP32 for BF16 inputs).
        return F.rms_norm(x,(x.size(-1),),eps=self.config.norm_eps)

    def rotary(self,x,positions):
        cos=self.cos[positions,None,:].to(x.dtype);sin=self.sin[positions,None,:].to(x.dtype)
        a,b=x.chunk(2,-1)
        return torch.cat((a*cos+b*sin,a*(-sin)+b*cos),-1)

    def forward_hidden(self,ids,positions,backend,feats,states):
        c=self.config;w,l,hd=c.width,c.layers,c.head_dim;h=w//hd
        x=F.embedding(ids,self.embed.weight if c.split_embed else self.lm_head.weight)
        t=feats[:,:c.feats].float()
        v=torch.log1p(t.clamp(min=0))[...,None]/10
        freqs=torch.pi*2.0**torch.arange(8,device=t.device)
        f=torch.cat((torch.sin(v*freqs),torch.cos(v*freqs),v,torch.ones_like(v)),-1)
        f=(f*(t>=0)[...,None]).flatten(1)
        f=F.pad(f,(0,64-f.shape[1]))
        x=x+f.type_as(x)@self.feat_embed.weight.type_as(x)
        x=x+self.board(states,x.dtype)
        previous=backend.previous_embedding(x,positions)
        x=x+self.scalars[3*l]*torch.sigmoid(linear(x[...,:16],self.smear_gate.weight))*previous
        x=x0=self.norm(x);x02=self.norm(F.embedding(ids,self.embed2.weight))
        ne=len(self.value_embeds);values=[e(ids) for e in self.value_embeds]
        values=values+[None]*(l-2*ne)+values
        x0lam=self.x0_lambdas.view(l,2);sa=self.scalars[l:3*l].view(l,2)
        skips=[];skip_idx=0;backout=None
        for i,block in enumerate(self.blocks):
            if i in self.skip_out:
                gate=2*torch.sigmoid(self.scalars[3*l+2+skip_idx])*torch.sigmoid(linear(x0[...,:16],self.skip_gates[skip_idx].weight))
                x=x+gate*skips.pop();skip_idx+=1
            if i==0:x=(self.scalars[0]+x0lam[0,0])*x+x0lam[0,1]*x02
            else:x=self.scalars[i]*x+x0lam[i,0]*x0+x0lam[i,1]*x02
            a=block.attn;an=self.norm(x)
            q,k,v=F.linear(an,sa[i,0]*a.qkvo_w[:3*w].to(an.dtype)).view(-1,3*h,hd).chunk(3,dim=-2)
            q=self.rotary(self.norm(q),positions);k=self.rotary(self.norm(k),positions)
            if values[i] is not None:
                gate=2*torch.sigmoid(linear(an[...,:16],a.value_embed_gate.weight)).view(-1,h,1)
                v=v+gate*values[i].view_as(v)
            window=c.ws_long*128 if i in self.long_layers else c.ws_short*128
            y=backend.attention(q,k,v,i,window,c.attn_scale).view(-1,h,hd)
            y=y*torch.sigmoid(linear(an[...,:16],a.attn_gate.weight)).view(-1,h,1)
            y=y.reshape(-1,w)
            x=x+F.linear(y,sa[i,1]*a.qkvo_w[3*w:].to(y.dtype))
            mn=self.norm(x);m=linear(mn,block.mlp.c_fc)
            ma,mb=m.chunk(2,-1);m=F.silu(ma)*mb
            x=x+linear(m,block.mlp.c_proj.T)
            if i in self.skip_in:skips.append(x)
            if i==self.skip_out[-1]:backout=x
        return self.norm(x-self.scalars[3*l+1]*backout)

    def logits(self,hidden):
        z=linear(hidden,self.lm_head.weight)
        return 23*torch.sigmoid((z+5)/7.5)

    def forward(self,ids,positions,backend,feats,states):
        return self.logits(self.forward_hidden(ids,positions,backend,feats,states))


class DenseBackend:
    """One game, full prefix: eager SDPA on CPU or CUDA."""
    def previous_embedding(self,x,positions):
        return torch.cat((torch.zeros_like(x[:1]),x[:-1]))

    def attention(self,q,k,v,layer,window,scale):
        q,k,v=(x.transpose(0,1)[None] for x in (q,k,v));n=q.shape[-2]
        ix=torch.arange(n,device=q.device);mask=(ix[:,None]>=ix[None,:])&(ix[:,None]-ix[None,:]<=window)
        return F.scaled_dot_product_attention(q,k,v,attn_mask=mask,scale=scale)[0].transpose(0,1)


def load_checkpoint(path,device='cpu'):
    from pathlib import Path
    path=Path(path);state=torch.load(path,map_location='cpu',weights_only=False)
    if 'directory' in state:
        path=path.parent/state['directory']/'model.pt'
        state=torch.load(path,map_location='cpu',weights_only=False)
    cfg=state['config'];inf=state['inference'];weights=state['model']
    arch=cfg.get('arch',{})
    assert arch=={'board':'conv','key_offset':False,'mlp':'swiglu'},arch
    assert cfg.get('feats')==3 and not cfg.get('clock') and not cfg.get('elo')
    assert not cfg.get('doc_rope',False) and not cfg.get('rope_fp32',False)
    assert all(cfg.get(k,True) for k in ('value_embeds','skips','smear'))
    config=dict(width=cfg['width'],layers=cfg['layers'],head_dim=cfg['head_dim'],
        vocab_size=weights['lm_head.weight'].shape[0],scalars_size=weights['scalars'].numel(),
        rotary_length=inf['yarn']['cos'].shape[0],split_embed=inf['split_embed'],
        ws_short=inf['ws_short'],ws_long=inf['ws_long'],attn_scale=inf['yarn']['attn_scale'],
        feats=cfg['feats'],mlp_hidden=weights['blocks.0.mlp.c_proj'].shape[0],norm_eps=torch.finfo(torch.float32).eps)
    model=ChessLM(config)
    model.load_state_dict(dict(weights)|dict(cos=inf['yarn']['cos'],sin=inf['yarn']['sin']))
    return model.to(device).eval(),path
