"""Compact move LM with portable modded-NanoGPT/Slowrun ingredients.

No board features. RoPE/QK norm, SwiGLU, input mixing, U-Net skips and value
projections are fixed across the initial size sweep. KV cache is explicit.
"""
from dataclasses import dataclass, asdict
import math
import os
# Each DDP rank gets four CPUs; avoid32 compiler workers per rank.
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','4')
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    layers:int=12
    width:int=384
    heads:int=6
    ff:int=1024
    vocab:int=2350
    dropout:float=0.
    rope_theta:float=10000.
    extras:bool=True
    bf16_residual:bool=False
    mlp:str='swiglu'
    game_attention:bool=False


class Attention(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg;self.dim=cfg.width//cfg.heads
        self.qkv=nn.Linear(cfg.width,3*cfg.width,bias=False)
        self.out=nn.Linear(cfg.width,cfg.width,bias=False)
        self.value_mix=nn.Parameter(torch.tensor(.1))
        self.register_buffer('inv_freq',cfg.rope_theta**(-torch.arange(0,self.dim,2).float()/self.dim),persistent=False)

    def rope(self,x,start):
        pos=torch.arange(start,start+x.shape[-2],device=x.device,dtype=torch.float32)
        phase=torch.outer(pos,self.inv_freq)
        cos,sin=phase.cos().to(x.dtype),phase.sin().to(x.dtype)
        even,odd=x[...,0::2],x[...,1::2]
        return torch.stack([even*cos-odd*sin,even*sin+odd*cos],dim=-1).flatten(-2)

    def forward(self,x,value=None,past=None,use_cache=False,game_mask=None):
        b,t,d=x.shape
        q,k,v=self.qkv(x).reshape(b,t,3,self.cfg.heads,self.dim).unbind(2)
        q,k=F.rms_norm(q,(self.dim,)),F.rms_norm(k,(self.dim,))
        if value is not None:
            v=v+self.value_mix*value.reshape(b,t,self.cfg.heads,self.dim)
        q,k,v=(z.transpose(1,2) for z in (q,k,v))
        start=0 if past is None else past[0].shape[-2]
        q,k=self.rope(q,start),self.rope(k,start)
        if past is not None:
            k=torch.cat([past[0],k],dim=-2);v=torch.cat([past[1],v],dim=-2)
        mask=None;causal=past is None
        if past is not None and t>1:
            mask=torch.arange(k.shape[-2],device=x.device)[None,:]<=torch.arange(start,start+t,device=x.device)[:,None]
        if game_mask is not None and not isinstance(game_mask,torch.Tensor):
            from lm_attention import compiled_flex
            # A later game can have no valid keys in the first scheduled KV tile.
            # Keep FlexAttention's softmax guard even though every row sees itself.
            y=compiled_flex(q,k,v,block_mask=game_mask)
        else:
            if game_mask is not None:mask=game_mask;causal=False
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=causal,
                                            dropout_p=self.cfg.dropout if self.training else 0.)
        return self.out(y.transpose(1,2).reshape(b,t,d)),(k,v) if use_cache else None


class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.attn=Attention(cfg)
        self.norm1=nn.RMSNorm(cfg.width);self.norm2=nn.RMSNorm(cfg.width)
        assert cfg.mlp in ['swiglu','relu2']
        self.mlp=cfg.mlp
        # ReLU^2 gets 1.5x hidden width to match SwiGLU's parameter/matmul budget.
        hidden=cfg.ff if cfg.mlp=='swiglu' else 3*cfg.ff//2
        self.up=nn.Linear(cfg.width,2*hidden if cfg.mlp=='swiglu' else hidden,bias=False)
        self.down=nn.Linear(hidden,cfg.width,bias=False)
        self.dropout=cfg.dropout

    def forward(self,x,value=None,past=None,use_cache=False,game_mask=None):
        y,cache=self.attn(self.norm1(x),value,past,use_cache,game_mask)
        x=x+F.dropout(y,self.dropout,self.training)
        h=self.up(self.norm2(x))
        if self.mlp=='swiglu':
            gate,value=h.chunk(2,-1);h=F.silu(gate)*value
        else:h=F.relu(h).square()
        x=x+F.dropout(self.down(h),self.dropout,self.training)
        return x,cache


class MoveLM(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        assert cfg.layers%2==0 and cfg.width%cfg.heads==0 and (cfg.width//cfg.heads)%2==0
        assert cfg.mlp!='relu2' or cfg.ff%2==0
        assert not cfg.game_attention or cfg.dropout==0.,'Flex attention currently requires attention dropout zero'
        self.embed=nn.Embedding(cfg.vocab,cfg.width)
        self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.value_proj=nn.ModuleList([nn.Linear(cfg.width,cfg.width,bias=False) for _ in range(min(3,cfg.layers//2) if cfg.extras else 0)])
        self.input_mix=nn.Parameter(torch.tensor([[1.,0.]]*cfg.layers))
        self.skip_mix=nn.Parameter(torch.full((cfg.layers//2,),.1))
        self.norm=nn.RMSNorm(cfg.width)
        self.head=nn.Linear(cfg.width,cfg.vocab,bias=False)
        if not cfg.extras:
            self.input_mix.requires_grad_(False);self.skip_mix.requires_grad_(False)
        nv=len(self.value_proj)
        for i, block in enumerate(self.blocks):
            if not (i<nv or i>=cfg.layers-nv):block.attn.value_mix.requires_grad_(False)
        for module in self.modules():
            if isinstance(module,(nn.Linear,nn.Embedding)):
                nn.init.normal_(module.weight,mean=0,std=.02)
        # Zero-initialized residual projections, as in the speedrun baseline.
        for block in self.blocks:
            nn.init.zeros_(block.attn.out.weight);nn.init.zeros_(block.down.weight)

    def forward(self,ids,past=None,use_cache=False,targets=None):
        mask=None;documents=None
        if self.cfg.game_attention:
            from lm_attention import game_mask
            mask,documents=game_mask(ids,None if past is None else past[0][2])
        embedding=self.embed(ids)
        if self.cfg.bf16_residual and embedding.is_cuda:embedding=embedding.bfloat16()
        x0=F.rms_norm(embedding,(self.cfg.width,))
        x=x0;skips=[];caches=[]
        values=[proj(x0) for proj in self.value_proj]
        n=len(self.blocks);nv=len(values)
        for i,block in enumerate(self.blocks):
            if self.cfg.extras:x=self.input_mix[i,0]*x+self.input_mix[i,1]*x0
            if self.cfg.extras and i>=n//2:
                x=x+self.skip_mix[i-n//2]*skips.pop()
            value=values[i] if i<nv else (values[n-1-i] if i>=n-nv else None)
            x,cache=block(x,value,None if past is None else past[i],use_cache,mask)
            if self.cfg.extras and i<n//2:skips.append(x)
            if use_cache:caches.append((*cache,documents) if self.cfg.game_attention else cache)
        logits=self.head(self.norm(x))
        if targets is not None:
            labels=(targets-378).masked_fill((targets<378)|(targets>2345),-100)
            return F.cross_entropy(logits[...,378:2346].float().reshape(-1,1968),labels.reshape(-1))
        return (logits,caches) if use_cache else logits

    def parameters_count(self):
        return sum(p.numel() for p in self.parameters())

    def flop_estimate(self,context=128):
        # Matmul FLOPs for one cached next-move prediction; counts shared projections
        # by actual calls, not unique parameters. Adds QK^T and attention-value cost.
        c=self.cfg
        dense=2*(c.layers*(4*c.width*c.width+3*c.width*c.ff)+len(self.value_proj)*c.width*c.width+c.width*c.vocab)
        attention=4*c.layers*context*c.width
        return dict(matmul_flops_per_cached_move=dense+attention,context=context,
                    note='Analytical matmul FLOPs; excludes elementwise/norm/softmax and cache initialization.')


@torch.compile
def zeropower(g):
    x=g.bfloat16()
    transposed=x.shape[-2]>x.shape[-1]
    if transposed:x=x.transpose(-1,-2)
    x=x/(x.norm(dim=(-2,-1),keepdim=True)+1e-7)
    for _ in range(5):
        a=x@x.transpose(-1,-2)
        b=-4.775*a+2.0315*(a@a)
        x=3.4445*x+b@x
    return x.transpose(-1,-2) if transposed else x


class BatchedMuon(torch.optim.Optimizer):
    def __init__(self,params,lr=.003,weight_decay=.1,momentum=.95):
        groups={}
        for p in params:
            assert p.ndim==2
            groups.setdefault(tuple(p.shape),[]).append(p)
        super().__init__([{'params':ps} for ps in groups.values()],dict(lr=lr,weight_decay=weight_decay,momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            ps=group['params'];grads=torch.stack([p.grad for p in ps])
            state=self.state[ps[0]]
            if 'momentum' not in state:state['momentum']=torch.zeros_like(grads)
            buf=state['momentum'];buf.lerp_(grads,1-group['momentum'])
            update=zeropower(grads.lerp(buf,group['momentum']))
            # Match-RMS AdamW scaling, as used in chess-v2's Muon implementation.
            lr=group['lr']*.2*math.sqrt(max(ps[0].shape))
            for p,u in zip(ps,update.unbind()):
                p.mul_(1-group['lr']*group['weight_decay']).add_(u,alpha=-lr)


def optimizers(model,lr=.003,weight_decay=.1):
    matrix=[];adam_decay=[];adam_no_decay=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        if p.ndim==2 and name not in ['embed.weight','head.weight','input_mix']:
            matrix.append(p)
        elif name in ['embed.weight','head.weight']:
            adam_decay.append(p)
        else:adam_no_decay.append(p)
    muon=BatchedMuon(matrix,lr=lr,weight_decay=weight_decay)
    adam=torch.optim.AdamW([{'params':adam_decay,'weight_decay':weight_decay},
                           {'params':adam_no_decay,'weight_decay':0.}],lr=lr/3,betas=(.9,.95),fused=True)
    return [muon,adam]
