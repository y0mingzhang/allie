"""Inference-only block metadata for 128-aligned prefix packs.

The token-level mask is unchanged. Construct its block metadata from monotone
document IDs instead of evaluating a dense length-squared mask. Unaligned games
are rejected, so this cannot silently substitute for the training row packer.
"""
import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask


def metadata(x,window):
    x=np.asarray(x);length=len(x)
    assert x.ndim==1 and length%128==0 and x[0]==2348 and window>=0
    start=x==2348;docs=np.cumsum(start,dtype=np.int64)
    # Every non-padding document must begin on a block boundary. Padding may
    # consist of consecutive BOS documents inside a partly used final block.
    real=np.flatnonzero(start & np.r_[x[1:]!=2348,False])
    assert np.all(real%128==0),'Only 128-aligned prefix packs are supported'
    q=np.arange(length//128)[:,None];k=np.arange(length//128)[None,:]
    first=docs[::128];last=docs[127::128]
    distance=(q-k)*128
    nonempty=(q==k)|((q>k)&(first[:,None]==last[None,:])&(distance-127<=window))
    full=(q>k)&(first[:,None]==last[None,:])&(first[:,None]==last[:,None])&(distance+127<=window)
    partial=nonempty&~full
    def sparse(mask):
        # Match ascending active KV order from create_block_mask.
        indices=np.argsort(~mask,axis=1,kind='stable').astype(np.int32)
        count=mask.sum(1,dtype=np.int32)
        return count[None,None],indices[None,None]
    return docs,start,sparse(partial),sparse(full)


def make_context(x,device_inputs,short_window,long_window,Context):
    docs_np,starts,_,_=metadata(x,short_window)
    device=device_inputs.device;length=len(x)
    docs=torch.as_tensor(docs_np,device=device);same=torch.as_tensor(~starts,device=device)
    def make(window):
        _,_,partial,full=metadata(x,window)
        def allowed(b,h,q,k):
            return ((q<length)&(k<length)&(q>=k)&(q-k<=window)&
                    (docs[q.clamp(max=length-1)]==docs[k.clamp(max=length-1)]))
        tensors=[torch.as_tensor(a,device=device) for pair in (partial,full) for a in pair]
        return BlockMask.from_kv_blocks(*tensors,BLOCK_SIZE=128,mask_mod=allowed,
                                       seq_lengths=(length,length),compute_q_blocks=False)
    short=make(short_window);long=short if short_window==long_window else make(long_window)
    return Context(docs,same,short_window,long_window,short,long,'flex')
