"""Causal attention isolated at original BOS boundaries; no new input features."""
import torch
from torch.nn.attention.flex_attention import create_block_mask,flex_attention

compiled_flex=torch.compile(flex_attention,dynamic=False)

@torch.compiler.disable
def game_mask(ids,past_documents=None):
    documents=(ids==2348).to(torch.int32).cumsum(-1)
    if past_documents is not None:
        documents=documents+past_documents[:,-1:]
        documents=torch.cat([past_documents,documents],dim=1)
    batch,length=ids.shape;total=documents.shape[1];start=total-length
    if not ids.is_cuda or past_documents is not None:
        q=torch.arange(start,total,device=ids.device)
        k=torch.arange(total,device=ids.device)
        mask=(q[:,None]>=k[None,:])[None,None]&(
            documents[:,None,start:,None]==documents[:,None,None,:])
    else:
        def allowed(b,h,q,k):
            return (q<length)&(k<total)&(q+start>=k)&(
                documents[b,(q+start).clamp(max=total-1)]==documents[b,k.clamp(max=total-1)])
        mask=create_block_mask(allowed,batch,None,length,total,device=ids.device,
            BLOCK_SIZE=128,_compile=True)
    return mask,documents
