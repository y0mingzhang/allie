"""Experimental integer-only batch plumbing; model and attention math unchanged.

Fill child KV-slot maps on the GPU from their parent rows. This avoids copying every
prefix slot through Python/Numpy and the host-to-device link for each tree edge.
The existing runner remains the reference and is restored after every benchmark.
"""
import time
import numpy as np
import torch
import triton
import triton.language as tl
from .direct import DirectOracle


@triton.jit
def inherit_rows(table, records, stride:tl.constexpr, B:tl.constexpr):
    i=tl.program_id(0);x=tl.arange(0,B)
    row=tl.load(records+i*5);parent=tl.load(records+i*5+1)
    prefix=tl.load(records+i*5+2);length=tl.load(records+i*5+3);slot=tl.load(records+i*5+4)
    previous=tl.load(table+tl.maximum(parent,0)*stride+x,mask=(parent>=0)&(x<prefix),other=0)
    current=tl.where(x<prefix,previous,slot+x-prefix)
    tl.store(table+row*stride+x,current,mask=x<length)


class PackedOracle(DirectOracle):
    @torch.no_grad()
    def _forward(self,records,columns):
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch,ForwardMode,CaptureHiddenMode
        r=self.runner;table=r.req_to_token_pool.req_to_token
        seqs=[];rows=[];prefix=[];lengths=[];slots=[];positions=[];mapping=[]
        for key,row,parent,plen in records:
            n=len(key)-plen
            if self.next_slot+n>r.max_total_num_tokens:raise RuntimeError('Tree KV cache full')
            mapping.append([row,-1 if parent is None else parent,plen,len(key),self.next_slot])
            slots.extend(range(self.next_slot,self.next_slot+n));self.next_slot+=n
            seqs.extend(key[plen:]);positions.extend(range(plen,len(key)))
            rows.append(row);prefix.append(plen);lengths.append(len(key))
        ext=[l-p for l,p in zip(lengths,prefix)];starts=np.cumsum([0]+ext[:-1]).tolist()
        # One contiguous integer upload supplies all GPU-side request metadata.
        arrays=[rows,lengths,slots,positions,ext,prefix,starts]
        ends=np.cumsum([0]+[len(a) for a in arrays]);packed=np.concatenate(arrays).astype(np.int32)
        gpu=torch.as_tensor(packed,device='cuda');items=[gpu[ends[i]:ends[i+1]] for i in range(len(arrays))]
        gpu_rows,gpu_lengths,gpu_slots,gpu_positions,gpu_ext,gpu_prefix,gpu_starts=items
        mapping=torch.tensor(mapping,device='cuda',dtype=torch.int32)
        inherit_rows[(len(rows),)](table,mapping,table.stride(0),triton.next_power_of_2(max(lengths)))
        b=ForwardBatch(forward_mode=ForwardMode.EXTEND,batch_size=len(rows),
            input_ids=torch.tensor(seqs,device='cuda',dtype=torch.long),req_pool_indices=gpu_rows,seq_lens=gpu_lengths,
            seq_lens_cpu=torch.tensor(lengths,dtype=torch.int32),out_cache_loc=gpu_slots,seq_lens_sum=sum(lengths),
            positions=gpu_positions.long(),extend_num_tokens=len(seqs),extend_seq_lens=gpu_ext,
            extend_prefix_lens=gpu_prefix,extend_start_loc=gpu_starts,extend_prefix_lens_cpu=prefix,
            extend_seq_lens_cpu=ext,extend_logprob_start_lens_cpu=ext,
            req_to_token_pool=r.req_to_token_pool,token_to_kv_pool=r.token_to_kv_pool,
            attn_backend=r.attn_backend,return_logprob=False,capture_hidden_mode=CaptureHiddenMode.NULL)
        start=time.monotonic();out,_=r.forward_extend(b);scores=out.next_token_logits
        if columns is not None:scores=scores[:,columns]
        result=scores.float().cpu().numpy()
        self.forward_seconds+=time.monotonic()-start;self.new_tokens+=len(seqs);self.calls+=1
        return result
