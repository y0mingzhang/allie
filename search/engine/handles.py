"""Compact node-handle bridge to the existing resident SGLang runner.

Only root prefill uses token histories. Child queries carry native node, parent,
new move and sequence length. KV indices are inherited on device; no CPU tuple
hashing, full-history serialization, or per-node CPU KV-path construction.
"""
import time
import numpy as np
import torch

class HandleOracle:
    def __init__(self,base,prefixes):
        self.base=base
        self.root_logits=base(prefixes)
        self.rows=np.full(base.capacity,-1,np.int32)
        self.lengths=np.zeros(base.capacity,np.int32)
        for i,p in enumerate(prefixes):
            self.rows[i]=base.nodes[tuple(p)];self.lengths[i]=len(p)
        self.queries=0

    @torch.no_grad()
    def __call__(self,handles):
        handles=np.asarray(handles,dtype=np.int32)
        assert handles.ndim==2 and handles.shape[1]==4
        outputs=[]
        for lo in range(0,len(handles),4096):outputs.append(self._forward(handles[lo:lo+4096]))
        return np.concatenate(outputs,axis=0)

    def _forward(self,h):
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch,ForwardMode,CaptureHiddenMode
        base=self.base;r=base.runner
        ids,parents,tokens,lengths=h.T;n=len(h)
        assert n and ids.min()>=0 and ids.max()<len(self.rows)
        assert np.all(self.rows[ids]<0),'A search node must be evaluated exactly once'
        assert np.all(parents>=0) and parents.max()<len(self.rows)
        parent_rows=self.rows[parents].copy()
        assert np.all(parent_rows>=0),'Parents must already be evaluated'
        assert np.all(lengths==self.lengths[parents]+1)
        assert tokens.min()>=378 and tokens.max()<2346 and lengths.max()<=1025
        if base.next_row+n>base.capacity:raise RuntimeError('Tree row cache full')
        if base.next_slot+n>r.max_total_num_tokens:raise RuntimeError('Tree KV cache full')
        rows=np.arange(base.next_row,base.next_row+n,dtype=np.int32)
        slots=np.arange(base.next_slot,base.next_slot+n,dtype=np.int32)
        base.next_row+=n;base.next_slot+=n;self.rows[ids]=rows;self.lengths[ids]=lengths
        # Entries beyond a parent's length are deliberately unspecified; attention
        # sees only seq_lens. The newly appended entry is always overwritten.
        t=lambda a:torch.as_tensor(a,device='cuda',dtype=torch.int32)
        gpu_rows=t(rows);gpu_lengths=t(lengths);gpu_slots=t(slots);maxlen=int(lengths.max())
        table=r.req_to_token_pool.req_to_token
        table[gpu_rows,:maxlen]=table[t(parent_rows),:maxlen]
        table[gpu_rows,gpu_lengths-1]=gpu_slots
        ones=torch.ones(n,device='cuda',dtype=torch.int32)
        b=ForwardBatch(forward_mode=ForwardMode.EXTEND,batch_size=n,
            input_ids=t(tokens).long(),req_pool_indices=gpu_rows,seq_lens=gpu_lengths,
            seq_lens_cpu=torch.from_numpy(lengths.copy()),out_cache_loc=gpu_slots,seq_lens_sum=int(lengths.sum()),
            positions=(gpu_lengths-1).long(),extend_num_tokens=n,extend_seq_lens=ones,
            extend_prefix_lens=gpu_lengths-1,extend_start_loc=torch.arange(n,device='cuda',dtype=torch.int32),
            extend_prefix_lens_cpu=(lengths-1).tolist(),extend_seq_lens_cpu=[1]*n,extend_logprob_start_lens_cpu=[1]*n,
            req_to_token_pool=r.req_to_token_pool,token_to_kv_pool=r.token_to_kv_pool,
            attn_backend=r.attn_backend,return_logprob=False,capture_hidden_mode=CaptureHiddenMode.NULL)
        start=time.monotonic();out,_=r.forward_extend(b)
        result=out.next_token_logits.float().cpu().numpy()
        base.forward_seconds+=time.monotonic()-start;base.new_tokens+=n;base.calls+=1;self.queries+=n
        return result
