"""Run the pinned SGLang setup function without importing its serving stack.

We execute the exact, hash-checked function AST from the installed Apache-2.0
SGLang0.5.9 engine.py. Its globals are explicit; tokenizer/scheduler/image imports
are unnecessary for the standalone ModelRunner. A dependency update fails closed.
"""
import ast
import hashlib
import importlib.util
import logging
import multiprocessing as mp
import os
from pathlib import Path
import random
import signal
import time

FUNCTION_SHA256='d9c432e8c99c1df678f40166dcb703d062e35f657b686f88876fada65543c202'


def setup(server_args):
    from sglang.srt.utils import (assert_pkg_version,get_bool_env_var,is_cuda,
                                  kill_process_tree,set_prometheus_multiproc_dir,set_ulimit)
    package=Path(importlib.util.find_spec('sglang').origin).parent
    source=package/'srt/entrypoints/engine.py';text=source.read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='_set_envs_and_config')
    assert hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()==FUNCTION_SHA256,'Review changed SGLang setup before running'
    scope=dict(os=os,time=time,random=random,signal=signal,mp=mp,
               ServerArgs=type(server_args),logger=logging.getLogger(__name__),_is_cuda=is_cuda(),
               assert_pkg_version=assert_pkg_version,get_bool_env_var=get_bool_env_var,
               kill_process_tree=kill_process_tree,set_prometheus_multiproc_dir=set_prometheus_multiproc_dir,
               set_ulimit=set_ulimit)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    scope['_set_envs_and_config'](server_args)

import numpy as np
import torch
from .board import advance_boards, advance_clocks, predicted_seconds, root_other_previous

class ShipOracle:
    def handles(self, prefixes, features, clock_rule='predicted'):
        return ShipHandles(self, prefixes, features, clock_rule)

    def __init__(self,model_path,capacity=262144,mem_fraction_static=.7):
        start=time.monotonic()
        os.environ['SGLANG_EXTERNAL_MODEL_PACKAGE']='allie.search.sglang_models'
        # Our explicit custom architecture is the only model this process serves.
        # Avoid eagerly importing every unrelated built-in model on shared NFS.
        package=Path(importlib.util.find_spec('sglang').origin).parent
        os.environ['SGLANG_DISABLED_MODEL_ARCHS']=','.join(p.stem for p in (package/'srt/models').glob('*.py'))
        os.environ.pop('ALLIE_BINARY_OUTPUT',None)
        from sglang.srt.server_args import ServerArgs,PortArgs
        from .runtime import setup as _set_envs_and_config
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.layers.moe import initialize_moe_config
        from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
        from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
        args=ServerArgs(model_path=str(model_path),
            trust_remote_code=True,skip_tokenizer_init=True,dtype='bfloat16',
            attention_backend='flashinfer',context_length=1025,mem_fraction_static=mem_fraction_static,
            max_total_tokens=capacity,max_running_requests=capacity,
            disable_cuda_graph=True,disable_overlap_schedule=True,
            enable_piecewise_cuda_graph=True,piecewise_cuda_graph_compiler='inductor',
            piecewise_cuda_graph_tokens=[16,64,256,512,1024,2048,4096],
            chunked_prefill_size=4096,max_prefill_tokens=4096)
        _set_envs_and_config(args)
        initialize_moe_config(args);initialize_fp4_gemm_config(args);initialize_fp8_gemm_config(args)
        ports=PortArgs.init_new(args)
        self.runner=ModelRunner(model_config=ModelConfig.from_server_args(args),
            mem_fraction_static=args.mem_fraction_static,gpu_id=0,tp_rank=0,tp_size=1,
            moe_ep_rank=0,moe_ep_size=1,pp_rank=0,pp_size=1,nccl_port=ports.nccl_port,server_args=args)
        self.capacity=min(capacity,self.runner.req_to_token_pool.size)
        self.reset()
        self.startup_seconds=time.monotonic()-start

    def reset(self):
        self.nodes={};self.paths={};self.root_metadata={};self.last_root_rows=[];self.next_row=0;self.next_slot=1
        self.calls=0;self.new_tokens=0;self.forward_seconds=0.

    @torch.no_grad()
    def _forward(self,records,columns):
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch,ForwardMode,CaptureHiddenMode
        r=self.runner;table=r.req_to_token_pool.req_to_token
        seqs=[];rows=[];prefix=[];lengths=[];slots=[];positions=[];feats=[];states=[]
        maxlen=max(len(x[0]) for x in records)
        mapping=np.zeros((len(records),maxlen),dtype=np.int32)
        for ix,(key,row,parent,plen) in enumerate(records):
            n=len(key)-plen
            ff,bb=self.root_metadata[row];feats.extend(ff[plen:]);states.extend(bb[plen:])
            if self.next_slot+n>r.max_total_num_tokens:raise RuntimeError('Tree KV cache full; reset between root batches')
            loc=list(range(self.next_slot,self.next_slot+n));self.next_slot+=n
            if parent is not None:mapping[ix,:plen]=self.paths[parent][:plen]
            mapping[ix,plen:len(key)]=loc
            self.paths[row]=mapping[ix,:len(key)]
            seqs.extend(key[plen:]);positions.extend(range(plen,len(key)));slots.extend(loc)
            rows.append(row);prefix.append(plen);lengths.append(len(key))
        ext=[l-p for l,p in zip(lengths,prefix)];starts=np.cumsum([0]+ext[:-1]).tolist()
        t=lambda x:torch.tensor(x,device='cuda',dtype=torch.int32)
        gpu_rows=t(rows)
        table[gpu_rows,:maxlen]=torch.as_tensor(mapping,device='cuda')
        b=ForwardBatch(forward_mode=ForwardMode.EXTEND,batch_size=len(rows),
            input_ids=t(seqs).long(),req_pool_indices=gpu_rows,seq_lens=t(lengths),
            seq_lens_cpu=torch.tensor(lengths,dtype=torch.int32),out_cache_loc=t(slots),seq_lens_sum=sum(lengths),
            positions=t(positions).long(),extend_num_tokens=len(seqs),extend_seq_lens=t(ext),
            extend_prefix_lens=t(prefix),extend_start_loc=t(starts),extend_prefix_lens_cpu=prefix,
            extend_seq_lens_cpu=ext,extend_logprob_start_lens_cpu=ext,
            req_to_token_pool=r.req_to_token_pool,token_to_kv_pool=r.token_to_kv_pool,
            attn_backend=r.attn_backend,return_logprob=False,capture_hidden_mode=CaptureHiddenMode.NULL)
        side=r.model.model
        side.side_feats[:len(seqs)].copy_(torch.as_tensor(np.asarray(feats),device='cuda',dtype=torch.float32))
        side.side_boards[:len(seqs)].copy_(torch.as_tensor(np.asarray(states),device='cuda',dtype=torch.uint8))
        start=time.monotonic()
        out,_=r.forward_extend(b)
        scores=out.next_token_logits
        if columns is not None:scores=scores[:,columns]
        result=scores.float().cpu().numpy()
        self.forward_seconds+=time.monotonic()-start;self.new_tokens+=len(seqs);self.calls+=1
        return result

    def prefill(self,prefixes,features,columns=None):
        # Root ownership is explicit: identical token histories with different clocks
        # MUST NOT share KV state. No cross-query prefix reuse in this comparison.
        from .board import encode
        records=[];outputs=[];ntokens=0;self.last_root_rows=[]
        def flush():
            nonlocal records,ntokens
            if not records:return
            outputs.extend(self._forward(records,columns));records=[];ntokens=0
        for prefix,feat in zip(prefixes,features):
            key=tuple(prefix);assert 1<=len(key)<=1025
            assert np.asarray(feat).shape==(len(key),3)
            row=self.next_row;self.next_row+=1
            assert row<self.capacity
            board=encode(np.array(prefix,dtype=np.int64)[None])[0]
            self.root_metadata[row]=(np.asarray(feat),board)
            self.last_root_rows.append(row)
            if ntokens+len(key)>4096:flush()
            records.append((key,row,None,0));ntokens+=len(key)
        flush()
        return np.asarray(outputs)

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

class ShipHandles(HandleOracle):
    def __init__(self,base,prefixes,features,clock_rule='predicted'):
        assert clock_rule in ('predicted','zero')
        self.base=base;self.clock_rule=clock_rule
        self.root_logits=base.prefill(prefixes,features)
        n=len(prefixes);cap=base.capacity
        self.rows=np.full(cap,-1,np.int32);self.lengths=np.zeros(cap,np.int32)
        self.boards=np.zeros((cap,68),np.uint8);self.feats=np.full((cap,3),-1.,np.float32)
        self.other_previous=np.full(cap,-1.,np.float32);self.inc=np.full(cap,-1.,np.float32)
        self.elapsed=np.zeros(cap,np.float32);self.queries=0
        self.owners=np.full(cap,-1,np.int32);self.owners[:n]=np.arange(n)
        self.per_root_queries=np.zeros(n,np.int64)
        for i,(p,feat,row) in enumerate(zip(prefixes,features,base.last_root_rows)):
            self.rows[i]=row;self.lengths[i]=len(p)
            self.boards[i]=base.root_metadata[row][1][-1];self.feats[i]=feat[-1]
            self.inc[i]=p[2]-10 if 10<=p[2]<191 else -1
            self.other_previous[i]=root_other_previous(p,feat,self.inc[i])
        self.elapsed[:n]=predicted_seconds(self.root_logits) if clock_rule=='predicted' else 0

    def _forward(self,handles):
        ids,parents,tokens,lengths=handles.T
        self.owners[ids]=self.owners[parents]
        np.add.at(self.per_root_queries,self.owners[ids],1)
        self.boards[ids]=advance_boards(self.boards[parents],tokens)
        self.feats[ids],self.other_previous[ids]=advance_clocks(self.feats[parents],self.other_previous[parents],
            self.lengths[parents],self.inc[parents],self.elapsed[parents])
        self.inc[ids]=self.inc[parents]
        side=self.base.runner.model.model;n=len(ids)
        side.side_feats[:n].copy_(torch.as_tensor(self.feats[ids],device='cuda'))
        side.side_boards[:n].copy_(torch.as_tensor(self.boards[ids],device='cuda'))
        result=super()._forward(handles)
        if self.clock_rule=='predicted':self.elapsed[ids]=predicted_seconds(result)
        return result
