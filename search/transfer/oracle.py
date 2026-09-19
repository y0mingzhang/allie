"""In-process SGLang execution for batched chess tree edges.

Uses the pinned engine's ModelRunner, paged attention and compiled CUDA graphs,
without generation requests/tokenizers/samplers/HTTP. A bounded research batch
owns the entire cache; reset() releases it between independent sets of roots.
"""
import os
import importlib.util
import time
from pathlib import Path
import numpy as np
import torch


class ShipOracle:
    def __init__(self,model_path,capacity=262144,mem_fraction_static=.7):
        start=time.monotonic()
        os.environ['SGLANG_EXTERNAL_MODEL_PACKAGE']='search.transfer.sglang_models'
        # Our explicit custom architecture is the only model this process serves.
        # Avoid eagerly importing every unrelated built-in model on shared NFS.
        package=Path(importlib.util.find_spec('sglang').origin).parent
        os.environ['SGLANG_DISABLED_MODEL_ARCHS']=','.join(p.stem for p in (package/'srt/models').glob('*.py'))
        os.environ.pop('ALLIE_BINARY_OUTPUT',None)
        from sglang.srt.server_args import ServerArgs,PortArgs
        from search.engine.startup_env import setup as _set_envs_and_config
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.layers.moe import initialize_moe_config
        from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
        from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
        root=Path(__file__).resolve().parents[2]
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
