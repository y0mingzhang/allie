"""Single-GPU SGLang adapter with radix-slot-aligned extra recurrent state.

Each physical KV slot also owns its pre-smear embedding and unshifted stationary
key dimensions. Shared prefixes consequently share exactly the same extra state;
eviction/reuse overwrites these entries together with the corresponding KV entry.
No mutable per-request Python state is needed for branching.
"""
import torch
from torch import nn
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from search.transfer.model import ChessLM


class AllieLogitsProcessor(LogitsProcessor):
    def _compute_lm_head(self,hidden_states,lm_head,embedding_bias=None):
        # The nonlinear head transform belongs before normalization. Preserve its
        # BF16 rounding; converting to FP32 before sigmoid changes this checkpoint.
        z=torch.nn.functional.linear(hidden_states,lm_head.weight.to(hidden_states.dtype))
        return (23*torch.sigmoid((z+5)/7.5)).float()


class RadixBackend:
    def __init__(self,owner,batch,positions):
        self.owner,self.batch=owner,batch
        self.slots=owner.current_slots[:positions.numel()]
        self.prev=owner.previous_slots[:positions.numel()]
        self.first=(positions==0)

    def previous_embedding(self,x,positions):
        store=self.owner.smear_cache
        store[self.slots]=x
        return torch.where(self.first[:,None],0,store[self.prev])

    def shift_keys(self,k,positions,layer):
        d=k.shape[-1];store=getattr(self.owner,f'raw_keys_{layer}')
        store[self.slots]=torch.cat((k[:,:,d//4:d//2],k[:,:,3*d//4:]),-1)
        prev=store[self.prev];out=k.clone()
        out[:,:,d//4:d//2]=torch.where(self.first[:,None,None],k[:,:,d//4:d//2],prev[:,:,:d//4])
        out[:,:,3*d//4:]=torch.where(self.first[:,None,None],k[:,:,3*d//4:],prev[:,:,d//4:])
        return out

    def attention(self,q,k,v,layer,window,scale):
        return self.owner.layers[layer].attn(q.reshape(q.shape[0],-1),k,v,self.batch)


class SGLangChessModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        from sglang.srt.server_args import get_global_server_args
        args=get_global_server_args();c=config.allie
        self.math=ChessLM(c)
        h=c['width']//c['head_dim'];self.layers=nn.ModuleList()
        for i in range(c['layers']):
            layer=nn.Module();layer.attn=RadixAttention(h,c['head_dim'],c['attn_scale'],h,i)
            self.layers.append(layer)
        assert args.max_total_tokens is not None, 'Set --max-total-tokens to bound side caches'
        capacity=args.max_total_tokens+max(args.page_size,64)
        self.register_buffer('current_slots',torch.zeros(args.max_prefill_tokens,dtype=torch.long),persistent=False)
        self.register_buffer('previous_slots',torch.zeros(args.max_prefill_tokens,dtype=torch.long),persistent=False)
        self.register_buffer('smear_cache',torch.zeros(capacity,c['width'],dtype=torch.bfloat16),persistent=False)
        self.register_buffer('side_feats',torch.full((args.max_prefill_tokens,3),-1.,dtype=torch.float32),persistent=False)
        self.register_buffer('side_boards',torch.zeros((args.max_prefill_tokens,68),dtype=torch.uint8),persistent=False)
        for i in self.math.long_layers if c['key_offset'] else []:
            self.register_buffer(f'raw_keys_{i}',torch.zeros(capacity,h,c['head_dim']//2,dtype=torch.bfloat16),persistent=False)

    def forward(self,input_ids:torch.Tensor,positions:torch.Tensor,forward_batch:ForwardBatch,**kwargs):
        backend=RadixBackend(self,forward_batch,positions)
        return self.math.forward_hidden(input_ids,positions,backend,feats=self.side_feats[:positions.numel()],states=self.side_boards[:positions.numel()])


class AllieShipForCausalLM(nn.Module):
    def __init__(self,config,quant_config=None,prefix=''):
        super().__init__()
        import os
        if os.environ.get('ALLIE_BINARY_OUTPUT')=='1':
            from search.engine.transport import install_binary_scores
            install_binary_scores()
        from sglang.srt.distributed import get_tensor_model_parallel_world_size
        assert get_tensor_model_parallel_world_size()==1, 'Initial port supports one GPU'
        assert quant_config is None, 'Quantization needs separate parity tests'
        c=config.allie
        assert not c['clock'] and not c['elo'], 'No side-channel request transport yet'
        assert min(c['ws_short'],c['ws_long'])*128>=config.max_position_embeddings
        self.model=SGLangChessModel(config)
        self.logits_processor=AllieLogitsProcessor(config)

    @torch.no_grad()
    def forward(self,input_ids,positions,forward_batch,**kwargs):
        # Batch metadata has a variable request dimension. Keep it outside the
        # piecewise-compiled transformer and copy to stable graph input buffers.
        b=forward_batch;n=positions.numel()
        assert n<=len(self.model.current_slots)
        assert self.model.smear_cache.shape[0]>=b.token_to_kv_pool.get_key_buffer(0).shape[0]
        if b.forward_mode.is_decode():requests=b.req_pool_indices.long()
        else:
            row=torch.searchsorted(b.extend_start_loc,torch.arange(n,device=positions.device),right=True)-1
            requests=b.req_pool_indices[row.clamp_min(0)].long()
        prev=b.req_to_token_pool.req_to_token[requests,(positions-1).clamp_min(0)].long()
        self.model.current_slots[:n].copy_(b.out_cache_loc)
        self.model.previous_slots[:n].copy_(prev)
        hidden=self.model(input_ids,positions,forward_batch)
        return self.logits_processor(input_ids,hidden,self.model.math.lm_head,forward_batch)

    def load_weights(self,weights):
        params=dict(self.model.math.named_parameters())|dict(self.model.math.named_buffers())
        loaded=set()
        for name,weight in weights:
            if name not in params:raise KeyError(name)
            assert params[name].shape==weight.shape,(name,params[name].shape,weight.shape)
            params[name].data.copy_(weight);loaded.add(name)
        assert loaded==set(params),f'Missing weights: {set(params)-loaded}'
        return loaded


EntryClass=AllieShipForCausalLM
