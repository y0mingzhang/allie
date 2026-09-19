"""Pinned SGLang 0.5.9 adapter for compact, exact BF16-score transport.

The standard token-logprob API serializes thousands of Python tuples per node.
Attach binary scores after graph execution, before sampling transforms logits.
SGLang's existing customized_info route carries them without changing its files.
This hook only applies to AllieForCausalLM and must be retested on upgrades.
"""
import base64
import torch


def install_binary_scores():
    from sglang.srt.model_executor.model_runner import ModelRunner
    if getattr(ModelRunner.sample,'allie_binary_scores',False):return
    original=ModelRunner.sample

    def sample(self,logits_output,forward_batch,*args,**kwargs):
        if self.model.__class__.__name__=='AllieForCausalLM':
            # The checkpoint head returns BF16 values in [0,23]. FP16 represents
            # every such BF16 value exactly, unlike arbitrary FP32 model logits.
            scores=logits_output.next_token_logits.to(torch.float16).cpu().numpy()
            logits_output.customized_info={'allie_logits_fp16':[
                base64.b64encode(row.tobytes()).decode('ascii') for row in scores]}
        return original(self,logits_output,forward_batch,*args,**kwargs)

    sample.allie_binary_scores=True
    ModelRunner.sample=sample
