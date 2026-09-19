"""Read hidden states through SGLang's supported capture mode, scoped per call."""
from contextlib import contextmanager
import numpy as np
import torch

@contextmanager
def capture(base,full=False):
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    original=base.runner.forward_extend;states=[]
    def wrapped(batch,*args,**kwargs):
        batch.capture_hidden_mode=CaptureHiddenMode.FULL if full else CaptureHiddenMode.LAST
        output,extra=original(batch,*args,**kwargs)
        hidden=output.hidden_states
        assert hidden is not None and hidden.ndim==2 and hidden.shape[1]==512
        expected=batch.extend_num_tokens if full else batch.batch_size
        assert hidden.shape[0]==expected,(hidden.shape,expected)
        states.append(hidden.to(torch.float16).cpu().numpy())
        return output,extra
    base.runner.forward_extend=wrapped
    try:yield states
    finally:base.runner.forward_extend=original

def last(base,prefixes):
    keys=[tuple(p) for p in prefixes];unique=list(dict.fromkeys(keys))
    with capture(base) as states:logits=base(prefixes)
    hidden=np.concatenate(states)
    assert len(hidden)==len(unique)
    lookup={k:i for i,k in enumerate(unique)}
    return logits,hidden[[lookup[k] for k in keys]]

def full(base,prefixes):
    # One fresh prefill batch: no cached parent/chunk ambiguity for token alignment.
    assert base.next_row==0 and sum(map(len,prefixes))<=4096
    assert len(set(map(tuple,prefixes)))==len(prefixes)
    with capture(base,True) as states:logits=base(prefixes)
    hidden=np.concatenate(states)
    assert len(hidden)==sum(map(len,prefixes))
    return logits,hidden
