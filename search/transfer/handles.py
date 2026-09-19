"""Node handles with query-owned board state and causal imagined clocks."""
import numpy as np
import torch
from search.engine.handles import HandleOracle
from .context import advance_boards,advance_clocks,predicted_seconds,root_other_previous


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
        for i,(p,feat,row) in enumerate(zip(prefixes,features,base.last_root_rows)):
            self.rows[i]=row;self.lengths[i]=len(p)
            self.boards[i]=base.root_metadata[row][1][-1];self.feats[i]=feat[-1]
            self.inc[i]=p[2]-10 if 10<=p[2]<191 else -1
            self.other_previous[i]=root_other_previous(p,feat,self.inc[i])
        self.elapsed[:n]=predicted_seconds(self.root_logits) if clock_rule=='predicted' else 0

    def _forward(self,handles):
        ids,parents,tokens,lengths=handles.T
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
