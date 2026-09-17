"""FP32 masters and Adam state only for native Qwen's small vector group."""
import copy
import torch


class FP32VectorAdam:
    def __init__(self,native_optimizer,*,vector_decay=0.):
        assert native_optimizer._muon is not None and native_optimizer._adam is not None
        assert 0<=vector_decay<1
        self.native=native_optimizer;self.vector_decay=float(vector_decay)
        self.parameters=list(native_optimizer._adam_params)
        assert all(p.ndim!=2 and p.dtype==torch.bfloat16 for p in self.parameters)
        self.masters=[torch.nn.Parameter(p.detach().float()) for p in self.parameters]
        index={id(p):m for p,m in zip(self.parameters,self.masters)}
        groups=[]
        for group in native_optimizer._adam.param_groups:
            settings={k:copy.deepcopy(v) for k,v in group.items() if k!='params'}
            settings['params']=[index[id(p)] for p in group['params']]
            settings['weight_decay']=self.vector_decay;groups.append(settings)
        self.adam=torch.optim.AdamW(groups)

    @property
    def param_groups(self):return self.native._muon.param_groups+self.adam.param_groups

    def zero_grad(self,set_to_none=True):
        self.native.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self,closure=None):
        assert closure is None
        self.native._muon.step()
        for p,m in zip(self.parameters,self.masters):m.grad=None if p.grad is None else p.grad.float()
        self.adam.step()
        for p,m in zip(self.parameters,self.masters):p.copy_(m)

    def state_dict(self):
        return dict(format='native-fp32-vector-adam-v1',vector_decay=self.vector_decay,
            muon=self.native._muon.state_dict(),adam=self.adam.state_dict(),
            masters=[m.detach() for m in self.masters],shapes=[list(p.shape) for p in self.parameters])

    @torch.no_grad()
    def load_state_dict(self,state):
        if 'format' not in state:
            assert set(state)=={'muon','adam'}
            self.native._muon.load_state_dict(state['muon'])
            for p,m in zip(self.parameters,self.masters):m.copy_(p)
            self.adam.load_state_dict(state['adam'])
            for group in self.adam.param_groups:group['weight_decay']=self.vector_decay
        else:
            assert state['format']=='native-fp32-vector-adam-v1' and state['vector_decay']==self.vector_decay
            assert state['shapes']==[list(p.shape) for p in self.parameters]
            assert len(state['masters'])==len(self.masters)
            self.native._muon.load_state_dict(state['muon'])
            for p,m,value in zip(self.parameters,self.masters,state['masters']):
                assert value.dtype==torch.float32 and value.shape==p.shape
                m.copy_(value)
                assert torch.equal(m.to(p.dtype),p),'Vector master and forward checkpoint disagree'
            self.adam.load_state_dict(state['adam'])
            assert all(g['weight_decay']==self.vector_decay for g in self.adam.param_groups)
        for value in self.adam.state.values():
            for key in ('exp_avg','exp_avg_sq'):
                if key in value:assert value[key].dtype==torch.float32
