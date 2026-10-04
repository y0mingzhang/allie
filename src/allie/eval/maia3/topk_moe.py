"""MoE.forward with fewer routed experts per token at inference (topk_score.py patches it in). A module of its
own: under runpy the launcher's globals sit in a swapped __main__ that torch.compile's guards cannot find."""

import torch
import torch.nn.functional as F

# the patched MoE module's expert counts and routed-expert kernel, set by topk_score.py
counts = routed = None


def forward(self, x):
    k, mode = self.eval_topk
    assert not self.training and getattr(self, "score", "sigmoid") == "sigmoid"
    assert not getattr(self, "log_gates", False) and getattr(self, "pos", None) is None
    shape, d = x.shape, x.shape[-1]
    h = x.reshape(-1, d)
    g = torch.promote_types(h.dtype, torch.float32)
    z = F.linear(h.to(g) - self.mu if self.center else h.to(g), self.router.to(g))
    s = torch.sigmoid(z)
    full = torch.topk(s + self.bias, self.topk, dim=-1).indices
    idx = full[:, :k]
    w = s.gather(1, idx)
    if mode == "trunc":
        w = w * (self.topk**0.5 / self.floored(s.gather(1, full).sum(-1, keepdim=True)))
    else:
        w = w * (k**0.5 / self.floored(w.sum(-1, keepdim=True)))
    flat = idx.flatten()
    order = flat.argsort(stable=True)
    count = counts(flat[order], self.experts)
    route = k, flat[order], order, count.cumsum(0), w.type_as(h), self.remat
    up, down = self.up.type_as(h), self.down.type_as(h)
    out = routed(h, up.transpose(1, 2), down, *route)
    if self.shared:
        shared = self.act(F.linear(h, self.shared_up.type_as(h)))
        out = out + F.linear(shared, self.shared_down.T.type_as(h))
    return out.view(shape)
