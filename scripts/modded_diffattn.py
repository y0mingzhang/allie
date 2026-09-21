"""Differential attention (model track, tier 2; Ye et al. 2024, "Differential Transformer").

Heads are taken in pairs (2j, 2j+1): their two softmax maps A_2j, A_2j+1 are subtracted,
(A_2j - lambda * A_2j+1) applied to both heads' values, the pair RMS-normalised as one group and
scaled by (1 - lambda_init). lambda = exp(lq1 . lk1) - exp(lq2 . lk2) + lambda_init per layer. Built
from four ordinary attention calls on half the heads, so any attention kernel (flex here) works
unchanged; score and value FLOPs are twice standard attention (modded_arch.attn_factor).
"""

import math

import torch
from torch import nn


def lambda_init(layer):
    return 0.8 - 0.6 * math.exp(-0.3 * layer)


class DiffLambda(nn.Module):
    def __init__(self, head_dim, layer):
        super().__init__()
        self.init = lambda_init(layer)
        self.q1, self.k1, self.q2, self.k2 = (
            nn.Parameter(torch.randn(head_dim) * 0.1) for _ in range(4)
        )
        for p in (self.q1, self.k1, self.q2, self.k2):
            p.label, p.lr_mul, p.wd_mul = "diff", 0.1, 0.0

    def forward(self):
        return torch.exp(self.q1 @ self.k1) - torch.exp(self.q2 @ self.k2) + self.init


def diff_attention(q, k, v, lam, init, attend):
    """q, k, v: (B, T, H, D) with H even; attend(q, k, v) -> (B, T, h, D) standard attention."""
    qa, qb, ka, kb, va, vb = (x[:, :, i::2] for x in (q, k, v) for i in (0, 1))
    lam = lam.type_as(q)
    ya = attend(qa, ka, va) - lam * attend(qb, kb, va)
    yb = attend(qa, ka, vb) - lam * attend(qb, kb, vb)
    y = torch.stack((ya, yb), 3)  # (B, T, H/2, 2, D): heads 2j, 2j+1 of pair j
    y = y * torch.rsqrt(y.float().square().mean((-2, -1), keepdim=True) + 1e-6).type_as(
        y
    )
    return (y * (1 - init)).flatten(2, 3)
