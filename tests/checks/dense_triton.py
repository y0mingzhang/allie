"""--dense-triton (model.moe_kernels.linear with DENSE_TRITON): bitwise F.linear, forward and both grads, on the
dense GEMMs it takes at width 2048 (QKV, O, the shared expert's up at Allie 2.1's split, the head), and F.linear
itself on the shapes it leaves (K != 2048, fewer than 2048 rows, FP32). CUDA; the claim holds where cuBLAS runs
these shapes as one sequential k16 chain (checked on sm_89).

    .venv/bin/python tests/checks/dense_triton.py
"""

import sys

import torch
import torch.nn.functional as F

from allie.model import moe_kernels

D = 2048
# (rows, N) at K 2048: QKV, O, the shared expert's up (2 x 1360 at Allie 2.1's split), the head (2350 -> 2432)
TAKEN = [(16384, 3 * D), (16384, D), (16384, 2 * 1360), (16384, 2432), (2048, D)]
LEFT = [(16384, 1360, D), (2047, D, D), (16384, 1536, 1536)]  # (rows, K, N)


def run(x, w, triton):
    moe_kernels.DENSE_TRITON = triton
    x, w = x.clone().requires_grad_(), w.clone().requires_grad_()
    y = moe_kernels.linear(x, w)
    dy = torch.randn(y.shape, device="cuda", generator=torch.Generator("cuda").manual_seed(2)).to(y.dtype)
    y.backward(dy)
    return y.detach(), x.grad, w.grad


def main():
    gen = torch.Generator("cuda").manual_seed(0)
    ok = True
    for t, k, n in [(t, D, n) for t, n in TAKEN] + LEFT:
        x = torch.randn(t, k, device="cuda", generator=gen).bfloat16()
        w = (torch.randn(n, k, device="cuda", generator=gen) * k**-0.5).bfloat16()
        ref = run(x, w, False)
        got = run(x, w, True)
        same = [torch.equal(a, b) for a, b in zip(ref, got)]
        plain = torch.equal(ref[0], F.linear(x, w))
        ok &= all(same) and plain
        print(f"T{t} K{k} N{n}: out / dx / dw bitwise {same}, off is F.linear {plain}")
    x = torch.randn(4096, D, device="cuda", generator=gen)
    w = torch.randn(D, D, device="cuda", generator=gen)
    moe_kernels.DENSE_TRITON = True
    fp32 = torch.equal(moe_kernels.linear(x, w), F.linear(x, w))
    ok &= fp32
    print(f"FP32 left to F.linear: {fp32}; {torch.cuda.get_device_name()}")
    print("PASS" if ok else "FAIL")
    sys.exit(not ok)


if __name__ == "__main__":
    main()
