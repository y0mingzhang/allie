"""make_context's block-level flex masks vs the dense create_block_mask they replace: every BlockMask
tensor (values, dtype, shape, strides) must match, on real validation rows and edge cases, and on
--row-tokens rows longer than the windows whose games keep at most 1025 tokens (span).

    TORCH_COMPILE_DISABLE=1 .venv/bin/python scripts/test_game_blocks.py         # CPU, eager reference
    <runtime python> scripts/test_game_blocks.py --device cuda                   # GPU, compiled reference
"""

import os
import sys

import numpy as np
import torch
from torch.nn.attention.flex_attention import create_block_mask

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lm_data import Packed
from modded_medium import BOS, make_context

DATA = "/data/group_data/dei-group/yimingz3/allie/lichess_tokens_v2"
FIELDS = (
    "kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices",
    "q_num_blocks", "q_indices", "full_q_num_blocks", "full_q_indices",
)  # fmt: skip


def reference(inputs, window, compiled):
    """The pre-change make_context mask, verbatim."""
    flat = inputs.flatten()
    starts = flat == BOS
    starts[:: inputs.size(1)] = True
    docs = starts.to(torch.int32).cumsum(0)
    length = flat.numel()

    def allowed(b, h, q, k):
        return (
            (q < length)
            & (k < length)
            & (q >= k)
            & (q - k <= window)
            & (docs[q.clamp(max=length - 1)] == docs[k.clamp(max=length - 1)])
        )

    return create_block_mask(
        allowed, 1, None, length, length, device=flat.device, BLOCK_SIZE=128,
        _compile=compiled,
    )  # fmt: skip


def same(a, b, what):
    assert a.BLOCK_SIZE == b.BLOCK_SIZE and a.seq_lengths == b.seq_lengths, what
    for f in FIELDS:
        x, y = getattr(a, f), getattr(b, f)
        assert (x is None) == (y is None), (what, f)
        if x is not None:
            assert x.dtype == y.dtype and x.shape == y.shape, (
                what,
                f,
                x.dtype,
                y.dtype,
            )
            assert x.stride() == y.stride(), (what, f, x.stride(), y.stride())
            assert torch.equal(x, y), (what, f)


def check(rows, windows, device, what, span=None):
    x = torch.as_tensor(rows, device=device)
    ctx = make_context(x, *windows, board=False, span=span)
    compiled = device == "cuda"
    for mask, w in ((ctx.short_mask, windows[0]), (ctx.long_mask, windows[1])):
        same(mask, reference(x, w, compiled), f"{what} window {w}")
        assert mask.mask_mod(0, 0, torch.tensor(0), torch.tensor(0)).item()


def synthetic(rng, n, length):
    """Games of random lengths laid over n rows, plus boundaries forced onto block/row edges."""
    rows = rng.integers(0, BOS, size=(n, length))
    for r in range(n):
        pos = 0
        while pos < length:
            rows[r, pos] = BOS
            pos += int(rng.choice([1, 2, 5, 60, 127, 128, 129, 300, 700, 2000]))
    edges = [p for p in (127, 128, 255, 256, length - 1) if p < length]
    rows[0, edges] = BOS  # boundaries at block edges and the row's last token
    if n > 1:
        rows[1] = rng.integers(
            0, BOS, size=length
        )  # no game start: the row start splits
    if n > 2:
        rows[2] = BOS  # every token starts a game
    return rows


def main():
    device = (
        sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
    )
    rng = np.random.default_rng(0)
    val = Packed(DATA, "val")
    real = val.rows(rng.choice(int(val.ends[-1]), 17, replace=False))[:, :-1]
    train_windows = (11 * 128, 23 * 128)  # the fixed WSD windows, in 128-token blocks
    n = 0
    for rows in (
        16,
        1,
        4,
        17,
    ):  # micro 16 training, eval batches incl. a short last one
        check(real[:rows], train_windows, device, f"real {rows}x1024")
        n += 1
    for length in (
        1024,
        1000,
        300,
        128,
        1,
    ):  # 1000/300: partial last block, blocks span rows
        for count in (1, 3, 16):
            rows = synthetic(rng, count, length)
            for windows in (
                train_windows,
                (length - 1, 2944),
                (max(length - 2, 0), 64),
            ):
                check(rows, windows, device, f"synthetic {count}x{length} {windows}")
                n += 1
    for count in (1, 3):
        rows = synthetic(rng, count, 4096)
        rows[:, ::1025] = BOS  # games of at most 1025 tokens
        check(rows, train_windows, device, f"span 1025 {count}x4096", span=1025)
        n += 1
    print(
        f"{n} cases on {device}: every BlockMask tensor identical to create_block_mask"
    )


if __name__ == "__main__":
    main()
