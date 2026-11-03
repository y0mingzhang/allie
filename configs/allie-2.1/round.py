"""Allie 2.1 (user go 2026-10-12): the paused v3 run from step 0 with two recipe changes.

the paused v3 run = study bigrun-v3-24x2048d112ft2-c8s200f0v4 (round bigrun-v3/round.py v3 "f", takeover t2): S16 24x2048
(1.22B active), 213,623 steps of 524,288 tokens (112B), table c8s200f0v4d112 on the inv4 pin, seed 42, micro 4, the
ship flags, general 8 x L40S, 500 GB, 2-day legs. Changes:
  - peak LR x0.35 (--lr-scale), same schedule shape: warmup 2013 steps, linear decay to 0.004 of peak, split 307;
  - c4x as scale22's c4x runs (scale22/round.py P1["c4x"]): router-input centring in the first 4 MoE layers + XSA.
Readout: tail EMA (horizon 8% of the steps, every 64 steps; training-neutral) for scale22's tail-EMA blend.
Source: lab 57caf4bc (scale22's source: v3's 51c1131 + centre switch, XSA, tail EMA for moe_shard), fresh start.
"""

import modelexp as mx
from modelexp import MOE, variants, wave

RECIPE = ("c8s200f0v4", "table:c8s200f0v4d112-b8276fd781ac")
INV = "/home/yimingz3/src/allie/results/recipe10x/data-recipe-inv4"
PIN = dict(pin=f"{INV}/data-pin.json", history=f"{INV}/history-counts.json")
L, W, TOKENS = 24, 2048, 112e9
STEPS = round(TOKENS / 524288)  # 213623
S16 = MOE(256, 16, moe_shared_frac=0.25, moe_round=16, moe_update="quantile", fp32_small_masters=True)
D3 = dict(moe_router_lr_mul=0.05, moe_log_gates=True, moe_dense_first=3, moe_router_center=0.9)
C4X = dict(moe_router_center_first=4, xsa=True)
ARCH = S16["arch"] | D3 | C4X | dict(moe_shard=True)
SHIP = [
    "--smoe-tuned",
    "--shard-early",
    "--shard-prefetch",
    2,
    "--ckpt-frac",
    0.95,
    "--shard-hold",
    "--optimizers-reversed",
]
EXTRA = ["--row-tokens", 16384, "--ckpt", "eager", "--checkpoint-every", 1024, "--moe-remat", "--moe-chunks", 4]
EXTRA += ["--keep-checkpoints", 1, *SHIP, "--keep-model-every", round(STEPS / 10)]
TEMA = ["--tail-ema", 0.08, "--tail-ema-every", 64]
RUN = dict(
    policy=RECIPE[1], history=True, final_tokens=TOKENS, tag=RECIPE[0], micro_batch=4, input_lr=1.25,
    arch=ARCH, layers=L, width=W, lr=0.35, extra_args=[*EXTRA, *TEMA],
    sched=dict(mtp=0, split=307 / STEPS, final_lr=0.004, decay=2013 / STEPS, warmup=2013 / STEPS),
)  # fmt: skip
KEY = f"{L}x{W}d{TOKENS / 1e9:.0f}"
# FLOP-matched steps pinned to TOKENS / 524288 under the run's own arch (as bigrun-v3's v3())
full = mx.BASE["arch"] | ARCH
mx.SIZES[KEY] = dict(layers=L, width=W, steps=STEPS * mx.per_token(L, W, full) / mx.per_token(L, W))
mx.LANES["general8m5"] = (1, 8, "general", "gpu:L40S:8", None, 48, 500, 1)
wave(f"v4-{KEY}", f"bigrun-v4-{KEY}-c4x-lr0p35-{RECIPE[0]}", "v4", variants(KEY, {f"s16-{L}x{W}-c4x-lr0.35": RUN}),
     "Allie 2.1: the paused v3 run from step 0 with peak LR x0.35 and c4x (centring 4 MoE layers + XSA), tail-EMA readout",
     "general8m5", **PIN)  # fmt: skip
mx.WAVES[f"v4-{KEY}"]["sbatch"] = mx.WAVES[f"v4-{KEY}"]["sbatch"].replace("--time=12:00:00", "--time=2-00:00:00")
