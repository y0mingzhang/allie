"""Final scaling sweep (user GO 2026-09-22 ~20:00 ET): isoFLOPs of S16 and dense at 1e17 / 3e17 / 1e18 on the final
bundle, one study per budget (sweep-BUDGET-RECIPE), plus the pre-launch check (sweep-check-RECIPE).
S16 = MoE 256 experts, top-16, shared expert 1/4 of the active MLP, routed experts (3/4 of it) / 16, every width a
multiple of 16 (moe_round 16), sigmoid scores + Kimi K3 quantile balancing, moe_seq 1e-3, router LR x0.1. Dense = the
ship recipe. Both: fp32 small masters, 16384-token rows (micro-batch 1 row, 524288 tokens per step), x0 on, clock-input
LR x5, RECIPE on the v4 pool (inv4 pin, history mode, 75B final_tokens, pool_frac = the run's tokens / 75B), seed 42,
FLOP-matched steps. Ladder a..h; grid and adaptive rule: moe-knobs/SWEEP-PLAN.md. Frozen from branch sweep (main 570f88b
+ data-recipe-3e17 45fbbc6); RECIPE filled by moe-knobs/sweep-plan/freeze.py."""

import modelexp as mx
from modelexp import MOE, variants, wave

RECIPE = ('c8s200f0v4', 'table:c8s200f0v4-fcfbf8858a28')
INV = "/home/yimingz3/src/allie/results/recipe10x/data-recipe-inv4"
LADDER = dict(
    a=(10, 448), b=(12, 512), c=(14, 640), d=(16, 768), e=(18, 896), f=(20, 1024), g=(22, 1152), h=(24, 1280)
)
# 1e17 / 3e17 carry their upper edge size from the start (e / g: cheap, and it takes an adaptive round off the critical path)
GRID = {"1e17": ("abcde", "abcde", "dei5"), "3e17": ("cdefg", "cdefg", "l40s4"), "1e18": ("defg", "efgh", "l40s4")}
mx.LANES["l40s4"] = (1, 4, "preempt", "gpu:L40S:4", None, 24, 200, 8)
# 1e17: five 1-GPU runs per 5 x A6000 dei job (dei_group_qos allows 10 jobs per user; one node's page cache per pack)
mx.LANES["dei5"] = (5, 5, "dei-group", "gpu:A6000:5", None, 40, 320, None)
FIX = dict(fp32_small_masters=True)
FAMILY = dict(
    s16=MOE(256, 16, moe_shared_frac=0.25, moe_round=16, moe_update="quantile", **FIX), dense=dict(arch=FIX)
)
DATA = dict(policy=RECIPE[1], history=True, final_tokens=75e9, tag=RECIPE[0], micro_batch=1)
DATA |= dict(extra_args=["--row-tokens", 16384])
arm = lambda fam, x, **kw: {f"{fam}-{LADDER[x][0]}x{LADDER[x][1]}": DATA | kw | FAMILY[fam] | dict(zip(("layers", "width"), LADDER[x]))}
PIN = dict(pin=f"{INV}/data-pin.json", history=f"{INV}/history-counts.json")
for b, (s, dn, lane) in GRID.items():
    arms = {k: v for x in s for k, v in arm("s16", x).items()} | {k: v for x in dn for k, v in arm("dense", x).items()}
    wave(f"sw{b}", f"sweep-{b}-{RECIPE[0]}", "sw", variants(b, arms), f"final scaling sweep {b}, recipe {RECIPE[0]}", lane, **PIN)
# pre-launch checks: the largest 1e18 size of each family in two legs, 128 and 256 steps (a resume between them; past MTP /
# the embed split), then scored; swchk3 on babel 4 x L40S, swchkh on orchard 4 x H100 (orun.sh with OLEGS=128,256; its own
# names: a run name never runs on both clusters)
check = arm("s16", "g", stop_after=[128, 256]) | arm("dense", "h", stop_after=[128, 256])
wave("swchk3", f"sweep-check3-{RECIPE[0]}", "swchk3", variants("1e18", check), "sweep pre-launch check", "l40s4", **PIN)
wave("swchkh", f"sweep-checkh-{RECIPE[0]}", "swchkh", variants("1e18", check), "sweep pre-launch check, H100", "l40s4", **PIN)
# S16 22x1152 does not fit 4 x L40S at 16K rows (check3 OOM at the start, 44.3 of 44.4 GB): the 3e17 / 1e18 S16 upper edge
# on L40S with --ckpt eager (each block recomputed in backward: same math, other kernels, ~30% slower), under its own prefix
ck = arm("s16", "g", extra_args=[*DATA["extra_args"], "--ckpt", "eager"])
for b in ("3e17", "1e18"):
    wave(f"sw{b}ck", f"sweep-{b}ck-{RECIPE[0]}", "swc", variants(b, ck), f"final scaling sweep {b}, S16 22x1152 eager ckpt", "l40s4", **PIN)
# seed 43 at the old fits' best 1e18 sizes, queued before the readout to take a ~14 h run off the critical chain (main, 2026-09-23
# 00:40): it measures run-to-run noise, it is not the new readout's best size
wave("sw1e18s", f"sweep-1e18s-{RECIPE[0]}", "sw", variants("1e18", arm("s16", "e") | arm("dense", "g"), (43,)), "final scaling sweep 1e18, seed 43", "l40s4", **PIN)
# seed 43 copy of the whole 1e17 grid on the dei A6000s after the seed-42 packs (main, 2026-09-23 00:50): run-to-run noise
# from 10 pairs; supersedes seeds at the two 1e17 best sizes
s, dn, lane = GRID["1e17"]
arms = {k: v for x in s for k, v in arm("s16", x).items()} | {k: v for x in dn for k, v in arm("dense", x).items()}
wave("sw1e17s", f"sweep-1e17s-{RECIPE[0]}", "sw", variants("1e17", arms, (43,)), "final scaling sweep 1e17, seed 43", lane, **PIN)
# adaptive edge sizes (user 2026-09-23 via main: every rung bracketed, extend until it is): z below a on the same x16 /
# aspect rule; 1e17 edges on 2 x A6000 (dei; 1-GPU runs of these sizes take ~12 h), one study per extension
LADDER |= dict(z=(8, 384))
mx.LANES["dei2"] = (1, 2, "dei-group", "gpu:A6000:2", None, 16, 128, None)
wave("sw1e17x", f"sweep-1e17x-{RECIPE[0]}", "sw", variants("1e17", arm("s16", "z") | arm("dense", "z")), "final scaling sweep 1e17, adaptive lower edge", "dei2", **PIN)
wave("sw3e17x", f"sweep-3e17x-{RECIPE[0]}", "sw", variants("3e17", arm("s16", "b") | arm("dense", "b")), "final scaling sweep 3e17, adaptive lower edge", "l40s4", **PIN)
wave("sw1e18x", f"sweep-1e18x-{RECIPE[0]}", "sw", variants("1e18", arm("s16", "c") | arm("dense", "d")), "final scaling sweep 1e18, adaptive lower edge", "l40s4", **PIN)
wave("sw3e17x2", f"sweep-3e17x2-{RECIPE[0]}", "sw", variants("3e17", arm("s16", "a")), "final scaling sweep 3e17, adaptive lower edge 2", "l40s4", **PIN)
wave("sw1e18x2", f"sweep-1e18x2-{RECIPE[0]}", "sw", variants("1e18", arm("s16", "b")), "final scaling sweep 1e18, adaptive lower edge 2", "l40s4", **PIN)
# big run (user 2026-09-23: prepared, NOT launched; the shape is discussed first): S16 on one 8-GPU node, the sweep
# bundle + the spike fix (input_lr_mul 1.25: spk-1e17-dense-12x512-s43-inlr125 golden 1.4062 vs 1.4089), --ckpt eager,
# checkpoint every 1024 steps (operational), D <= 75B (the Elo-ramp recipe's final_tokens). The budget key pins the
# FLOP-matched steps to D / 524288 at the run's own shape and arch.
def big(prefix, L, W, tokens, lane="general8", ckpt="eager", micro=1, tag=""):
    arch = mx.BASE["arch"] | FAMILY["s16"]["arch"]
    key = f"{L}x{W}d{tokens / 1e9:.0f}" + f"m{micro}" * (micro > 1) + tag
    mx.SIZES[key] = dict(layers=L, width=W, steps=round(tokens / 524288) * mx.per_token(L, W, arch) / mx.per_token(L, W))
    extra = [*DATA["extra_args"], "--ckpt", ckpt, "--checkpoint-every", 1024]
    spec = {f"s16-{L}x{W}": DATA | FAMILY["s16"] | dict(layers=L, width=W, input_lr=1.25, micro_batch=micro, extra_args=extra)}
    wave(f"{prefix}-{key}", f"{prefix}-{key}-{RECIPE[0]}", prefix, variants(key, spec), f"big run candidate S16 {L}x{W}, {tokens / 1e9:.0f}B tokens", lane, **PIN)


# throughput / memory calibration on 8 x L40S (profile.py: 50 steps on fake data), D irrelevant
for L, W in ((28, 1536), (30, 1664), (32, 1792), (36, 2048)):
    big("bigprof", L, W, 75e9)
# candidate configs (frozen, NOT launched). D is fixed at freeze time (the WSD schedule depends on it) from the
# central tok/s; re-freeze with the smoke test's measured tok/s before any launch.
big("bigrun", 25, 2048, 75e9)  # 8 x H100 (orchard, omake --gpus 8): 1.26B active, ~130 h at central tok/s
big("bigrun", 33, 2048, 60e9)  # 8 x H100 max: 1.68B active, memory tight
big("bigrun", 24, 1536, 50e9)  # 8 x L40S (general8): 0.69B active, ~150 h
big("bigrun", 22, 1408, 60e9)  # 8 x L40S: 0.53B active, ~145 h
# the big run (user GO 2026-09-24 via main): 24x1536d50 at micro-batch 4 rows (64K tokens per micro-step, one
# micro-step per step; global batch unchanged), 119K tok/s in parallel-tput/baseline; its soak leg is the run's first leg
big("bigrun", 24, 1536, 50e9, micro=4)
# its fallback launch study (main 2026-09-24): the same run on the perf/bundle-r5 trainer (9f8eb98: expert tiles, 512 MiB
# in-flight grad cap), frozen by freeze.py from that commit
big("bigrun", 24, 1536, 50e9, micro=4, tag="r5")
# and on perf/bundle-r5-ckpt (7de2ec6: r5 + the paced checkpoint writer and threaded row log; host RAM 194 GiB steady /
# 273 GiB peak, under general8's 400G)
big("bigrun", 24, 1536, 50e9, micro=4, tag="r5ck")
# the launch source (main 2026-09-24): perf/ship 5906c9a = r5-ckpt 7de2ec6 + selective recompute S1 (3ac5807: the routed
# combine is an opaque op a checkpoint's recompute skips)
big("bigrun", 24, 1536, 50e9, micro=4, tag="ship")
# the big run itself (user 2026-09-24 via main): the full 75B tokens the Elo-ramp recipe was solved for (pool_frac 1)
big("bigrun", 24, 1536, 75e9, micro=4, tag="ship")
# big-run fixes (main 2026-09-24 ~17:35 ET, after the d75 ship run failed its health bar: MoE layers 0-3 router starvation
# rising from step ~1225, layer 0 91/256; moe-knobs/bigrun-plateau.md): the same d75 ship run from scratch to step 2600,
# each wave one frozen study that differs from bigrun-24x1536d75m4ship only in the named plan fields
def bigfix(tag, arch, sched=None, stop=2600, L=24, W=1536, tokens=75e9, micro=4, more=(), study=""):
    base = mx.BASE["arch"] | FAMILY["s16"]["arch"]
    key = f"{L}x{W}d{tokens / 1e9:.0f}m{micro}ship{tag}"
    mx.SIZES[key] = dict(layers=L, width=W, steps=round(tokens / 524288) * mx.per_token(L, W, base) / mx.per_token(L, W))
    extra = [*DATA["extra_args"], "--ckpt", "eager", "--checkpoint-every", 1024, *more]
    s16 = FAMILY["s16"] | dict(arch=FAMILY["s16"]["arch"] | arch)
    run = DATA | s16 | dict(layers=L, width=W, input_lr=1.25, micro_batch=micro, extra_args=extra) | (dict(stop_after=[stop]) if stop else {})
    spec = {f"s16-{L}x{W}": run | (dict(sched=sched) if sched else {})}
    wave(f"bigfix-{key}{study}", f"bigfix-{key}{study}-{RECIPE[0]}", "bigfix", variants(key, spec), f"big-run fix {tag}: S16 {L}x{W} d75 to step {stop}", "general8", **PIN)


STEPS = 143051  # the d75 run's steps; the sweep's MTP-off / split steps (18x896: 302 / 307), not D-scaled
bigfix("f1", dict(moe_router_lr_mul=0.05))  # router Adam LR x0.5
bigfix("f2", dict(moe_router_lr_mul=0.05), sched=dict(mtp=302 / STEPS, split=307 / STEPS))  # + short MTP / tie phases
# escalation (main 2026-09-24 ~20:10 ET): F2 + the router-input centring switch (moe_router_center, EMA decay 0.9), frozen
# from main eb8c6a0 = 5906c9a + off-by-default header_feats, the signal-handler prewarm and the switch
bigfix("f3", dict(moe_router_lr_mul=0.05, moe_router_center=0.9), sched=dict(mtp=302 / STEPS, split=307 / STEPS))
# F4 (main 2026-09-25 ~02:40 ET, nanogpt-audit): F3 + arch adam_every (Adam and scalar optimizers every step at half lr,
# twice their lr^2 decay, square-rooted betas; F2's collapse spikes sat on the logged steps right after a router Adam
# update), frozen from main 39cb5ea = eb8c6a0 + header_lr_mul and the adam_every / wd_scale switches, in parallel with F3
bigfix("f4", dict(moe_router_lr_mul=0.05, moe_router_center=0.9, adam_every=True), sched=dict(mtp=302 / STEPS, split=307 / STEPS))
# F3 again under its own name for a fast race on general / preempt (main 2026-09-25 ~06:50 ET): identical plan fields
bigfix("f3b", dict(moe_router_lr_mul=0.05, moe_router_center=0.9), sched=dict(mtp=302 / STEPS, split=307 / STEPS))
# the relaunch (user GO 2026-09-25 ~12:40 ET): F4 + no multi-token prediction + decay to ~0 (nanogpt-audit L: mtp0 -1.1,
# d2z -1.9 mn), time-control tokens kept, full 75B tokens, no stop; frozen from main f9baa37; 2-day chunks on general
bigfix("v2", dict(moe_router_lr_mul=0.05, moe_router_center=0.9, adam_every=True), sched=dict(mtp=0, split=307 / STEPS, final_lr=0.004), stop=None)
# v2 resumed from its last.pt (step 59392) on main 7baccfe with arch moe_gate_floor 1e-12 (step 60081 went nonfinite
# in the backward of the layer-1 gate renormalisation, moe-knobs/bigrun-nan-60k.md); same run name, its own study, which
# the run's owner.json is handed to (the v2 manifest kept as owner-v2.json)
bigfix("v2", dict(moe_router_lr_mul=0.05, moe_router_center=0.9, adam_every=True, moe_gate_floor=1e-12), sched=dict(mtp=0, split=307 / STEPS, final_lr=0.004), stop=None, more=["--resume-new-source"], study="nf")
