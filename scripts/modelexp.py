"""Model-track waves: one model/recipe change per run on the pinned data baseline (B_2).

Runs are matched on trainer FLOPs (modded_train.useful_flops): a variant architecture trains for the
steps that give the same FLOPs as the budget's baseline shape. Fixed-step schedule knobs are fractions
of training, taken from the 3e16 baseline's absolute values, so the baseline reproduces the data track
exactly at 3e16 and scales with the run length elsewhere.

plan WAVE / submit WAVE / task WAVE / status WAVE / table WAVE [WAVE...]
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataexp as dx
from modded_arch import attn_factor, extra_flops

ROOT = dx.ROOT
# the pinned data baseline this branch screens on: B_3 (B_2 + OTB x4; main, 2026-09-19 ~10:00)
B2 = ROOT / "results/recipe10x/data-v1-B3.json"
B2_META = (
    "source_study",
    "source_run",
    "final_tokens",
    "history_counts",
    "history_counts_sha256",
    "chessmix_sha256",
    "sha256",
)
# B_2 input keys -> modded_train flags (True: bare flag; other truthy values: flag + value)
DATA_FLAGS = dict(
    clock="--clock",
    elo="--elo",
    feats="--clock-feats",
    input_lr="--input-lr-mul",
    aux_time="--aux-time",
    aux_wdl="--aux-wdl",
)
# 3e16 baseline (8x512, 2274 steps): absolute schedule steps as fractions of training
FRAC = dict(warmup=32 / 2274, mtp=64 / 2274, split=65 / 2274, decay=32 / 2274)
# mean in-game position per token, fitted to r2-3e16-control's logged FLOPs (old 16-layer count)
POS = 45.7
# results/recipe10x/gpu-allocation.md fast types as sinfo names them (4-GPU A100 nodes say A100_80G)
FAST = "RTX_PRO_6000|H200|H100|A100_80GB|A100_80G|L40S|6000Ada"
# lane -> (runs per task, GPUs per task, sbatch header, array throttle = model share of the pool)
LANES = dict(
    dei=(
        4,
        4,
        """#SBATCH --account=dippolit
#SBATCH --partition=dei-group
#SBATCH --qos=dei_group_qos
#SBATCH --gres=gpu:A6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00""",
        None,
    ),
    **{
        lane: (
            1,
            1,
            f"""#SBATCH --account=dippolit
#SBATCH --partition={part}
#SBATCH --qos={qos}
#SBATCH --gres=gpu:1
#SBATCH --constraint={types}
#SBATCH --exclude=babel-q9-32,babel-x9-32
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00""",
            share,
        )
        for lane, part, qos, types, share in (
            ("preempt", "preempt", "preempt_qos", FAST, 8),
            ("general", "general", "normal", FAST, 4),
            # identity pilots: same GPU type as the reference runs (determinism is per type)
            ("l40s", "preempt", "preempt_qos", "L40S", 2),
            # single A6000s: dei-group first (user), preempt overflow
            ("dei1", "dei-group", "dei_group_qos", "A6000", 8),
            ("a6000p", "preempt", "preempt_qos", "A6000", 14),
        )
    },
    # one run on 4 fast preempt GPUs (the ladder's second seed)
    preempt4=(
        1,
        4,
        f"""#SBATCH --account=dippolit
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:4
#SBATCH --constraint={FAST}
#SBATCH --exclude=babel-q9-32,babel-x9-32
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=12:00:00""",
        1,
    ),
    # one run on a whole 8 x L40S node (the 3e17 ship run; user, ~11:30)
    general8=(
        1,
        8,
        """#SBATCH --account=dippolit
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:8
#SBATCH --exclude=babel-q9-32,babel-x9-32
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=12:00:00""",
        1,
    ),
    # one run on the whole 8 x A6000 dei node (moe-v1 round 3's 3e17 gap test)
    dei8=(
        1,
        8,
        """#SBATCH --account=dippolit
#SBATCH --partition=dei-group
#SBATCH --qos=dei_group_qos
#SBATCH --gres=gpu:A6000:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=12:00:00""",
        1,
    ),
    # one run on 4 L40S (the 3e17 ladder, as the data track's round6)
    general4=(
        1,
        4,
        """#SBATCH --account=dippolit
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:4
#SBATCH --exclude=babel-q9-32,babel-x9-32
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=12:00:00""",
        1,
    ),
)


def baseline():
    """Pinned data config (B_2 once written by the data track; provisional control + clock + aux)."""
    if B2.exists():  # scripts/write_b2.py; provenance kept under "meta"
        b2 = json.loads(B2.read_text())
        body = {k: v for k, v in b2.items() if k != "sha256"}
        want = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        assert b2.get("sha256") == want, "B_2 was edited after write_b2.py hashed it"
        return b2 | dict(meta={k: b2.pop(k) for k in B2_META if k in b2})
    return dict(
        policy="control",
        clock=True,
        aux_time=0.2,
        aux_wdl=0.2,
        pool_frac={b: dx.pool_frac(b) for b in ("3e16", "1e17")},
        history=True,
        stores=dx.STORES,
        months=dx.R2P_MONTHS,
        provisional=True,
    )


def per_token(layers, width, head_dim=64, arch=None):
    """Forward trainer FLOPs per token: modded_train.useful_flops for this depth's layer pattern,
    with both attention windows covering whole rows (every layer attends to POS earlier positions),
    plus the arch's extra branches (modded_arch.extra_flops)."""
    gates = layers + 2 * min(5, layers // 2)
    return (
        24 * layers * width**2
        + 2 * width * 2432
        + 2 * (gates * (width // head_dim) * 16 + 64)
        + 4 * width * layers * POS * attn_factor(arch or {})
        + extra_flops(arch or {}, width, layers)
    )


def shape(r):
    """Shapes are relative to the budget's base (depth x, width_mul x, width in head_dim steps), so a
    variant means the same aspect-ratio change at every budget."""
    base = dx.SIZES[r["budget"]]
    layers = round(base["layers"] * r.get("depth", 1))
    width = round(base["width"] * r.get("width_mul", 1) / 64) * 64
    steps = round(
        base["steps"]
        * per_token(base["layers"], base["width"])
        / per_token(layers, width, arch=r.get("arch"))
    )
    return layers, width, steps


def schedule(r, steps):
    """Absolute schedule for this run length; r['sched'] overrides fractions, split=None never splits."""
    if r.get("abs_sched"):  # the control model's absolute steps at every budget
        return dx.SCHEDULE, 32
    f = FRAC | r.get("sched", {})
    warmup = max(1, round(f["warmup"] * steps))
    mtp = 0 if not r.get("mtp", True) else max(1, round(f["mtp"] * steps))
    split = steps + 1 if f["split"] is None else max(mtp, round(f["split"] * steps))
    s = dx.SCHEDULE | dict(warmup_steps=warmup, mtp_steps=mtp, split_step=split | 1)
    if r.get("momentum_warmup") is False:  # constant Muon momentum 0.95
        s |= dict(momentum_warmup=False)
    return s, max(warmup, round(f["decay"] * steps))


def variants(budget, base, specs, seeds=(42,)):
    """Runs: baseline config x model spec (label -> overrides) x seeds. The baseline's arch (B_4 on)
    is merged into every run, a spec's arch overriding it key by key. pool_frac is set per run in
    plan() from its own tokens, so FLOP-matched shapes keep the final run's repetition."""
    data = {
        k: v for k, v in base.items() if k not in ("pool_frac", "provisional", "meta")
    }
    unknown = (
        set(data) - {"policy", "stores", "months", "history", "arch"} - set(DATA_FLAGS)
    )
    assert not unknown, f"baseline keys modelexp does not forward: {unknown}"
    pf = base["pool_frac"].get(budget) or dx.pool_frac(budget)
    tag = f"pf{round(pf * 1000):03d}" + "h" * bool(data.get("history"))
    arch = lambda spec: data.get("arch", {}) | spec.get("arch", {})
    return [
        data
        | spec
        | ({"arch": arch(spec)} if arch(spec) else {})
        | dict(v=label, budget=budget, seed=s, tag=tag)
        for label, spec in specs.items()
        for s in seeds
    ]


def merge(*specs):
    """Combine model specs; schedule fractions merge key by key."""
    out = {}
    for s in specs:
        out |= {k: v for k, v in s.items() if k != "sched"}
        out["sched"] = out.get("sched", {}) | s.get("sched", {})
    return {k: v for k, v in out.items() if v != {}}


def stack(budget, base, parts, seed=43):
    """Leave-one-out stacking wave (user): B_2 baseline, the full stack of passers, and the stack
    without each part, all with a fresh seed. parts: label -> spec, one per axis."""
    specs = {"base": {}, "stack": merge(*parts.values())} | {
        f"stack-no-{k}": merge(*(v for j, v in parts.items() if j != k)) for k in parts
    }
    return variants(budget, base, specs, (seed,))


def confirm(base, parts, loo=(), seed=42):
    """1e17 scale check of a kept stack: B_2 baseline (scale-free schedule), the control model
    (absolute 32/64/65/32-step schedule; identical to base at 3e16), the stack, and leave-one-out
    runs for the parts in loo (large or scale-sensitive 3e16 gains)."""
    specs = {
        "base": {},
        "control": dict(abs_sched=True),
        "stack": merge(*parts.values()),
    } | {f"stack-no-{k}": merge(*(v for j, v in parts.items() if j != k)) for k in loo}
    return variants("1e17", base, specs, (seed,))


SCREEN1 = {
    # depth >= 8 (modded_medium.configure); 10x448 and 12x384 ~ equal params, 8x576 bigger model on fewer tokens
    "d1.25w0.875": dict(depth=1.25, width_mul=0.875),  # 10x448 at 3e16
    "d1.5w0.75": dict(depth=1.5, width_mul=0.75),  # 12x384
    "w1.125": dict(width_mul=1.125),  # 8x576
    "split50": dict(sched=dict(split=0.5)),
    "nosplit": dict(sched=dict(split=None)),
    "nomtp": dict(mtp=False),
    "nove": dict(value_embeds=False),
    "noskip": dict(skips=False),
    "nosmear": dict(smear=False),
    "lr0.5": dict(lr=0.5),
    "lr2": dict(lr=2.0),
    "wd2": dict(wd=2.0),
    "wsd80": dict(sched=dict(decay=0.8)),
    # Codex (inference work): BF16 rotary tables indexed by packed-batch position
    "docrope": dict(doc_rope=True),
    "ropefp32": dict(rope_fp32=True),
    # quirk sweep A (user, 2026-09-18 ~23:55): inherited modded-nanogpt choices, one each
    "noembed2": dict(arch=dict(embed2=False)),
    "nox0": dict(arch=dict(x0=False)),
    "nosoftcap": dict(arch=dict(softcap=False)),
    "nogates": dict(arch=dict(gates=False)),
    "nokeyoffset": dict(arch=dict(key_offset=False)),
    # full RoPE has no stationary dims for the key offset to shift (reviewer, 2026-09-19)
    "fullropenokeyoffset": dict(arch=dict(full_rope=True, key_offset=False)),
    "gelu": dict(arch=dict(mlp="gelu")),
    "swiglu": dict(arch=dict(mlp="swiglu")),
    "plaininit": dict(arch=dict(plain_init=True)),
    # Adam matrices at peak lr 1e-3 / 3e-3 (base lr x the schedule's 4.0 plateau), FP32 moments and
    # DistAdam's decay: cautious and lr^2-scheduled, calibrated to 0.1 * lr at the peak only
    "adamw1e-3": dict(arch=dict(matrix_adam=2.5e-4, matrix_wd=0.1 / 1e-3)),
    "adamw3e-3": dict(arch=dict(matrix_adam=7.5e-4, matrix_wd=0.1 / 3e-3)),
    "adamevery": dict(arch=dict(adam_every=True)),
    "uniformmults": dict(arch=dict(uniform_mults=True)),
    "fp32embed": dict(arch=dict(fp32_embed=True)),
    "untieve": dict(arch=dict(untie_ve=True)),
    "noqknorm": dict(arch=dict(qk_norm=False)),
    "plainwd": dict(arch=dict(cautious_wd=False)),
    # plain decay hits every entry, cautious about half: 0.5x plain ~ 1x cautious in effective decay
    # (base / wd2 / plainwd / plainwd-half = mask x strength 2x2; main, 2026-09-19)
    "plainwd-half": dict(arch=dict(cautious_wd=False), wd=0.5),
    "plainmuon": dict(arch=dict(normuon=False)),
    "nomomwarmup": dict(momentum_warmup=False),
    # board input (B6 / B7): Codex's encoder and CNN, zero-init branch
    "boarddirect": dict(arch=dict(board="direct")),
    "boardcnn": dict(arch=dict(board="conv")),
}

WAVES = {}


def wave(
    key,
    study,
    prefix,
    runs,
    purpose,
    lane="dei",
    group=None,
    pool_sources=None,
    identity=None,
):
    """group: waves split across lanes share one set of controls in table(). pool_sources: the
    study whose frozen sources pooled controls must match, for later tiers whose own sources differ
    by new modules; identity: the pilot study whose identity.json (see identity()) proves this
    code's default path bit-identical to those controls; plan() binds it into the study."""
    pack, gpus, sbatch, throttle = LANES[lane]
    WAVES[key] = dict(
        study=study,
        prefix=prefix,
        runs=runs,
        purpose=purpose,
        pack=pack,
        gpus=gpus,
        sbatch=sbatch,
        throttle=throttle,
        group=group or key,
        pool_sources=pool_sources,
        identity=identity,
    )


b = baseline()
B2_HASH = (  # full sha256: verified in baseline(); the provisional config hashes itself
    b.get("meta", {}).get("sha256")
    or hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest()
)


def history_file():
    """The history counts B_2 pins (checked against its hash), else the data track's default."""
    if "meta" not in b:
        return dx.HISTORY
    p, meta = Path(b["meta"]["history_counts"]), b["meta"]
    assert (
        hashlib.sha256(p.read_bytes()).hexdigest() == meta["history_counts_sha256"]
    ), p
    assert meta["final_tokens"] == dx.FINAL_TOKENS, meta["final_tokens"]
    return p


wave(
    "smoke",
    "model-v1-smoke",
    "msmk",
    [
        r | dict(stop_after=[50, 90])  # crosses the step-65 split; second leg resumes
        for r in variants(
            "3e16",
            b,
            {"base": {}, "frozen": dict(src="data-v1-round2p")}
            | {
                k: SCREEN1[k]
                for k in ("d1.25w0.875", "d1.5w0.75", "w1.125", "nosplit", "nomtp")
            }
            | {k: SCREEN1[k] for k in ("nove", "noskip", "nosmear", "wd2")},
        )
    ],
    "1-GPU-each smoke of every screen-1 variant (40 steps, golden eval) on the provisional baseline; "
    "base vs frozen round2p source checks default-off flags are training-identical",
)
wave(
    "smoke3",
    "model-v1-smoke3",
    "msmk3",
    [
        r | dict(stop_after=[50, 90])
        for r in variants(
            "3e16", b, {"base": {}} | {k: SCREEN1[k] for k in ("d1.5w0.75", "docrope")}
        )
    ],
    "smoke on the 87effcb sources (child-process sampler prefetch): base must match "
    "model-v1-smoke's base step for step; a shape and a rotary variant run end to end",
)
TIER1_NEW = [
    k for k in SCREEN1 if "arch" in SCREEN1[k] or "momentum_warmup" in SCREEN1[k]
]
wave(
    "smoke4",
    "model-v1-smoke4",
    "msmk4",
    [
        r | dict(stop_after=[50, 90])
        for r in variants(
            "3e16",
            b,
            {"base": {}} | {k: SCREEN1[k] for k in ["d1.5w0.75", *TIER1_NEW]},
        )
    ],
    "tier-1 smoke on 74999ad + the model-track switches: base must match model-v1-smoke's base "
    "step for step (default-off identity, sampler_wait logged); every new variant runs 50 -> "
    "resume -> 90 steps and scores; d1.5w0.75 rerun after the run-name fix",
    "preempt",
)
wave(
    "smoke5",
    "model-v1-smoke5",
    "msmk5",
    [
        r | dict(stop_after=[50, 90])
        for r in variants(
            "3e16",
            b,
            {"base": {}} | {k: SCREEN1[k] for k in ["d1.5w0.75", *TIER1_NEW]},
        )
    ],
    "smoke4 after the per-variant review fixes (boarddirect / 8 and lr 0.1, boardcnn one-hot meta, "
    "AdamW wd and FP32 moments, nogates 0.5, fullrope without key offset, host-side board encoding, "
    "DistAdam hook disarm): base must still match model-v1-smoke's base step for step",
    "general",
)
# tier 1 (36 runs): longest / board runs on general (not preempted), schedule-sensitive and optimizer
# runs on fast preempt, cheap flags packed on dei A6000. Controls (user, 2026-09-19 ~00:25): the data
# track's round-3 B_2 x 3 (one seed per lane, same commit, recipe, budget, pool, schedule), pooled in
# table() only if their executed args, source and evaluator hashes and B_2 match
# dei is unavailable (babel-t9-24 draining, babel-s9-24's GPUs mostly held by others; main, ~00:45):
# 12 runs on general (4 at a time, 3 cycles), 25 on preempt (8 at a time, 4 cycles)
S1 = dict(
    general=("d1.5w0.75", "d1.25w0.875", "w1.125", "boarddirect", "boardcnn", "lr0.5")
    + ("lr2", "swiglu", "adamw1e-3", "adamw3e-3", "untieve", "fp32embed"),
    preempt=("docrope", "ropefp32", "wd2", "wsd80", "noqknorm", "plainwd", "plainmuon")
    + ("plainwd-half",)
    + ("nox0", "nomomwarmup", "noembed2", "nosoftcap", "split50", "nosplit", "nomtp")
    + ("nove", "noskip", "nosmear", "nogates", "nokeyoffset", "fullropenokeyoffset")
    + ("gelu",)
    + ("plaininit", "adamevery", "uniformmults"),
)
assert sorted(k for ks in S1.values() for k in ks) == sorted(SCREEN1)
for lane, keys in S1.items():
    wave(
        f"screen1{lane[0]}",
        f"model-v1-screen1{lane[0]}",
        f"m1{lane[0]}",
        variants("3e16", b, {k: SCREEN1[k] for k in keys}),
        "model screen 1 at 3e16 on pinned B_2, trainer-FLOP matched: shape, embedding split, "
        "MTP, speedrun components, LR, weight decay, schedule, rotary",
        lane,
        "screen1",
    )
wave(
    "smoke2",
    "model-v1-smoke2",
    "msmk2",
    [
        r | dict(stop_after=[50, 90])
        for r in variants(
            "3e16", b, {"base": {}} | {k: SCREEN1[k] for k in ("docrope", "ropefp32")}
        )
    ],
    "smoke of the rotary variants (document-relative positions, FP32 tables); base must match "
    "model-v1-smoke's base step for step (default-off identity)",
)

# tier 2: DeepSeek MoE at the dense MLP's active FLOPs (shared 2d + k experts of 2d / k, dense first
# layer) and differential attention
SCREEN2 = {
    "moe32k4": dict(arch=dict(moe=[32, 4])),
    "moe64k8": dict(arch=dict(moe=[64, 8])),
    "diffattn": dict(arch=dict(diff_attn=True)),
    # aux-head weights (user via main, 2026-09-19 ~03:30: an objective question, owned here; B_2 has
    # think-time / W-D-L at 0.2 / 0.2, which cost 0.95x / 0.96x CM at 1e17; heads stay on unless
    # CM < 0.9x, so the question is which weight); noaux is the cost reference
    "aux0.1": dict(aux_time=0.1, aux_wdl=0.1),
    "aux0.05": dict(aux_time=0.05, aux_wdl=0.05),
    "noaux": dict(aux_time=0.0, aux_wdl=0.0),
}
wave(
    "pilot2",
    "model-v1-pilot2",
    "mpil2",
    [
        r | dict(stop_after=[50, 300])
        for r in variants(
            "3e16",
            b,
            {"base": {}} | {k: SCREEN2[k] for k in ("moe32k4", "moe64k8", "diffattn")},
        )
    ],
    "tier-2 sanity pilot on L40S: loss falls, no NaN, tokens/s vs base, MoE load not collapsed, "
    "resume; base must match data-v1-round3g's B_2 s42 (L40S, same B_2) at common steps",
    "l40s",
)
# tier 2b: MoE with routing tuned for 2274-step runs (pilot2's defaults collapsed): DeepSeek's router
# init 0.006, router lr ~DeepSeek's (lr_mul 0.01), bias speed 1e-2, and a sequence-wise loss 1e-3
# proportional bias updates (no +-gamma limit cycle once balanced), decayed to 0 over the last 20%
# of training (main's reviewer)
MOE_V2 = dict(moe_init=0.006, moe_router_lr_mul=0.01, moe_gamma=1e-2, moe_update="prop")
SCREEN2B = {
    "moe32k4b": dict(arch=dict(moe=[32, 4]) | MOE_V2),
    "moe32k4bs": dict(arch=dict(moe=[32, 4], moe_seq=1e-3) | MOE_V2),
    "moe64k8bs": dict(arch=dict(moe=[64, 8], moe_seq=1e-3) | MOE_V2),
}
wave(
    "pilot2b",
    "model-v1-pilot2b",
    "mpil2b",
    [
        r | dict(stop_after=[50, 300])
        for r in variants("3e16", b, {"base": {}} | SCREEN2B)
    ],
    "tier-2b MoE routing pilot on L40S: drops < 2%, starved ~0 and max / mean <= ~2 after step 100; "
    "base must match data-v1-round3g's B_2 s42 through step 300 (identity proof)",
    "l40s",
)
# tier 2c: the prop rule oscillated at gamma 1e-2 (bias range 0.17 vs a k / k+1 affinity margin of
# ~0.006, drops 0.1-6% on alternating logged steps); gamma <= ~1/3 of the margin per step. Same code as
# pilot2b (caf0c23), so pilot2b's identity proof binds and pilot2b's base is the reference
# with the sequence-wise loss: pilot2b's no-seq v2 arm generalised far worse off training batches
SCREEN2C = {
    f"moe{e}k{k}sg{tag}": dict(
        arch=dict(moe=[e, k], moe_seq=1e-3) | MOE_V2 | dict(moe_gamma=g)
    )
    for e, k in ((32, 4), (64, 8))
    for tag, g in (("2e-3", 2e-3), ("1e-3", 1e-3))
}
wave(
    "pilot2c",
    "model-v1-pilot2c",
    "mpil2c",
    [r | dict(stop_after=[50, 300]) for r in variants("3e16", b, SCREEN2C)],
    "tier-2c MoE bias-speed pilot on L40S (pilot2_gate vs pilot2b's base)",
    "l40s",
)
# main (~06:45): moe64k8bs as is (drops 3.6%, already priced into golden) into the 3e16 screen, with
# executed FLOPs and throughput reported beside CM; pilot2c's better gamma gets its own run later
wave(
    "screen2b",
    "model-v1-screen2b",
    "m2b",
    variants("3e16", b, {"moe64k8bs": SCREEN2B["moe64k8bs"]}),
    "model screen 1, tier 2b at 3e16 on pinned B_2, trainer-FLOP matched (nominal): MoE 64 x top-8 v2",
    "preempt",
    pool_sources="model-v1-screen1g",
    identity="model-v1-pilot2b",
)
# aux heads on stop-gradient features (main, ~08:00): the heads cost ~0.009 golden at any weight
# 0.05-0.2 through the aux gradient on the trunk (CPU: no code-path change); aux_detach removes that
# gradient. Pilot: default-path identity for the new core + the detached arm's compile / sanity
AUXSG = {"auxsg": dict(arch=dict(aux_detach=True))}  # at B_2's 0.2 / 0.2
wave(
    "pilot2d",
    "model-v1-pilot2d",
    "mpil2d",
    [
        r | dict(stop_after=[50, 300])
        for r in variants(
            "3e16",
            b,
            # same switch (so the same FLOP-matched steps and schedule), heads unweighted
            {"base": {}, "auxsg0": AUXSG["auxsg"] | dict(aux_time=0.0, aux_wdl=0.0)}
            | AUXSG,
        )
    ],
    "aux_detach pilot on L40S through 300 (past the split and a resume): base identity vs data-v1-"
    "round3g's B_2 s42; auxsg's move train_ce must equal auxsg0's (same switch, aux weights 0) step for "
    "step: detached heads leave the trunk's training unchanged (same GEMMs in BF16)",
    "l40s",
)
wave(
    "screen2d",
    "model-v1-screen2d",
    "m2d",
    variants("3e16", b, AUXSG, seeds=(42, 43)),
    "aux heads on stop-gradient features at 3e16 (0.2 / 0.2), s42 / s43; golden plus time_ce / wdl_ce",
    "preempt",
    pool_sources="model-v1-screen1g",
    identity="model-v1-pilot2d",
)
wave(
    "screen2",
    "model-v1-screen2",
    "m2",
    # MoE left out: its routing collapsed in pilot2 (ledger 04:20); it returns as tier 2b
    variants("3e16", b, {k: v for k, v in SCREEN2.items() if not k.startswith("moe")}),
    "model screen 1, tier 2 at 3e16 on pinned B_2, trainer-FLOP matched: diff attention, "
    "aux-head weights 0.1 / 0.05 / 0",
    "preempt",
    pool_sources="model-v1-screen1g",
    identity="model-v1-pilot2",
)


# B_4 = B_3 + boardcnn (user, ~10:00: confirmed model-track wins join the baseline). (1) the 1e17
# check against B_3's 1e17 pair (control's absolute schedule, FLOP-matched), s42 / s43, on general;
# (2) B_4 x 4 seeds at 3e16 as the next screens' pooled controls, on the faster sampler (7e8d2a2)
BOARD = dict(arch=dict(board="conv"))
wave(
    "b4check",
    "model-v1-b4check",
    "mb4c",
    variants("1e17", b, {"b4": BOARD | dict(abs_sched=True)}, seeds=(42, 43)),
    "B_4 (B_3 + boardcnn) at 1e17, control schedule, s42 / s43, vs B_3's 1e17 pair",
    "general",
)
wave(
    "b4ctl",
    "model-v1-b4ctl",
    "mb4",
    variants("3e16", b, {"b4": BOARD}, seeds=(42, 43, 44, 45)),
    "B_4 x 4 seeds at 3e16: pooled controls for the next screens on B_4",
    "preempt",
)

# 3e17 attribution ladder (user via main, ~11:00): the model recipe (boardcnn + swiglu + no key offset)
# on the data recipe B_3, one run on 4 L40S, control's absolute schedule, FLOP-matched; its CM is read
# against data-v1-round6 (B_3 at 3e17)
wave(
    "ladder3e17",
    "model-v1-ladder3e17",
    "ml3",
    variants(
        "3e17",
        b,
        {
            "model": dict(
                arch=dict(board="conv", mlp="swiglu", key_offset=False), abs_sched=True
            )
        },
    ),
    "3e17: B_3 + boardcnn + swiglu + no key offset, s42, vs data-v1-round6's B_3",
    "general4",
)

wave(
    "ladder3e17s43",
    "model-v1-ladder3e17s43",
    "ml3s",
    variants(
        "3e17",
        b,
        {
            "model": dict(
                arch=dict(board="conv", mlp="swiglu", key_offset=False), abs_sched=True
            )
        },
        seeds=(43,),
    ),
    "3e17 model recipe seed 43 on 4 fast preempt GPUs (second seed of the ladder arm)",
    "preempt4",
)

# the ship recipe at 3e17 on one 8-GPU node (user: ship fast); same run as ladder3e17 (the gradient
# normalisation is fixed at /8, so world size does not change the math)
wave(
    "ship3e17",
    "model-v1-ship3e17",
    "msh",
    variants(
        "3e17",
        b,
        {
            "model": dict(
                arch=dict(board="conv", mlp="swiglu", key_offset=False), abs_sched=True
            )
        },
    ),
    "3e17 ship recipe on 8 x L40S: B_3 + boardcnn + swiglu + no key offset, s42",
    "general8",
)


# moe-v1 round 1 (main, user 2026-09-19: >= 2x CM over the dense control at 1e18, growing with scale): 3e16 screens on
# the ship recipe (B_3 + boardcnn + swiglu + no key offset); experts SwiGLU unless noted; MoE v2 routing + seq loss 1e-3
SHIP = dict(board="conv", mlp="swiglu", key_offset=False)
MOE = lambda e, k, **kw: dict(arch=SHIP | dict(moe=[e, k], moe_seq=1e-3) | MOE_V2 | kw)
MOE1 = {
    "moe64k8": MOE(64, 8),
    "moe64k8relu": MOE(64, 8, mlp="relu2"),
    "moe32k4": MOE(32, 4),
    "moe128k8": MOE(128, 8),
    "moe128k6": MOE(128, 6),
    "moe256k6": MOE(256, 6),
    "moe384k6": MOE(384, 6),
    "moe64k8ssp": MOE(64, 8, moe_score="sqrtsoftplus"),
    "moe64k8cap2": MOE(64, 8, moe_capacity=2.0),
    "moe64k8noshared": MOE(64, 8, moe_shared=False),
}
wave(
    "moe1",
    "moe-v1-round1",
    "mo1",
    variants("3e16", b, {"dense": dict(arch=SHIP)}, seeds=(42, 43)) + variants("3e16", b, MOE1),
    "moe-v1 round 1 at 3e16 on single A6000s: dense ship recipe x2 seeds vs 10 MoE arms (granularity, sparsity incl. "
    "DeepSeek-V4.1 Flash's top-6 / sqrtsoftplus, ReLU^2 vs SwiGLU experts, capacity, shared expert), FLOP-matched",
    "dei1",
)

# round 2's 1e17 control, started while round 1 runs (user: keep general full): dense ship recipe
wave(
    "moe2d",
    "moe-v1-round2d",
    "mo2d",
    variants("1e17", b, {"dense": dict(arch=SHIP)}, seeds=(42, 43)),
    "moe-v1 round 2 dense control at 1e17: ship recipe (B_3 + boardcnn + swiglu + no key offset) x2 seeds, 4 L40S each",
    "general4",
)

# round 2 (1e17 slope point): the sparsity winners of round 1 on the fused dropless kernels (capacity 2
# matched 1.25, so drops cost nothing), 4 fast preempt GPUs each; dense control is moe2d
MOE2 = {k: MOE(e, 6, moe_kernel="scatter") for k, e in (("moe128k6", 128), ("moe384k6", 384))}
wave(
    "moe2",
    "moe-v1-round2",
    "mo2",
    variants("1e17", b, MOE2),
    "moe-v1 round 2 at 1e17: moe128k6 (best CM per chip-hour at 3e16) and moe384k6 (best CM), scatter kernel",
    "preempt4",
)

# round 3 (user, ~20:50): the 3e17 gap test of moe128k6 against the ship run (msh-3e17-model, same shape and
# budget), and moe256k6 (round 1's best) at 1e17 beside moe2's arms; both on the scatter kernel
wave(
    "moe3",
    "moe-v1-round3",
    "mo3",
    variants("3e17", b, {"moe128k6": MOE2["moe128k6"]}),
    "moe-v1 round 3 at 3e17: moe128k6 on 8 dei A6000s vs the ship run as its dense control",
    "dei8",
)
# the same 3e17 gap test on the 3D-expert code (9b05b7e: bit-identical, faster) on 4 fast preempt GPUs;
# moe3 (pre-3D, 4 A6000s) ran at 60K tok/s, ~12 h (science fork, ~22:30)
wave(
    "moe3f",
    "moe-v1-round3f",
    "mo3f",
    variants("3e17", b, {"moe128k6": MOE2["moe128k6"]}),
    "moe-v1 round 3 (fast lane) at 3e17: moe128k6 on 4 fast preempt GPUs vs the ship run",
    "preempt4",
)
wave(
    "moe2b",
    "moe-v1-round2b",
    "mo2b",
    variants("1e17", b, {"moe256k6": MOE(256, 6, moe_kernel="scatter")}),
    "moe-v1 round 2b at 1e17: moe256k6 (round 1's best CM), scatter kernel, 4 fast preempt GPUs",
    "preempt4",
)

# isoFLOPs (MoE-science fork, ~21:30): compute-optimal active size at sparsity 16.5 (E=128 top-4 + shared
# half-width expert, scatter kernel) at the 1e17 and 3e17 budgets, with ship-recipe dense points at the
# outer shapes (the rung shapes' dense controls exist: mo2d at 12x512, the ship run at 16x768). Expert
# weights stay FP32 and replicated, so the largest shapes use smaller micro-batches
S16 = MOE(128, 4, moe_kernel="scatter")
SHAPES = {
    "1e17": {"8x384": dict(depth=8 / 12, width_mul=0.75), "12x512": {}, "14x640": dict(depth=14 / 12, width_mul=1.25),
             "16x768": dict(depth=16 / 12, width_mul=1.5, micro_batch=8)},
    "3e17": {"10x512": dict(depth=10 / 16, width_mul=2 / 3), "12x640": dict(depth=0.75, width_mul=5 / 6),
             "16x768": dict(micro_batch=8), "20x1024": dict(depth=1.25, width_mul=4 / 3, micro_batch=4)},
}  # fmt: skip
ISO_DENSE = {"1e17": ("8x384", "16x768"), "3e17": ("12x640", "20x1024")}
iso = lambda budget: variants(
    budget, b, {f"moe{k}": S16 | v for k, v in SHAPES[budget].items()}
    | {f"dense{k}": dict(arch=SHIP) | SHAPES[budget][k] for k in ISO_DENSE[budget]},
)
# the 3e17 shapes on BF16 weights (bit-identical to FP32 weights, 09dec11; moe20x1024 OOMs without it)
iso_bf16 = lambda budget: [r | dict(bf16_weights=True) for r in iso(budget)]
for budget in SHAPES:
    wave(
        f"iso{budget}",
        f"moe-v1-iso{budget}",
        f"mi{budget[-2:]}",
        iso_bf16(budget) if budget == "3e17" else iso(budget),
        f"moe-v1 isoFLOPs at {budget}: 4 active sizes at sparsity 16.5 (E=128 top-4, scatter) + 2 dense ship shapes",
        "preempt4",
    )
wave(
    "isosmoke",
    "moe-v1-isosmoke",
    "mis",
    [r | dict(stop_after=[12]) for r in iso("3e17") if r["v"] in ("moe20x1024", "moe16x768", "dense20x1024")],
    "memory / speed smoke of the largest isoFLOP shapes on one A6000 (12 steps)",
    "dei1",
)

# router ablation (science fork, 2026-09-20 ~07:30): the 3e17 gap test tied dense with 98-99% of top-k set by the
# balancing bias (affinity margin ~0.007 vs bias range ~0.48): is the router under-trained? 1e17, moe128k6 base
# (mo2's run is the reference arm), scatter-accum + bf16 weights (bit-identical to mo2's scatter)
R = lambda **kw: MOE(128, 6, moe_kernel="scatter-accum", **kw) | dict(bf16_weights=True)
ROUTER = {
    "rlr0.1": R(moe_router_lr_mul=0.1),
    "rlr1": R(moe_router_lr_mul=1.0),
    "g1e-3": R(moe_gamma=1e-3),
    "init0.02": R(moe_init=0.02),
}
wave(
    "moer",
    "moe-v1-router",
    "mor",
    variants("1e17", b, ROUTER),
    "moe-v1 router ablation at 1e17 on moe128k6: router lr x0.1 / x1, bias gamma 1e-3, init 0.02",
    "preempt4",
)

# router hyperparameter search at 3e16 (user 07:15: the fork finds the right router hypers). Round 1's moe128k6 was
# bias-routed at 3e16 too (98.8% of tokens' top-k set by the bias, margin 0.0065 vs bias range 0.38), so search here
# on single GPUs, then confirm at 1e17 and 3e17
ROUTER16 = {"ref": R()} | {
    k: R(**kw)
    for k, kw in {
        "rlr0.1": dict(moe_router_lr_mul=0.1),
        "rlr1": dict(moe_router_lr_mul=1.0),
        "init0.02": dict(moe_init=0.02),
        "init0.05": dict(moe_init=0.05),
        "g1e-3": dict(moe_gamma=1e-3),
        "g3e-3": dict(moe_gamma=3e-3),
        "rlr0.1g1e-3": dict(moe_router_lr_mul=0.1, moe_gamma=1e-3),
        "rlr1g1e-3": dict(moe_router_lr_mul=1.0, moe_gamma=1e-3),
        "rlr0.1init0.02": dict(moe_router_lr_mul=0.1, moe_init=0.02),
    }.items()
}
wave(
    "moer16",
    "moe-v1-router3e16",
    "mr16",
    variants("3e16", b, ROUTER16),
    "moe-v1 router search at 3e16 on moe128k6 (1 GPU each): router lr, init, bias gamma and pairs; ref = current",
    "preempt",
)

# fresh-vs-repeat mix ablation (user 08:35: fill D beyond a policy pass with fresh abundant tokens, never repeat
# experts / OTB past ~4x). Per-pass supply of B_3 over the current stores ~43B (bucket counts x KEEP); relax3 (3x
# the down-sampling keep ratio below 2400, capped at 1) makes a pass ~82B at expert share 0.20 vs 0.38. pool_frac
# = run tokens / final_tokens emulates a final run of that many tokens: 43B = 1 pass, 86B = 2 passes of B_3
# (experts 8x) or ~1 pass of relax3 (experts ~4x)
MIX = {
    "b3p1": dict(final_tokens=43.3e9),
    "b3p2": dict(final_tokens=86.6e9),
    "relax3p2": dict(final_tokens=86.6e9, policy=b["policy"].replace("mover_rule", "mover_rule+relax3")),
}
wave(
    "mixfresh",
    "moe-v1-mixfresh",
    "mxf",
    variants("3e16", b, {k: v | dict(arch=SHIP) for k, v in MIX.items()}, seeds=(42, 43))
    + variants("3e16", b, {f"moe{k}": v | dict(arch=R()["arch"], bf16_weights=True) for k, v in MIX.items()}),
    "fresh vs repeated tokens at 3e16: B_3 at 1 / 2 emulated passes vs relax3 at 2 (fresh abundant), dense x2 seeds + moe128k6",
    "preempt",
)

# router lr x0.1 won at 3e16 (1.4511 / 1.3606 vs ref 1.4655 / 1.3779: -0.014 / -0.017, doubling the gap to dense):
# refine the lr around it at 3e16, and confirm at 3e17 (16x768 vs the ship run); 1e17 is moer task 0
ROUTER16B = {k: R(**kw) for k, kw in {
    "rlr0.03": dict(moe_router_lr_mul=0.03), "rlr0.2": dict(moe_router_lr_mul=0.2),
    "rlr0.3": dict(moe_router_lr_mul=0.3), "rlr0.1g3e-3": dict(moe_router_lr_mul=0.1, moe_gamma=3e-3),
}.items()}
wave("moer16b", "moe-v1-router3e16b", "mr16b", variants("3e16", b, ROUTER16B),
     "moe-v1 router lr refinement at 3e16 on moe128k6 around x0.1", "preempt")
wave("moer3", "moe-v1-router3e17", "mor3", variants("3e17", b, {"rlr0.1": R(moe_router_lr_mul=0.1)}),
     "moe-v1 router lr x0.1 at 3e17: moe128k6 16x768 vs the ship run (the 3e17 gap test with the fixed router)", "preempt4")

# second seed for the headline (user 09:15): rlr0.1 and ref at s43, paired with moer16's s42
wave("moer16s", "moe-v1-router3e16s", "mr16s",
     variants("3e16", b, {"ref": R(), "rlr0.1": R(moe_router_lr_mul=0.1)}, seeds=(43,)),
     "moe-v1 router lr x0.1 vs ref at 3e16, seed 43 (pairs moer16's s42)", "preempt")

# fixed-router isoFLOPs (router lr x0.1, the 3e16 win) at 1e17, single GPUs: 6 MoE shapes spanning 15x in active N
# (the old-router curve was still falling at 12x512 -> smaller shapes bracket the minimum), plus dense 10x448 /
# 14x640 filling the dense curve (8x384 / 12x512 / 16x768 dense exist). Checks: the MoE minimum must not sit at an
# endpoint (else add the next shape); the MoE - dense gap vs D/N at the same shapes separates D/N from router speed
S16F = MOE(128, 4, moe_kernel="scatter-accum", moe_router_lr_mul=0.1) | dict(bf16_weights=True)
ISO17 = {"6x320": dict(depth=0.5, width_mul=0.625), "8x384": dict(depth=8 / 12, width_mul=0.75),
         "10x448": dict(depth=10 / 12, width_mul=0.875), "12x512": {}, "14x640": dict(depth=14 / 12, width_mul=1.25),
         "16x768": dict(depth=16 / 12, width_mul=1.5, micro_batch=8)}  # fmt: skip
wave("isof1e17", "moe-v1-isofix1e17", "mf17",
     variants("1e17", b, {f"moe{k}": S16F | v for k, v in ISO17.items()}
              | {f"dense{k}": dict(arch=SHIP, bf16_weights=True) | ISO17[k] for k in ("10x448", "14x640")}),
     "moe-v1 fixed-router isoFLOPs at 1e17: 6 MoE shapes (E128 top-4, router lr x0.1) + dense 10x448 / 14x640",
     "preempt")

# 6x320 violates the trainer's >= 8 layers; the small end of the fixed-router 1e17 grid is 8x256 instead
wave("isof1e17b", "moe-v1-isofix1e17b", "mf17b",
     variants("1e17", b, {"moe8x256": S16F | dict(depth=8 / 12, width_mul=0.5)}),
     "moe-v1 fixed-router isoFLOPs at 1e17, small end: 8x256 (replaces the invalid 6x320)", "preempt")

# fixed-router isoFLOPs at 3e17 (router lr x0.1; the 3e17 gap test 10510543 trains -0.01..-0.04 below the old router at
# step 500): 5 MoE shapes spanning 8x in active N around dense's 16x768 optimum, bf16 weights, 4 GPUs each; dense
# 10x512 pairs the small end (dense 12x640 / 16x768 / 20x1024 exist)
ISO37 = {"10x512": dict(depth=10 / 16, width_mul=2 / 3), "12x640": dict(depth=0.75, width_mul=5 / 6),
         "16x768": dict(micro_batch=8), "18x896": dict(depth=18 / 16, width_mul=7 / 6, micro_batch=8),
         "20x1024": dict(depth=1.25, width_mul=4 / 3, micro_batch=4)}  # fmt: skip
wave("isof3e17", "moe-v1-isofix3e17", "mf37",
     variants("3e17", b, {f"moe{k}": S16F | v for k, v in ISO37.items()}
              | {"dense10x512": dict(arch=SHIP, bf16_weights=True) | ISO37["10x512"]}),
     "moe-v1 fixed-router isoFLOPs at 3e17: 5 MoE shapes (E128 top-4, router lr x0.1) + dense 10x512", "preempt4")

wave(
    "moe1s",
    "moe-v1-smoke1",
    "mos1",
    [
        r | dict(stop_after=[40])
        for r in variants(
            "3e16", b, {k: MOE1[k] for k in ("moe64k8", "moe384k6", "moe64k8ssp", "moe64k8noshared")}
        )
    ],
    "moe-v1 smoke: 40 steps of the new MoE paths (SwiGLU experts, 384 experts' memory, sqrtsoftplus, no shared "
    "expert) on A6000: compile, finite loss, optimizer labels, tokens/s",
    "a6000p",
)

def name(w, r):
    v = r["v"].replace(".", "p")  # modded_train accepts only [A-Za-z0-9_-] in run names
    return f"{w['prefix']}-{r['budget']}-{v}-{r['tag']}-s{r['seed']}"


EVALUATOR = ("eval_modded.py", "ce_alignment.py", "modded_runtime.py", "eval_strat.py")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def study_hashes(study):
    """Hashes of a frozen study's copies, keyed as its plan's hashes (b2 from the plan)."""
    plan = json.loads((study / "plan.json").read_text())
    return {k: sha(study / k) for k in plan["hashes"] if k != "b2"} | {
        "b2": plan["hashes"]["b2"],
    }


def frozen_files(commit=None):
    """{path inside the study: bytes} for sources, evaluator and drivers, from a git commit (the
    shared B_2 baseline commit) or the working tree."""
    git = lambda *a: subprocess.check_output(["git", "-C", ROOT, *a])
    if commit:
        names = git("ls-tree", "-r", "--name-only", commit, "scripts").decode().split()
        read = lambda n: git("show", f"{commit}:{n}")
    else:  # this checkout (a worktree)
        here = Path(__file__).resolve().parent
        names = [f"scripts/{p.name}" for p in here.iterdir() if p.is_file()]
        read = lambda n: (here.parent / n).read_bytes()
    keep = ("lm_data.py", "lm_checkpoint.py", "chessmix.py", "chess_vocab.py")
    keep += ("board_encode.cpp", "move-table.json")  # modded_board's encoder (Codex)
    out = {
        f"source-ours/{Path(n).name}": read(n)
        for n in sorted(names)
        if Path(n).name in keep
        or (Path(n).name.startswith("modded_") and n.endswith(".py"))
    }
    out |= {
        f"evaluator-ours/{f}": (dx.OLD_EVAL / f).read_bytes() for f in EVALUATOR[:3]
    }
    out["evaluator-ours/eval_strat.py"] = read("scripts/eval_strat.py")
    return out | {
        f: read(f"scripts/{f}") for f in ("modelexp.py", "dataexp.py", "modded_arch.py")
    }


def planned(w, r):
    layers, width, steps = shape(r)
    s, decay = schedule(r, steps)
    return r | dict(
        pool_frac=steps * 512 * 1024 / r.get("final_tokens", dx.FINAL_TOKENS),
        b2=B2_HASH,
        group=w["group"],
        name=name(w, r),
        layers=layers,
        width=width,
        steps=steps,
        schedule=s,
        decay_start=decay,
    )


def plan(key, commit=None):
    w = WAVES[key]
    assert key.startswith("smoke") or not b.get("provisional"), "B_2 not written yet"
    # store contents change as history months land: screen exactly B_2's month list
    assert b.get("months"), "B_2 has no pinned month list"
    files = frozen_files(commit)
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    hashes |= {"history-counts.json": sha(history_file()), "b2": B2_HASH}
    for (
        k,
        o,
    ) in WAVES.items():  # lane-split waves share controls only if they froze the same
        p = ROOT / "results/recipe10x" / o["study"] / "plan.json"
        if k != key and o["group"] == w["group"] and p.exists():
            assert json.loads(p.read_text()).get("hashes") == hashes, (
                f"{k} froze other files"
            )
    if w["pool_sources"]:
        proof = json.loads(
            (ROOT / "results/recipe10x" / w["identity"] / "identity.json").read_text()
        )
        mine = {k: v for k, v in hashes.items() if k.startswith("source-ours/")}
        assert proof["equal"] and all(proof["checks"].values()), proof["checks"]
        assert proof["sources"] == mine, "identity proof is for other code"
        assert proof["b2"] == hashes["b2"] and proof["reference"] == w["pool_sources"]
    study = ROOT / "results/recipe10x" / w["study"]
    assert not study.exists(), "never overwrite a frozen study"
    for d in ("source-ours", "evaluator-ours", "logs", "results"):
        (study / d).mkdir(parents=True)
    for rel, data in files.items():
        (study / rel).write_bytes(data)
    shutil.copy2(history_file(), study / "history-counts.json")
    if w["pool_sources"]:
        shutil.copy2(
            ROOT / "results/recipe10x" / w["identity"] / "identity.json", study
        )
        hashes["identity.json"] = sha(study / "identity.json")
    runs = [planned(w, r) for r in w["runs"]]
    for r in runs:
        if r["v"] == "base":
            assert abs(r["pool_frac"] - b["pool_frac"][r["budget"]]) < 1e-9, r[
                "pool_frac"
            ]
    (study / "plan.json").write_text(
        json.dumps(
            dict(
                wave=key, purpose=w["purpose"], commit=commit, hashes=hashes, runs=runs
            ),
            indent=2,
        )
        + "\n"
    )
    (study / "run.sbatch").write_text(
        f"""#!/bin/bash
#SBATCH --job-name={w["prefix"]}
{w["sbatch"]}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-{-(-len(runs) // w["pack"]) - 1}{f"%{w['throttle']}" if w["throttle"] else ""}
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output={study}/logs/%x-%A_%a.out
exec {ROOT}/.venv/bin/python {study}/modelexp.py task {key}
"""
    )


def submit(key):
    study = ROOT / "results/recipe10x" / WAVES[key]["study"]
    assert not (study / "submitted.json").exists(), "already submitted"
    assert key.startswith("smoke") or not b.get("provisional"), "B_2 not written yet"
    job = subprocess.check_output(
        ["sbatch", "--parsable", str(study / "run.sbatch")], text=True
    ).strip()
    (study / "submitted.json").write_text(
        json.dumps(dict(at=time.time(), job=job)) + "\n"
    )
    print(job)


def task(key):
    w = WAVES[key]
    study = ROOT / "results/recipe10x" / w["study"]
    plan = json.loads((study / "plan.json").read_text())
    got = study_hashes(study)
    bad = sorted(k for k in plan["hashes"] if got.get(k) != plan["hashes"][k])
    assert not bad, f"frozen copies changed since plan: {bad}"
    runs = plan["runs"]  # frozen at plan time
    i, k = int(os.environ["SLURM_ARRAY_TASK_ID"]), w["pack"]
    mine = runs[i * k : (i + 1) * k]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    rg = w["gpus"] // w["pack"]  # GPUs per run (one torchrun process each)
    with ThreadPoolExecutor(len(mine)) as ex:
        ok = list(
            ex.map(
                lambda j: run_one(
                    study, mine[j], ",".join(visible[j * rg : (j + 1) * rg])
                ),
                range(len(mine)),
            )
        )
    if not all(ok):
        subprocess.run(["scontrol", "requeue", dx.task_id()], check=True)


def train_args(study, r):
    """modded_train arguments of a planned run (all but --max-seconds / --stop-after / --resume)."""
    args = [
        "--name", r["name"], "--width", r["width"], "--layers", r["layers"], "--head-dim", 64,
        "--steps", r["steps"], "--extension-steps", 0, "--initial-batch-rows", 512,
        "--micro-batch", r.get("micro_batch", 16),
        "--lr-scale", r.get("lr", 1), "--seed", r["seed"], "--eval-every", 10**7,
        "--checkpoint-every", 128, "--keep-checkpoints", 2, "--val-rows", 1024, "--deterministic",
        "--data", dx.DATA, "--mix", r["policy"], "--mix-pool-frac", r["pool_frac"],
        "--mix-stores", ",".join(r["stores"]), "--mix-months", ",".join(r["months"]),
        "--wsd-schedule", json.dumps(r["schedule"], sort_keys=True),
        "--wsd-end-step", r["steps"], "--wsd-decay-start", r["decay_start"],
    ]  # fmt: skip
    if r.get("history"):
        args += ["--mix-history", study / "history-counts.json"]
    for k, flag in DATA_FLAGS.items():
        if r.get(k) is True:
            args += [flag]
        elif r.get(k):
            args += [flag, r[k]]
    args += ["--no-value-embeds"] * (not r.get("value_embeds", True))
    args += ["--no-skips"] * (not r.get("skips", True))
    args += ["--no-smear"] * (not r.get("smear", True))
    args += ["--doc-rope"] * bool(r.get("doc_rope"))
    args += ["--rope-fp32"] * bool(r.get("rope_fp32"))
    if r.get("arch"):
        args += ["--arch", json.dumps(r["arch"], sort_keys=True)]
    if r.get("wd", 1) != 1:
        args += ["--wd-scale", r["wd"]]
    args += ["--bf16-weights"] * bool(r.get("bf16_weights"))
    args += r.get("extra_args", [])  # perf flags under test (e.g. --zero2, --ckpt eager, --fp8)
    return args


# modded_train arguments that change what a run computes (operational ones such as name, seed,
# checkpoint cadence, time limits and resume paths are excluded); absent flags take these defaults
NUMERIC = dict(
    width=None, layers=None, head_dim=None, steps=None, extension_steps=None,
    initial_batch_rows=None, micro_batch=None, lr_scale=None, deterministic=False, mix=None,
    mix_stores=None, mix_pool_frac=None, mix_months=None, mix_history="", clock=False, elo=False,
    clock_feats=0, input_lr_mul=75.0, aux_time=0.0, aux_wdl=0.0, wsd_schedule=None,
    wsd_end_step=None, wsd_decay_start=None, no_value_embeds=False, no_skips=False,
    no_smear=False, wd_scale=1.0, doc_rope=False, rope_fp32=False, arch="{}",
)  # fmt: skip


def numerics(args):
    """Comparable form of a run's arguments: history file by content, numbers as floats."""
    out = {}
    for k, default in NUMERIC.items():
        v = args.get(k, default)
        if k == "mix_history":
            v = sha(v) if v else ""
        elif k == "arch":
            v = json.dumps(json.loads(v) if isinstance(v, str) else v, sort_keys=True)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            v = float(v)
        out[k] = v if isinstance(v, (bool, float)) else str(v)
    return out


def parse(argv):
    out, i = {}, 0
    while i < len(argv):
        k = str(argv[i]).removeprefix("--").replace("-", "_")
        more = i + 1 < len(argv) and not str(argv[i + 1]).startswith("--")
        out[k], i = (argv[i + 1], i + 2) if more else (True, i + 1)
    return out


def trainer_sources(src):
    """The files modded_train hashes into config.json's source_sha256 for a --mix run."""
    names = [p.name for p in src.glob("modded_*.py")] + [
        "lm_data.py", "lm_checkpoint.py", "chessmix.py", "chess_vocab.py"
    ]  # fmt: skip
    return {n: sha(src / n) for n in names}


def identity(key, control_study, control, reference):
    """Bit-identity proof that a pilot's default path is the reference study's: the pilot's base run
    and the control run (data track, same B_2, same GPU type) log equal train_ce at every 25th step
    through the pilot's last leg (past the embedding split and the resume); the control's executed
    numerics equal the pilot base's, its trainer sources and evaluator are the reference study's,
    and its study, result and the pilot share one B_2. Written to <pilot study>/identity.json."""
    R = ROOT / "results/recipe10x"
    study, cs, ref = R / WAVES[key]["study"], R / control_study, R / reference
    plan = json.loads((study / "plan.json").read_text())
    base = next(r for r in plan["runs"] if r["v"] == "base")
    runs = [ROOT / "results/pretrain" / n for n in (base["name"], control)]
    cfg = [json.loads((d / "config.json").read_text()) for d in runs]
    rec = [
        {
            r["step"]: r["train_ce"]
            for r in map(json.loads, open(d / "train.jsonl"))
            if "train_ce" in r
        }
        for d in runs
    ]
    steps = list(range(25, base["stop_after"][-1] + 1, 25))
    result = json.loads((cs / "results" / f"{control}.json").read_text())
    control_b2 = json.loads((cs / "plan.json").read_text()).get("b2_sha256")
    checks = dict(
        steps=all(s in rec[0] and s in rec[1] for s in steps),
        train_ce=all(rec[0].get(s) == rec[1].get(s) for s in steps),
        numerics=numerics(cfg[0]["args"]) == numerics(cfg[1]["args"]),
        control_sources=cfg[1]["source_sha256"] == trainer_sources(ref / "source-ours"),
        control_evaluator=all(
            sha(cs / "evaluator-ours" / f) == sha(ref / "evaluator-ours" / f)
            for f in EVALUATOR
        ),
        b2=control_b2 == result.get("b2_sha256") == plan["hashes"]["b2"],
        fresh=not cfg[1].get("continuation_provenance")
        and not any(
            cfg[1]["args"].get(k) for k in ("wsd_fork_from", "wsd_continue_from")
        ),
    )
    proof = dict(
        equal=all(checks.values()),
        checks=checks,
        steps=steps,
        train_ce=[[rec[0].get(s), rec[1].get(s)] for s in steps],
        runs=[base["name"], control],
        control_study=control_study,
        reference=reference,
        logs_sha256=[sha(d / "train.jsonl") for d in runs],
        jobs=[
            c["job_id"] for c in cfg
        ],  # pilot lane l40s; round3g's B_2 s42 ran on general L40S
        sources={
            k: v for k, v in plan["hashes"].items() if k.startswith("source-ours/")
        },
        b2=plan["hashes"]["b2"],
    )
    (study / "identity.json").write_text(json.dumps(proof, indent=2) + "\n")
    print("identity", proof["equal"], checks)


def pooled_controls(study, controls):
    """Data-track B_2 runs usable as this model study's controls: identical executed numerics (vs
    this study's base-equivalent run), trainer source hashes, evaluator files, B_2 (the control
    study's plan and the run's result both record this study's b2_sha256), and fresh runs (no fork
    or continuation; a same-run resume is an exact continuation and allowed)."""
    plan = json.loads((study / "plan.json").read_text())
    w = WAVES[plan["wave"]]
    rows = []
    for budget in sorted({r["budget"] for r in plan["runs"]}):
        # the run a control must equal: base at 3e16, the absolute-schedule control model above
        ref = "base" if budget == "3e16" else "control"
        spec = {} if ref == "base" else dict(abs_sched=True)
        base = planned(w, variants(budget, b, {ref: spec})[0])
        assert base["b2"] == plan["hashes"]["b2"], (
            "the loaded B_2 is not the one this study froze"
        )
        want = numerics(parse(train_args(study, base)))
        ref = (
            ROOT / "results/recipe10x" / w["pool_sources"]
            if w["pool_sources"]
            else study
        )
        mine_src = trainer_sources(ref / "source-ours")
        mine_ev = {f: sha(ref / "evaluator-ours" / f) for f in EVALUATOR}
        for c in controls:
            cs = ROOT / "results/recipe10x" / c
            ev = {f: sha(cs / "evaluator-ours" / f) for f in EVALUATOR}
            planned_b2 = json.loads((cs / "plan.json").read_text()).get("b2_sha256")
            for p in sorted((cs / "results").glob("*.json")):
                res = json.loads(p.read_text())
                cfg = ROOT / "results/pretrain" / res["name"] / "config.json"
                if res.get("budget") != budget or not cfg.exists():
                    continue
                config = json.loads(cfg.read_text())
                why = (
                    [k for k, v in numerics(config["args"]).items() if want[k] != v]
                    + ["sources"] * (config["source_sha256"] != mine_src)
                    + ["evaluator"] * (ev != mine_ev)
                    + ["B_2"] * (not planned_b2 == res.get("b2_sha256") == base["b2"])
                    + [
                        k
                        for k in ("wsd_fork_from", "wsd_continue_from")
                        if config["args"].get(k)
                    ]
                    + ["continuation"] * bool(config.get("continuation_provenance"))
                )
                if why:
                    print(f"not pooled: {res['name']} ({', '.join(why)})")
                    continue
                train = [json.loads(l) for l in open(cfg.parent / "train.jsonl")]
                rows.append(
                    res
                    | dict(
                        v=ref,
                        study=plan["runs"][0]["group"],
                        b2=plan["hashes"]["b2"],
                        useful_training_flops=train[-1]["useful_training_flops"],
                    )
                )
    return rows


def run_one(study, r, gpu):
    """Train, then score one run on original validation and the golden eval."""
    n = r["name"]
    result = study / "results" / f"{n}.json"
    if result.exists():
        return True
    started, left = time.monotonic(), dx.seconds_left()
    env = os.environ | dict(
        ALLIE_PROJECT_ROOT=str(ROOT),
        CUDA_VISIBLE_DEVICES=gpu,
        OMP_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
        TORCHINDUCTOR_COMPILE_THREADS="4",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        CUDA_MODULE_LOADING="LAZY",
        DISABLE_FP8="1",
        NCCL_CUMEM_HOST_ENABLE="0",
        NCCL_IB_DISABLE="1",
        NCCL_P2P_DISABLE="1",
        TORCHINDUCTOR_CACHE_DIR=f"/scratch/yimingz3/allie/model-inductor-{gpu}",
        TRITON_CACHE_DIR=f"/scratch/yimingz3/allie/model-triton-{gpu}",
        PYTHONPATH="/data/group_data/dei-group/yimingz3/allie/envs/chessmix-overlay",
    )
    src = (
        ROOT / "results/recipe10x" / r["src"] / "source-ours"
        if "src" in r
        else study / "source-ours"
    )
    py = subprocess.check_output(
        [sys.executable, src / "modded_runtime_stage.py"], env=env, text=True
    ).strip()
    out, log = ROOT / "results/pretrain" / n, study / "logs" / n
    done = out / "done.json"
    for leg in r.get("stop_after", [None]):
        d = json.loads(done.read_text()) if done.exists() else {}
        if d.get("stop_reason") == "steps" or (leg and d.get("step", 0) >= leg):
            continue
        cmd = [py, "-m", "torch.distributed.run", "--standalone"]
        cmd += [f"--nproc_per_node={len(gpu.split(','))}"]
        cmd += [src / "modded_train.py", *train_args(study, r)]
        cmd += [
            "--max-seconds",
            max(600, left - int(time.monotonic() - started) - 1500),
        ]
        if leg:
            cmd += ["--stop-after", leg]
        if (out / "last.pt").exists():
            cmd += ["--resume", out / "last.pt"]
        dx.run(cmd, env, f"{log}.train.log")
        d = json.loads(done.read_text())
        if d["stop_reason"] != ("stop_after" if leg else "steps"):
            return False
    ev = study / "evaluator-ours/eval_strat.py"
    for split, f in (("original_val", "original-val.json"), ("strat", "strat-v1.json")):
        if not (ROOT / "results/lm-eval" / n / f).exists():
            dx.run(
                [py, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1", ev,
                 "--checkpoint", out / "last.pt", "--source", src, "--split", split, "--batch", 16],
                env, f"{log}.{split}.log",
            )  # fmt: skip
    ov = json.loads((ROOT / "results/lm-eval" / n / "original-val.json").read_text())
    sv = json.loads((ROOT / "results/lm-eval" / n / "strat-v1.json").read_text())
    flops = [json.loads(l) for l in open(out / "train.jsonl")][-1][
        "useful_training_flops"
    ]
    result.write_text(
        json.dumps(
            r
            | dict(
                ce={k: ov[k + "_ce"] for k in ("move", "expert2400", "expert2600")},
                strat=dict(
                    macro=sv["macro"],
                    expert_macro=sv["expert_macro"],
                    cells=sv["cells"],
                ),
                useful_training_flops=flops,
                job=os.environ["SLURM_JOB_ID"],
                node=os.uname().nodename,
                gpu_name=subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name",
                        "--format=csv,noheader",
                        "-i",
                        gpu.split(",")[0],
                    ],
                    text=True,
                ).strip(),
                task_seconds=time.monotonic() - started,
            ),
            indent=2,
        )
        + "\n"
    )
    return True


def status(key):
    study = ROOT / "results/recipe10x" / WAVES[key]["study"]
    for r in json.loads((study / "plan.json").read_text())["runs"]:
        res, log = (
            study / "results" / f"{r['name']}.json",
            study / "logs" / f"{r['name']}.train.log",
        )
        if res.exists():
            s = json.loads(res.read_text())["strat"]
            print(
                f"{r['name']:42} golden macro {s['macro']:.4f} expert {s['expert_macro']:.4f}"
            )
            continue
        steps = (
            [
                json.loads(l)["step"]
                for l in open(log)
                if l.startswith('{"step"') and "train_ce" in l
            ]
            if log.exists()
            else []
        )
        print(f"{r['name']:42} step {steps[-1] if steps else 0}/{r['steps']}")


def table(*keys, controls=()):
    """Golden loss and CM vs the same-wave B_2 controls ('base') on the fixed control scaling curve."""
    import dmix_fit
    from isoflop_fit import best_loss

    rows = []
    for key in keys:
        study = WAVES[key]["study"]
        rows += [
            {"study": study} | json.loads(p.read_text())
            for p in sorted(
                (ROOT / "results/recipe10x" / study / "results").glob("*.json")
            )
        ]
    for r in rows:  # lane-split waves share controls
        r["study"] = r.get("group", r["study"])
    pooled = {}  # data-track controls, once per group
    for key in keys:
        g = WAVES[key]["group"]
        if controls and g not in pooled:
            pooled[g] = pooled_controls(
                ROOT / "results/recipe10x" / WAVES[key]["study"], controls
            )
    for g, group_rows in pooled.items():
        print(f"pooled for {g}:", ", ".join(r["name"] for r in group_rows))
    rows += [r for group_rows in pooled.values() for r in group_rows]
    out = {}
    for metric, field in (("strat_macro", "macro"), ("strat_expert", "expert_macro")):
        law = dmix_fit.law(metric)
        for group in dict.fromkeys(
            (r["study"], r.get("b2"), r["budget"]) for r in rows
        ):
            c = dmix_fit.BUDGETS[group[2]]
            got = [r for r in rows if (r["study"], r.get("b2"), r["budget"]) == group]
            ref = "control" if any(r["v"] == "control" for r in got) else "base"
            ctrl = [
                r["strat"][field] for r in got if r["v"] == ref
            ]  # 1e17: the control model
            assert ctrl, f"no {ref} for {group}: nothing pooled (reasons above)"
            shift = float(np.mean(ctrl)) - best_loss(law, c)
            curve = lambda x, s=shift: best_loss(law, x) + s
            ctrl_flops = np.mean(
                [r["useful_training_flops"] for r in got if r["v"] == ref]
            )
            for v in dict.fromkeys(r["v"] for r in got):
                mine = [r for r in got if r["v"] == v]
                loss = float(np.mean([r["strat"][field] for r in mine]))
                row = out.setdefault((*group, v), {})
                row["flops"] = (
                    np.mean([r["useful_training_flops"] for r in mine]) / ctrl_flops
                )  # logged, not planned: POS is an estimate
                sd = float(np.std(ctrl, ddof=1)) if len(ctrl) > 1 else float("nan")
                row[field] = (
                    loss,
                    loss - float(np.mean(ctrl)),
                    dmix_fit.multiplier(curve, loss, c),
                    (loss - float(np.mean(ctrl))) / sd,  # in control seed sigmas
                )
    print(
        "| study | B_2 | budget | variant | FLOPs/ctrl | golden macro | Δ | Δ/σ | CM | golden expert | Δ | Δ/σ | CM |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for (study, b2, budget, v), m in out.items():
        cells = " | ".join(
            f"{m[f][0]:.4f} | {m[f][1]:+.4f} | {m[f][3]:+.1f} | {m[f][2]:.2f}x"
            for f in ("macro", "expert_macro")
        )
        print(f"| {study} | {b2} | {budget} | {v} | {m['flops']:.3f} | {cells} |")
    print(
        "FLOPs are nominal: MoE counts k assigned experts per token. Dropped-route-adjusted FLOPs "
        "(x (1 - moe_dropped)) are useful assigned FLOPs, not executed kernel FLOPs: the expert "
        "bmm always runs the padded E x capacity shape; padding and wall time are reported apart"
    )


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "table":  # table WAVE... [--controls DATA_STUDY,...]
        ctl = (
            rest[rest.index("--controls") + 1].split(",")
            if "--controls" in rest
            else ()
        )
        table(
            *[k for k in rest if k != "--controls" and k.split(",")[0] not in ctl],
            controls=ctl,
        )
    elif cmd == "plan":  # plan WAVE [COMMIT]
        plan(*rest)
    elif (
        cmd == "identity"
    ):  # identity PILOT_WAVE CONTROL_STUDY CONTROL_RUN REFERENCE_STUDY
        identity(*rest)
    else:
        dict(submit=submit, task=task, status=status)[cmd](rest[0])
