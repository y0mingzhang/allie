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
B2 = ROOT / "results/recipe10x/data-v1-B2.json"
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
        )
    },
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
    """Runs: B_2 data config x model spec (label -> overrides) x seeds. pool_frac is set per run in
    plan() from its own tokens, so FLOP-matched shapes keep the final run's repetition."""
    data = {
        k: v for k, v in base.items() if k not in ("pool_frac", "provisional", "meta")
    }
    unknown = set(data) - {"policy", "stores", "months", "history"} - set(DATA_FLAGS)
    assert not unknown, f"B_2 keys modelexp does not forward: {unknown}"
    pf = base["pool_frac"][budget]
    tag = f"pf{round(pf * 1000):03d}" + "h" * bool(data.get("history"))
    return [
        data | spec | dict(v=label, budget=budget, seed=s, tag=tag)
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
wave(
    "screen2",
    "model-v1-screen2",
    "m2",
    variants("3e16", b, SCREEN2),
    "model screen 1, tier 2 at 3e16 on pinned B_2, trainer-FLOP matched: MoE x 2, diff attention, "
    "aux-head weights 0.1 / 0.05 / 0",
    "preempt",
    pool_sources="model-v1-screen1g",
    identity="model-v1-pilot2",
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
        pool_frac=steps * 512 * 1024 / dx.FINAL_TOKENS,
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
    with ThreadPoolExecutor(len(mine)) as ex:
        ok = list(
            ex.map(
                lambda j: run_one(study, mine[j], visible[j % w["gpus"]]),
                range(len(mine)),
            )
        )
    if not all(ok):
        subprocess.run(["scontrol", "requeue", dx.task_id()], check=True)


def train_args(study, r):
    """modded_train arguments of a planned run (all but --max-seconds / --stop-after / --resume)."""
    args = [
        "--name", r["name"], "--width", r["width"], "--layers", r["layers"], "--head-dim", 64,
        "--steps", r["steps"], "--extension-steps", 0, "--initial-batch-rows", 512, "--micro-batch", 16,
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
        base = planned(w, variants(budget, b, {"base": {}})[0])
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
                        v="base",
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
        cmd = [py, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1"]
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
                        gpu,
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
                row[field] = (
                    loss,
                    loss - float(np.mean(ctrl)),
                    dmix_fit.multiplier(curve, loss, c),
                )
    print(
        "| study | B_2 | budget | variant | FLOPs/ctrl | golden macro | Δ | CM | golden expert | Δ | CM |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for (study, b2, budget, v), m in out.items():
        cells = " | ".join(
            f"{m[f][0]:.4f} | {m[f][1]:+.4f} | {m[f][2]:.2f}x"
            for f in ("macro", "expert_macro")
        )
        print(f"| {study} | {b2} | {budget} | {v} | {m['flops']:.3f} | {cells} |")
    print(
        "FLOPs are nominal: MoE counts k executed experts per token; routes dropped over capacity in "
        "training do no MLP work (their rate is train.jsonl moe_dropped)"
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
