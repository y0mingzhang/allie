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

ROOT = dx.ROOT
B2 = ROOT / "results/recipe10x/data-v1-B2.json"
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
#SBATCH --partition={lane}
#SBATCH --qos={qos}
#SBATCH --gres=gpu:1
#SBATCH --constraint={FAST}
#SBATCH --exclude=babel-x9-32
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00""",
            share,
        )
        for lane, qos, share in (
            ("preempt", "preempt_qos", 8),
            ("general", "normal", 4),
        )
    },
)


def baseline():
    """Pinned data config (B_2 once written by the data track; provisional control + clock + aux)."""
    if B2.exists():
        return json.loads(B2.read_text())
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


def per_token(layers, width, head_dim=64):
    """Forward trainer FLOPs per token: modded_train.useful_flops for this depth's layer pattern,
    with both attention windows covering whole rows (every layer attends to POS earlier positions)."""
    gates = layers + 2 * min(5, layers // 2)
    return (
        24 * layers * width**2
        + 2 * width * 2432
        + 2 * (gates * (width // head_dim) * 16 + 64)
        + 4 * width * layers * POS
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
        / per_token(layers, width)
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
    return s, max(warmup, round(f["decay"] * steps))


def variants(budget, base, specs, seeds=(42,)):
    """Runs: B_2 data config x model spec (label -> overrides) x seeds. pool_frac is set per run in
    plan() from its own tokens, so FLOP-matched shapes keep the final run's repetition."""
    data = {k: v for k, v in base.items() if k not in ("pool_frac", "provisional")}
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
}

WAVES = {}


def wave(key, study, prefix, runs, purpose, lane="dei", group=None):
    """group: waves split across lanes share one set of controls in table()."""
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
    )


b = baseline()
B2_HASH = hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest()[:12]
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
# screen 1 over three lanes (gpu-allocation.md): controls and the longest shape on general (not
# preempted), fast preempt for the rest of the long / scale-sensitive runs, dei A6000 packed for the
# component ablations; table() pools them as group "screen1"
# one control seed per lane, so the control spread includes GPU-type variation (Codex)
S1 = dict(
    general=(42, ("d1.5w0.75", "lr0.5", "lr2")),
    preempt=(43, ("d1.25w0.875", "w1.125", "docrope", "ropefp32", "wd2", "wsd80")),
    dei=(44, ("split50", "nosplit", "nomtp", "nove", "noskip", "nosmear")),
)
assert sorted(k for _, ks in S1.values() for k in ks) == sorted(SCREEN1)
for lane, (seed, keys) in S1.items():
    wave(
        f"screen1{lane[0]}",
        f"model-v1-screen1{lane[0]}",
        f"m1{lane[0]}",
        variants("3e16", b, {"base": {}}, (seed,))
        + variants("3e16", b, {k: SCREEN1[k] for k in keys}),
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


def name(w, r):
    return f"{w['prefix']}-{r['budget']}-{r['v']}-{r['tag']}-s{r['seed']}"


def source_files():
    src = ROOT / "scripts"
    return [*src.glob("modded_*.py")] + [
        src / x
        for x in ("lm_data.py", "lm_checkpoint.py", "chessmix.py", "chess_vocab.py")
    ]


def frozen_hashes():
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return {f.name: h(f) for f in sorted(source_files())} | {
        "history-counts.json": h(dx.HISTORY),
        "b2": B2_HASH,
    }


def plan(key):
    w = WAVES[key]
    assert key.startswith("smoke") or not b.get("provisional"), "B_2 not written yet"
    hashes = frozen_hashes()
    for (
        k,
        o,
    ) in WAVES.items():  # lane-split waves share controls only if they froze the same
        p = ROOT / "results/recipe10x" / o["study"] / "plan.json"
        if k != key and o["group"] == w["group"] and p.exists():
            assert json.loads(p.read_text()).get("hashes") == hashes, (
                f"{k} froze other sources/B_2"
            )
    study = ROOT / "results/recipe10x" / w["study"]
    assert not study.exists(), "never overwrite a frozen study"
    for d in ("source-ours", "evaluator-ours", "logs", "results"):
        (study / d).mkdir(parents=True)
    src = ROOT / "scripts"
    for f in source_files():
        shutil.copy2(f, study / "source-ours")
    for f in ("eval_modded.py", "ce_alignment.py", "modded_runtime.py"):
        shutil.copy2(dx.OLD_EVAL / f, study / "evaluator-ours")
    shutil.copy2(src / "eval_strat.py", study / "evaluator-ours")
    for f in (__file__, src / "dataexp.py"):
        shutil.copy2(f, study)
    shutil.copy2(dx.HISTORY, study / "history-counts.json")
    runs = []
    for r in w["runs"]:
        layers, width, steps = shape(r)
        s, decay = schedule(r, steps)
        pf = steps * 512 * 1024 / dx.FINAL_TOKENS
        if r["v"] == "base":
            assert abs(pf - b["pool_frac"][r["budget"]]) < 1e-9, (pf, b["pool_frac"])
        runs.append(
            r
            | dict(
                pool_frac=pf,
                b2=B2_HASH,
                group=w["group"],
                name=name(w, r),
                layers=layers,
                width=width,
                steps=steps,
                schedule=s,
                decay_start=decay,
            )
        )
    (study / "plan.json").write_text(
        json.dumps(
            dict(wave=key, purpose=w["purpose"], hashes=hashes, runs=runs), indent=2
        )
        + "\n"
    )
    (study / "run.sbatch").write_text(f"""#!/bin/bash
#SBATCH --job-name={w["prefix"]}
{w["sbatch"]}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-{-(-len(runs) // w["pack"]) - 1}{f"%{w['throttle']}" if w["throttle"] else ""}
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output={study}/logs/%x-%A_%a.out
exec {ROOT}/.venv/bin/python {study}/modelexp.py task {key}
""")


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
    runs = json.loads((study / "plan.json").read_text())["runs"]  # frozen at plan time
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
        cmd = [
            py, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1", src / "modded_train.py",
            "--name", n, "--width", r["width"], "--layers", r["layers"], "--head-dim", 64,
            "--steps", r["steps"], "--extension-steps", 0, "--initial-batch-rows", 512, "--micro-batch", 16,
            "--lr-scale", r.get("lr", 1), "--seed", r["seed"], "--eval-every", 10**7,
            "--checkpoint-every", 128, "--keep-checkpoints", 2, "--val-rows", 1024,
            "--max-seconds", max(600, left - int(time.monotonic() - started) - 1500), "--deterministic",
            "--data", dx.DATA, "--mix", r["policy"], "--mix-pool-frac", r["pool_frac"],
            "--mix-stores", ",".join(r["stores"]), "--mix-months", ",".join(r["months"]),
            "--wsd-schedule", json.dumps(r["schedule"], sort_keys=True),
            "--wsd-end-step", r["steps"], "--wsd-decay-start", r["decay_start"],
        ]  # fmt: skip
        if r.get("history"):
            cmd += ["--mix-history", study / "history-counts.json"]
        for k, flag in DATA_FLAGS.items():
            if r.get(k) is True:
                cmd += [flag]
            elif r.get(k):
                cmd += [flag, r[k]]
        cmd += ["--no-value-embeds"] * (not r.get("value_embeds", True))
        cmd += ["--no-skips"] * (not r.get("skips", True))
        cmd += ["--no-smear"] * (not r.get("smear", True))
        cmd += ["--doc-rope"] * bool(r.get("doc_rope"))
        cmd += ["--rope-fp32"] * bool(r.get("rope_fp32"))
        if r.get("wd", 1) != 1:
            cmd += ["--wd-scale", r["wd"]]
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


def table(*keys):
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
            if not ctrl:
                continue
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


if __name__ == "__main__":
    cmd = sys.argv[1]
    (
        table(*sys.argv[2:])
        if cmd == "table"
        else dict(plan=plan, submit=submit, task=task, status=status)[cmd](sys.argv[2])
    )
