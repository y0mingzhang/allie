"""Science rounds on the pinned data baseline (B_3) and the ship model recipe.

A wave freezes one study. Its runs are budget x arm x seed, an arm being overrides of the
baseline: shape (depth, width_mul, micro_batch), lr, sched fractions, data keys (policy,
final_tokens, history, stores, months, feats, aux_time, ...), arch keys, stop_after legs
and extra trainer flags. Arms are matched on trainer FLOPs (modded_train.useful_flops):
an arm trains for the steps giving the FLOPs of the budget's base shape under the plain
recipe. Schedule knobs are fractions of training, from the 3e16 base's absolute steps
(8x512, 2274 steps). pool_frac = run tokens / final_tokens emulates the repetition of a
final run of final_tokens. A round declares, e.g.:

  wave("iso1", "moe-v2-iso1e17", "mi1",
       variants("1e17", {"base": {}, "moe128k4": MOE(128, 4, moe_round=64)}, (42, 43)),
       "dense ship vs E128 top-4 at 1e17", "preempt4")

  plan WAVE [COMMIT]   freeze COMMIT's (else this checkout's) trainer and evaluator and this
                       driver into results/recipe10x/STUDY: plan.json and run.sbatch
  submit WAVE          sbatch the study's array once
  task WAVE            one array task: train its runs, score them, write results/
  status WAVE          one line per run
  table WAVE... [--controls STUDY,...]   golden loss and CM vs the 'base' arm
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
from modded_arch import extra_flops

ROOT = Path("/home/yimingz3/src/allie")
STUDIES = ROOT / "results/recipe10x"
DATA = "/data/group_data/dei-group/yimingz3/allie/lichess_tokens_v2"  # validation rows
# B_3, the final data recipe (data-v1-ledger.md); these keys are provenance, not inputs
B3 = STUDIES / "data-v1-B3.json"
B3_META = ("source_study", "source_run", "final_tokens", "history_counts")
B3_META += ("history_counts_sha256", "chessmix_sha256", "sha256")
DATA_FLAGS = dict(
    feats="--clock-feats",
    input_lr="--input-lr-mul",
    aux_time="--aux-time",
    aux_wdl="--aux-wdl",
)
FINAL_TOKENS = 23e9  # the final run B_3's repetition emulates
# isoflop-v1 compute-optimal shapes of the plain recipe: each budget's base shape
SIZES = {
    "1e16": dict(layers=8, width=384, steps=1347),
    "3e16": dict(layers=8, width=512, steps=2274),
    "1e17": dict(layers=12, width=512, steps=5053),
    "3e17": dict(layers=16, width=768, steps=5053),
}
SCHEDULE = dict(batch_rows=512, plateau=4.0, final_lr=0.2, decay_shape="linear")
FRAC = dict(warmup=32 / 2274, mtp=64 / 2274, split=65 / 2274, decay=32 / 2274)
# mean in-game position per token, fitted to r2-3e16-control's logged FLOPs
POS = 45.7
SHIP = dict(board="conv", mlp="swiglu", key_offset=False)
ROUTER = dict(moe_seq=1e-3, moe_init=0.006, moe_router_lr_mul=0.1, moe_gamma=1e-2)
ROUTER |= dict(moe_update="prop", moe_kernel="scatter-dualgather")
MOE = lambda e, k, **kw: dict(arch=dict(moe=[e, k]) | ROUTER | kw)
# results/recipe10x/gpu-allocation.md fast types as sinfo names them
FAST = "RTX_PRO_6000|H200|H100|A100_80GB|A100_80G|L40S|6000Ada"
QOS = {"dei-group": "dei_group_qos", "preempt": "preempt_qos", "general": "normal"}
# lane: runs per task, GPUs per task, partition, gres, constraint, CPUs, GB, array throttle
LANES = dict(
    dei=(4, 4, "dei-group", "gpu:A6000:4", None, 16, 128, None),
    preempt=(1, 1, "preempt", "gpu:1", FAST, 8, 64, 8),
    general=(1, 1, "general", "gpu:1", FAST, 8, 64, 4),
    l40s=(1, 1, "preempt", "gpu:1", "L40S", 8, 64, 2),
    dei1=(1, 1, "dei-group", "gpu:1", "A6000", 8, 64, 8),
    a6000p=(1, 1, "preempt", "gpu:1", "A6000", 8, 64, 14),
    preempt4=(1, 4, "preempt", "gpu:4", FAST, 24, 200, 1),
    general4=(1, 4, "general", "gpu:L40S:4", None, 24, 200, 1),
    general8=(1, 8, "general", "gpu:L40S:8", None, 48, 400, 1),
    dei8=(1, 8, "dei-group", "gpu:A6000:8", None, 48, 400, 1),
)
SOURCE = ("lm_data.py", "lm_checkpoint.py", "chessmix.py", "chess_vocab.py")
SOURCE += ("board_encode.cpp", "move-table.json")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def baseline():
    """B_3's data inputs (hash-checked) with the ship model recipe."""
    b = json.loads(B3.read_text())
    body = {k: v for k, v in b.items() if k != "sha256"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    assert b["sha256"] == digest, "B_3 was edited after it was hashed"
    return b | dict(arch=SHIP, meta={k: b.pop(k) for k in B3_META})


BASE = baseline()


def history_file():
    """The all-history bucket counts B_3 pins."""
    meta = BASE["meta"]
    assert sha(meta["history_counts"]) == meta["history_counts_sha256"]
    assert meta["final_tokens"] == FINAL_TOKENS
    return Path(meta["history_counts"])


def pool_frac(budget):
    return SIZES[budget]["steps"] * 512 * 1024 / FINAL_TOKENS


def per_token(layers, width, arch=None, head_dim=64):
    """Forward trainer FLOPs per token: modded_train.useful_flops for this depth's layer
    pattern with every layer attending to POS earlier positions, plus the arch's extra
    branches (modded_arch.extra_flops); arch None is the plain recipe."""
    gates = layers + 2 * min(5, layers // 2)
    return (
        24 * layers * width**2
        + 2 * width * 2432
        + 2 * (gates * (width // head_dim) * 16 + 64)
        + 4 * width * layers * POS
        + (0 if arch is None else extra_flops(arch, width, layers))
    )


def shape(r):
    """Layers, width (in 64s) and FLOP-matched steps of a run, relative to its budget's
    base shape, so an arm means the same aspect-ratio change at every budget."""
    base = SIZES[r["budget"]]
    layers = round(base["layers"] * r.get("depth", 1))
    width = round(base["width"] * r.get("width_mul", 1) / 64) * 64
    flops = base["steps"] * per_token(base["layers"], base["width"])
    return layers, width, round(flops / per_token(layers, width, r["arch"]))


def schedule(r, steps):
    """WSD schedule and decay start for this run length (r['sched'] overrides FRAC)."""
    f = FRAC | r.get("sched", {})
    warmup = max(1, round(f["warmup"] * steps))
    mtp = max(1, round(f["mtp"] * steps))
    split = max(mtp, round(f["split"] * steps)) | 1
    s = SCHEDULE | dict(warmup_steps=warmup, mtp_steps=mtp, split_step=split)
    return s, max(warmup, round(f["decay"] * steps))


def variants(budget, specs, seeds=(42,)):
    """Runs: the baseline x arm (label -> overrides) x seed. An arm's arch overrides the
    ship recipe key by key."""
    data = {k: v for k, v in BASE.items() if k not in ("pool_frac", "meta")}
    unknown = set(data) - {"policy", "stores", "months", "history", "arch", *DATA_FLAGS}
    assert not unknown, f"baseline keys modelexp does not forward: {unknown}"
    pf = BASE["pool_frac"].get(budget) or pool_frac(budget)
    tag = f"pf{round(pf * 1000):03d}" + "h" * bool(data.get("history"))
    return [
        data
        | spec
        | dict(arch=data["arch"] | spec.get("arch", {}))
        | dict(v=label, budget=budget, seed=s, tag=tag)
        for label, spec in specs.items()
        for s in seeds
    ]


def merge(*specs):
    """One arm from several; arch and sched merge key by key."""
    out = {}
    for s in specs:
        out |= {k: v for k, v in s.items() if k not in ("arch", "sched")}
        for k in ("arch", "sched"):
            out[k] = out.get(k, {}) | s.get(k, {})
    return {k: v for k, v in out.items() if v != {}}


def stack(budget, parts, seed=43):
    """Leave-one-out stack: base, all parts merged, and the merge without each part."""
    specs = {"base": {}, "stack": merge(*parts.values())} | {
        f"stack-no-{k}": merge(*(v for j, v in parts.items() if j != k)) for k in parts
    }
    return variants(budget, specs, (seed,))


WAVES = {}


def wave(key, study, prefix, runs, purpose, lane="dei", group=None):
    """Declare a wave. Waves split across lanes share one group (and its 'base' runs)."""
    pack, gpus, part, gres, constraint, cpus, mem, throttle = LANES[lane]
    sbatch = ["--account=dippolit", f"--partition={part}", f"--qos={QOS[part]}"]
    sbatch += [f"--gres={gres}"] + [f"--constraint={constraint}"] * bool(constraint)
    sbatch += ["--exclude=babel-q9-32,babel-x9-32", f"--cpus-per-task={cpus}"]
    sbatch += [f"--mem={mem}G", "--time=12:00:00"]
    WAVES[key] = dict(
        study=study,
        prefix=prefix,
        runs=runs,
        purpose=purpose,
        pack=pack,
        gpus=gpus,
        sbatch="\n".join(f"#SBATCH {x}" for x in sbatch),
        throttle=throttle,
        group=group or key,
    )


def name(w, r):
    v = r["v"].replace(".", "p")  # run names allow only [A-Za-z0-9_-]
    return f"{w['prefix']}-{r['budget']}-{v}-{r['tag']}-s{r['seed']}"


def planned(w, r):
    layers, width, steps = shape(r)
    s, decay = schedule(r, steps)
    return r | dict(
        pool_frac=steps * 512 * 1024 / r.get("final_tokens", FINAL_TOKENS),
        baseline=BASE["meta"]["sha256"],
        group=w["group"],
        name=name(w, r),
        layers=layers,
        width=width,
        steps=steps,
        schedule=s,
        decay_start=decay,
    )


def frozen_files(commit=None):
    """{path in the study: bytes}: trainer sources, evaluator and this driver, from a commit
    or this checkout."""
    git = lambda *a: subprocess.check_output(["git", "-C", ROOT, *a])
    if commit:
        names = git("ls-tree", "-r", "--name-only", commit, "scripts").decode().split()
        read = lambda n: git("show", f"{commit}:{n}")
    else:
        here = Path(__file__).resolve().parent
        names = [f"scripts/{p.name}" for p in here.iterdir() if p.is_file()]
        read = lambda n: (here.parent / n).read_bytes()
    trainer = lambda n: n in SOURCE or n.startswith("modded_") and n.endswith(".py")
    out = {
        f"source-ours/{Path(n).name}": read(n)
        for n in sorted(names)
        if trainer(Path(n).name)
    }
    out["evaluator-ours/eval_strat.py"] = read("scripts/eval_strat.py")
    return out | {f: read(f"scripts/{f}") for f in ("modelexp.py", "modded_arch.py")}


def plan(key, commit=None):
    w = WAVES[key]
    study = STUDIES / w["study"]
    assert not study.exists(), "never overwrite a frozen study"
    if commit:  # as a sha; commits of every worktree are in ROOT's object store
        rev = ["git", "-C", Path(__file__).parent, "rev-parse", f"{commit}^{{commit}}"]
        commit = subprocess.check_output(rev, text=True).strip()
    files = frozen_files(commit)
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    hashes |= {
        "history-counts.json": sha(history_file()),
        "baseline": BASE["meta"]["sha256"],
    }
    for k, o in WAVES.items():  # a group's waves pool controls only on the same files
        p = STUDIES / o["study"] / "plan.json"
        if k != key and o["group"] == w["group"] and p.exists():
            assert json.loads(p.read_text())["hashes"] == hashes, (
                f"{k} froze other files"
            )
    for d in ("source-ours", "evaluator-ours", "logs", "results"):
        (study / d).mkdir(parents=True)
    for rel, data in files.items():
        (study / rel).write_bytes(data)
    shutil.copy2(history_file(), study / "history-counts.json")
    runs = [planned(w, r) for r in w["runs"]]
    p = dict(wave=key, purpose=w["purpose"], commit=commit, hashes=hashes, runs=runs)
    (study / "plan.json").write_text(json.dumps(p, indent=2) + "\n")
    tasks = -(-len(runs) // w["pack"])
    (study / "run.sbatch").write_text(f"""#!/bin/bash
#SBATCH --job-name={w["prefix"]}
{w["sbatch"]}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-{tasks - 1}{f"%{w['throttle']}" if w["throttle"] else ""}
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output={study}/logs/%x-%A_%a.out
exec {ROOT}/.venv/bin/python {study}/modelexp.py task {key}
""")


def submit(key):
    study = STUDIES / WAVES[key]["study"]
    assert not (study / "submitted.json").exists(), "already submitted"
    cmd = ["sbatch", "--parsable", str(study / "run.sbatch")]
    job = subprocess.check_output(cmd, text=True).strip()
    (study / "submitted.json").write_text(
        json.dumps(dict(at=time.time(), job=job)) + "\n"
    )
    print(job)


def task_id():
    return f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}"


def seconds_left():
    cmd = ["squeue", "-h", "-o", "%L", "-j", task_id()]
    days, _, hms = subprocess.check_output(cmd, text=True).strip().rpartition("-")
    seconds = 0
    for x in hms.split(":"):
        seconds = seconds * 60 + int(x)
    return seconds + 86400 * int(days or 0)


def run(cmd, env, log):
    with open(log, "a") as f:
        cmd = [str(x) for x in cmd]
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def task(key):
    """This array task's `pack` runs, concurrently, over its GPUs."""
    w = WAVES[key]
    study = STUDIES / w["study"]
    plan = json.loads((study / "plan.json").read_text())
    bad = [
        k for k, v in plan["hashes"].items() if k != "baseline" and sha(study / k) != v
    ]
    assert not bad, f"frozen copies changed since plan: {sorted(bad)}"
    i, k = int(os.environ["SLURM_ARRAY_TASK_ID"]), w["pack"]
    mine = plan["runs"][i * k : (i + 1) * k]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    rg = w["gpus"] // w["pack"]  # GPUs per run (one torchrun each)
    gpus = lambda j: ",".join(visible[j * rg : (j + 1) * rg])
    with ThreadPoolExecutor(len(mine)) as ex:
        ok = list(ex.map(lambda j: run_one(study, mine[j], gpus(j)), range(len(mine))))
    if not all(ok):
        subprocess.run(["scontrol", "requeue", task_id()], check=True)


def train_args(study, r):
    """modded_train arguments of a planned run (all but --max-seconds, --stop-after and
    --resume)."""
    args = [
        "--name", r["name"], "--width", r["width"], "--layers", r["layers"],
        "--head-dim", 64, "--steps", r["steps"], "--initial-batch-rows", 512,
        "--micro-batch", r.get("micro_batch", 16), "--lr-scale", r.get("lr", 1),
        "--seed", r["seed"], "--eval-every", 10**7, "--checkpoint-every", 128,
        "--keep-checkpoints", 2, "--val-rows", 1024, "--deterministic",
        "--data", DATA, "--mix", r["policy"], "--mix-pool-frac", r["pool_frac"],
        "--mix-stores", ",".join(r["stores"]), "--mix-months", ",".join(r["months"]),
        "--wsd-schedule", json.dumps(r["schedule"], sort_keys=True),
        "--wsd-end-step", r["steps"], "--wsd-decay-start", r["decay_start"],
    ]  # fmt: skip
    args += ["--mix-history", study / "history-counts.json"] * bool(r.get("history"))
    for k, flag in DATA_FLAGS.items():
        args += [flag, r[k]] * bool(r.get(k))
    args += ["--arch", json.dumps(r["arch"], sort_keys=True)]
    return args + r.get("extra_args", [])  # e.g. --attn-kernel, --ckpt eager


# modded_train arguments that change what a run computes, with the defaults of absent ones
NUMERIC = dict(
    width=None, layers=None, head_dim=None, steps=None, initial_batch_rows=None,
    micro_batch=None, lr_scale=None, deterministic=False, mix=None, mix_stores=None,
    mix_pool_frac=None, mix_months=None, mix_history="", clock_feats=0,
    input_lr_mul=75.0, aux_time=0.0, aux_wdl=0.0, wsd_schedule=None, wsd_end_step=None,
    wsd_decay_start=None, arch="{}", attn_kernel=False,
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
    """argv as modded_train's namespace keys (values unconverted, bare flags True)."""
    out, i = {}, 0
    while i < len(argv):
        k = str(argv[i]).removeprefix("--").replace("-", "_")
        more = i + 1 < len(argv) and not str(argv[i + 1]).startswith("--")
        out[k], i = (argv[i + 1], i + 2) if more else (True, i + 1)
    return out


def trainer_sources(src):
    """The files modded_train hashes into config.json's source_sha256."""
    names = [p.name for p in src.glob("modded_*.py")]
    names += ["lm_data.py", "lm_checkpoint.py", "chessmix.py", "chess_vocab.py"]
    return {n: sha(src / n) for n in names}


def pooled_controls(study, controls):
    """Runs of the control studies usable as this study's 'base' runs: equal executed
    numerics, trainer sources, evaluator and baseline, and fresh (no WSD fork or
    continuation; a same-run resume is exact and allowed)."""
    plan = json.loads((study / "plan.json").read_text())
    w = WAVES[plan["wave"]]
    src = trainer_sources(study / "source-ours")
    ev = sha(study / "evaluator-ours/eval_strat.py")
    rows = []
    for budget in sorted({r["budget"] for r in plan["runs"]}):
        base = planned(w, variants(budget, {"base": {}})[0])
        want = numerics(parse(train_args(study, base)))
        for c in controls:
            cs = STUDIES / c
            for p in sorted((cs / "results").glob("*.json")):
                res = json.loads(p.read_text())
                cfg = ROOT / "results/pretrain" / res["name"] / "config.json"
                if res["budget"] != budget or not cfg.exists():
                    continue
                config = json.loads(cfg.read_text())
                args = config["args"]
                why = [k for k, v in numerics(args).items() if want[k] != v]
                why += ["sources"] * (config["source_sha256"] != src)
                why += ["evaluator"] * (sha(cs / "evaluator-ours/eval_strat.py") != ev)
                why += ["baseline"] * (res.get("baseline") != base["baseline"])
                why += [
                    k for k in ("wsd_fork_from", "wsd_continue_from") if args.get(k)
                ]
                why += ["continuation"] * bool(config.get("continuation_provenance"))
                if why:
                    print(f"not pooled: {res['name']} ({', '.join(why)})")
                    continue
                rows.append(res | dict(v="base", group=w["group"]))
    return rows


def run_one(study, r, gpu):
    """Train, then score one run on original validation and the golden eval."""
    n = r["name"]
    result = study / "results" / f"{n}.json"
    if result.exists():
        return True
    started, left = time.monotonic(), seconds_left()
    env = os.environ | dict(
        ALLIE_PROJECT_ROOT=str(ROOT),
        CUDA_VISIBLE_DEVICES=gpu,
        OMP_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
        TORCHINDUCTOR_COMPILE_THREADS="4",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        CUDA_MODULE_LOADING="LAZY",
        NCCL_CUMEM_HOST_ENABLE="0",
        NCCL_IB_DISABLE="1",
        NCCL_P2P_DISABLE="1",
        TORCHINDUCTOR_CACHE_DIR=f"/scratch/yimingz3/allie/model-inductor-{gpu}",
        TRITON_CACHE_DIR=f"/scratch/yimingz3/allie/model-triton-{gpu}",
        PYTHONPATH="/data/group_data/dei-group/yimingz3/allie/envs/chessmix-overlay",
    )
    src = study / "source-ours"
    stage = [sys.executable, src / "modded_runtime_stage.py"]
    py = subprocess.check_output(stage, env=env, text=True).strip()
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
        cmd += ["--stop-after", leg] * bool(leg)
        cmd += ["--resume", out / "last.pt"] * (out / "last.pt").exists()
        run(cmd, env, f"{log}.train.log")
        if json.loads(done.read_text())["stop_reason"] != (
            "stop_after" if leg else "steps"
        ):
            return False
    scores = ROOT / "results/lm-eval" / n
    for split, f in (("original_val", "original-val.json"), ("strat", "strat-v1.json")):
        if not (scores / f).exists():
            ev = study / "evaluator-ours/eval_strat.py"
            cmd = [
                py,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=1",
            ]
            cmd += [ev, "--checkpoint", out / "last.pt", "--source", src]
            run([*cmd, "--split", split, "--batch", 16], env, f"{log}.{split}.log")
    ov = json.loads((scores / "original-val.json").read_text())
    sv = json.loads((scores / "strat-v1.json").read_text())
    last = json.loads((out / "train.jsonl").read_text().splitlines()[-1])
    query = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i"]
    gpu_name = subprocess.check_output([*query, gpu.split(",")[0]], text=True).strip()
    r = r | dict(
        ce={k: ov[k + "_ce"] for k in ("move", "expert2400", "expert2600")},
        strat={k: sv[k] for k in ("macro", "expert_macro", "cells")},
        useful_training_flops=last["useful_training_flops"],
        job=os.environ["SLURM_JOB_ID"],
        node=os.uname().nodename,
        gpu_name=gpu_name,
        task_seconds=time.monotonic() - started,
    )
    result.write_text(json.dumps(r, indent=2) + "\n")
    return True


def status(key):
    study = STUDIES / WAVES[key]["study"]
    for r in json.loads((study / "plan.json").read_text())["runs"]:
        res = study / "results" / f"{r['name']}.json"
        if res.exists():
            s = json.loads(res.read_text())["strat"]
            print(
                f"{r['name']:42} golden {s['macro']:.4f} expert {s['expert_macro']:.4f}"
            )
            continue
        log = ROOT / "results/pretrain" / r["name"] / "train.jsonl"
        rows = log.read_text().splitlines() if log.exists() else []
        step = json.loads(rows[-1])["step"] if rows else 0
        print(f"{r['name']:42} step {step}/{r['steps']}")


def table(*keys, controls=()):
    """Golden loss, delta and CM of each arm vs its group's 'base' runs: CM on the golden
    law (dmix_fit.law) shifted through the base mean at the budget."""
    import dmix_fit
    from isoflop_fit import best_loss

    rows = []
    for key in keys:
        results = (STUDIES / WAVES[key]["study"] / "results").glob("*.json")
        rows += [json.loads(p.read_text()) for p in sorted(results)]
    first = {WAVES[k]["group"]: WAVES[k]["study"] for k in reversed(keys)}
    for g, study in first.items() if controls else ():
        pooled = pooled_controls(STUDIES / study, controls)
        print(f"pooled for {g}:", ", ".join(r["name"] for r in pooled))
        rows += pooled
    out = {}
    for field, metric in (("macro", "strat_macro"), ("expert_macro", "strat_expert")):
        law = dmix_fit.law(metric)
        for group in dict.fromkeys((r["group"], r["budget"]) for r in rows):
            c = float(group[1])
            got = [r for r in rows if (r["group"], r["budget"]) == group]
            base = [r for r in got if r["v"] == "base"]
            assert base, f"no base runs for {group}"
            ctrl = [r["strat"][field] for r in base]
            curve = lambda x, s=np.mean(ctrl) - best_loss(law, c): best_loss(law, x) + s
            flops = np.mean([r["useful_training_flops"] for r in base])
            sd = float(np.std(ctrl, ddof=1)) if len(ctrl) > 1 else float("nan")
            for v in dict.fromkeys(r["v"] for r in got):
                mine = [r for r in got if r["v"] == v]
                loss = float(np.mean([r["strat"][field] for r in mine]))
                delta = loss - float(np.mean(ctrl))
                row = out.setdefault((*group, v), {})
                row["flops"] = (
                    np.mean([r["useful_training_flops"] for r in mine]) / flops
                )
                row[field] = (
                    loss,
                    delta,
                    delta / sd,
                    dmix_fit.multiplier(curve, loss, c),
                )
    print(
        "| group | budget | arm | FLOPs/base | golden macro | Δ | Δ/σ | CM "
        "| golden expert | Δ | Δ/σ | CM |\n|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for (group, budget, v), m in out.items():
        cells = " | ".join(
            "{:.4f} | {:+.4f} | {:+.1f} | {:.2f}x".format(*m[f])
            for f in ("macro", "expert_macro")
        )
        print(f"| {group} | {budget} | {v} | {m['flops']:.3f} | {cells} |")
    print(
        "FLOPs are logged useful training FLOPs; MoE counts k routed experts per token"
    )


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    match cmd:
        case "plan":
            plan(*rest)
        case "table":
            keys, ctl = rest, ()
            if "--controls" in rest:
                i = rest.index("--controls")
                keys, ctl = rest[:i], rest[i + 1].split(",")
            table(*keys, controls=ctl)
        case _:
            dict(submit=submit, task=task, status=status)[cmd](rest[0])
