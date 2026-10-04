"""Frozen Slurm studies on the pinned data baseline (results/recipe10x/data-v1-B3.json) and the
shipped model recipe.

A round file declares waves; a wave freezes one study. Its runs are budget x arm x seed, an arm
being overrides of the baseline: shape (depth, width_mul or absolute layers, width; micro_batch),
lr, sched fractions, data keys (policy, final_tokens, history, stores, months, feats, aux_time,
...), arch keys, stop_after legs and extra trainer flags. A wave's pin (data.pin) fixes its runs'
months and stores and is verified before every training leg; its history file replaces the
baseline's all-history counts. Arms are matched on trainer FLOPs (train.trainer.useful_flops): an
arm trains for the steps giving the FLOPs of the budget's base shape under the plain recipe.
Schedule knobs are fractions of training, from the 3e16 base's absolute steps (8x512, 2274 steps).
pool_frac = run tokens / final_tokens emulates the repetition of a final run of final_tokens. A
round file (results/recipe10x/STUDY/round.py) imports this module and declares, e.g.:

  from allie.experiments.modelexp import MOE, variants, wave
  wave("iso1", "moe-v2-iso1e17", "mi1",
       variants("1e17", {"base": {}, "moe128k4": MOE(128, 4, moe_round=64)}, (42, 43)),
       "dense vs E128 top-4 at 1e17", "preempt4")

  plan ROUND WAVE [COMMIT]   freeze COMMIT's (else this checkout's) trainer and evaluator,
                             this driver, the round file, pin, history and recipe tables
                             into results/recipe10x/STUDY: plan.json and run.sbatch
  submit ROUND WAVE          sbatch the study's array once
  task ROUND WAVE            one array task: train its runs, score them, write results/
  status ROUND WAVE          one line per run
  table ROUND WAVE... [--controls STUDY,...]   golden loss and CM vs the 'base' arm
"""

import fcntl
import hashlib
import json
import os
import runpy
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from allie import paths
from allie.data import pin as datapin
from allie.data.pin import sha
from allie.model.arch import extra_flops
from allie.train.provenance import source_hashes

ROOT = Path(os.environ.get("ALLIE_PROJECT_ROOT", "/home/yimingz3/src/allie"))
G = paths.DATA
STUDIES = ROOT / "results/recipe10x"
RECIPES = STUDIES / "recipes"  # data.mix.RECIPES
DATA = str(G / "lichess_tokens_v2")  # validation rows
# the baseline: the final data recipe; these keys are provenance, not inputs
B3 = STUDIES / "data-v1-B3.json"
B3_META = ("source_study", "source_run", "final_tokens", "history_counts")
B3_META += ("history_counts_sha256", "chessmix_sha256", "sha256")
DATA_FLAGS = dict(
    feats="--clock-feats",
    input_lr="--input-lr-mul",
    aux_time="--aux-time",
    aux_wdl="--aux-wdl",
)
FINAL_TOKENS = 23e9  # the final run whose data repetition the baseline emulates
# each budget's base shape: the plain recipe's compute-optimal one (results/recipe10x/isoflop-v1)
SIZES = {
    "1e16": dict(layers=8, width=384, steps=1347),
    "3e16": dict(layers=8, width=512, steps=2274),
    "1e17": dict(layers=12, width=512, steps=5053),
    "3e17": dict(layers=16, width=768, steps=5053),
    "1e18": dict(layers=16, width=768, steps=16843),  # 3e17's shape, 10/3 its steps
}
SCHEDULE = dict(batch_rows=512, plateau=4.0, final_lr=0.2, decay_shape="linear")
FRAC = dict(warmup=32 / 2274, mtp=64 / 2274, split=65 / 2274, decay=32 / 2274)
# mean in-game position per token, fitted to a 3e16 run's logged FLOPs
POS = 45.7
SHIP = dict(board="conv", mlp="swiglu", key_offset=False)
ROUTER = dict(moe_seq=1e-3, moe_init=0.006, moe_router_lr_mul=0.1, moe_gamma=1e-2)
ROUTER |= dict(moe_update="quantile", moe_kernel="scatter-dualgather")
MOE = lambda e, k, **kw: dict(arch=dict(moe=[e, k]) | ROUTER | kw)
# fast GPU types, as sinfo names them
FAST = "RTX_PRO_6000|H200|H100|A100_80GB|A100_80G|L40S|6000Ada"
QOS = {"dei-group": "dei_group_qos", "preempt": "preempt_qos", "general": "normal"}
# one node per line, # comments: every launcher excludes them
BAD_NODES = G / "bad-nodes"
# runs that stopped requeueing in a crash loop, one json row each
ALERTS = G / "alerts.jsonl"
# lane: runs per task, GPUs per task, partition, gres, constraint, CPUs, GB, array throttle
LANES = dict(
    dei=(4, 4, "dei-group", "gpu:A6000:4", None, 16, 128, None),
    dei4=(1, 4, "dei-group", "gpu:A6000:4", None, 24, 200, 4),
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
# the package a study freezes into STUDY/source/allie: all of it but the inference engine
PACKAGE = Path(__file__).resolve().parents[1]
FROZEN = lambda rel: "__pycache__" not in rel and "/." not in f"/{rel}" and not rel.startswith("search/")  # noqa: E731
EVALUATOR = "source/allie/eval/score.py"


def baseline():
    """The baseline's data inputs (hash-checked) with the shipped model recipe."""
    b = json.loads(B3.read_text())
    body = {k: v for k, v in b.items() if k != "sha256"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    assert b["sha256"] == digest, "B_3 was edited after it was hashed"
    return b | dict(arch=SHIP, meta={k: b.pop(k) for k in B3_META})


BASE = baseline()


def history_file():
    """The all-history bucket counts the baseline pins."""
    meta = BASE["meta"]
    assert sha(meta["history_counts"]) == meta["history_counts_sha256"]
    assert meta["final_tokens"] == FINAL_TOKENS
    return Path(meta["history_counts"])


def pool_frac(budget):
    return SIZES[budget]["steps"] * 512 * 1024 / FINAL_TOKENS


def per_token(layers, width, arch=None, head_dim=64):
    """Forward trainer FLOPs per token: train.trainer.useful_flops for this depth's layer
    pattern with every layer attending to POS earlier positions, plus the arch's extra
    branches (model.arch.extra_flops); arch None is the plain recipe."""
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
    layers = r.get("layers") or round(base["layers"] * r.get("depth", 1))
    width = r.get("width") or round(base["width"] * r.get("width_mul", 1) / 64) * 64
    flops = base["steps"] * per_token(base["layers"], base["width"])
    return layers, width, round(flops / per_token(layers, width, r["arch"]))


def schedule(r, steps):
    """WSD schedule and decay start for this run length (r['sched'] overrides FRAC and the
    SCHEDULE fields, e.g. final_lr; an mtp fraction of 0 turns multi-token prediction off)."""
    f = FRAC | r.get("sched", {})
    warmup = max(1, round(f["warmup"] * steps))
    mtp = max(int(f["mtp"] > 0), round(f["mtp"] * steps))
    split = max(mtp, round(f["split"] * steps)) | 1
    s = SCHEDULE | {k: f[k] for k in SCHEDULE if k in f}
    s |= dict(warmup_steps=warmup, mtp_steps=mtp, split_step=split)
    if f["decay"] < 0:  # stable WSD mainline: only its --wsd-fork-from branches decay
        return s, -1
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
        | dict(tag=tag)
        | spec
        | dict(arch=data["arch"] | spec.get("arch", {}))
        | dict(v=label, budget=budget, seed=s)
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
ROUND = None  # the round file being loaded


def wave(
    key,
    study,
    prefix,
    runs,
    purpose,
    lane="dei",
    group=None,
    pin=None,
    history=None,
    local=False,
):
    """Declare a wave. Waves split across lanes share one group (and its 'base' runs).
    pin: a data.pin file fixing every run's months and stores; history: the all-history
    counts file of history runs (default the baseline's)."""
    pack, gpus, part, gres, constraint, cpus, mem, throttle = LANES[lane]
    sbatch = ["--account=dippolit", f"--partition={part}", f"--qos={QOS[part]}"]
    sbatch += [f"--gres={gres}"] + [f"--constraint={constraint}"] * bool(constraint)
    sbatch += [f"--exclude={bad_nodes()}", f"--cpus-per-task={cpus}"]
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
        pin=pin and Path(pin),
        history=Path(history) if history else history_file(),
        round=ROUND,
        local=local,
    )


def bad_nodes():
    return ",".join(
        x.split("#")[0].strip()
        for x in BAD_NODES.read_text().splitlines()
        if x.split("#")[0].strip()
    )


def name(w, r):
    v = r["v"].replace(".", "p")  # run names allow only [A-Za-z0-9_-]
    return f"{w['prefix']}-{r['budget']}-{v}-{r['tag']}-s{r['seed']}"


def planned(w, r):
    layers, width, steps = shape(r)
    s, decay = schedule(r, steps)
    pf = steps * 512 * 1024 / r.get("final_tokens", FINAL_TOKENS)
    assert pf <= 1, f"{name(w, r)} trains past its final_tokens"
    if w["pin"]:
        pin = json.loads(w["pin"].read_text())
        r = r | dict(months=pin["months"], stores=pin["stores"])
    return (
        r
        | dict(
            pool_frac=pf,
            baseline=BASE["meta"]["sha256"],
            group=w["group"],
            name=name(w, r),
            layers=layers,
            width=width,
            steps=steps,
            schedule=s,
            decay_start=decay,
        )
        | fork(r, steps)
    )


def fork(r, steps):
    """A WSD decay branch of the stable mainline r['fork']['parent'] (a run of the same shape and
    schedule whose fork_steps include 'at'): decays from 'at' to 'end'."""
    if "fork" not in r:
        return {}
    at, end = r["fork"]["at"], r["fork"]["end"]
    assert 0 < at < end <= steps, r["fork"]
    return dict(decay_start=at, end_step=end, fork_steps=[])


def frozen_files(commit=None):
    """{path in the study: bytes}: the allie package (trainer, evaluator and this driver) as
    source/allie/..., from a commit or this checkout."""
    if commit:
        top = subprocess.check_output(["git", "-C", PACKAGE, "rev-parse", "--show-toplevel"], text=True)
        git = lambda *a: subprocess.check_output(["git", "-C", top.strip(), *a])  # noqa: E731
        prefix = PACKAGE.relative_to(top.strip()).as_posix()
        names = git("ls-tree", "-r", "--name-only", commit, "--", prefix)
        names = [n.removeprefix(f"{prefix}/") for n in names.decode().split()]
        assert names, f"{commit} has no {prefix}/: it predates the package layout"
        read = lambda n: git("show", f"{commit}:{prefix}/{n}")  # noqa: E731
    else:
        names = [p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob("*") if p.is_file()]
        read = lambda n: (PACKAGE / n).read_bytes()  # noqa: E731
    return {f"source/allie/{n}": read(n) for n in sorted(names) if FROZEN(n)}


def tables(w, runs):
    """Recipe tables (data.mix table:NAME policies) the runs train on, checked against the
    wave: the table's pin is the wave's and covers it, its basis matches the run's history
    mode (disk: no history; full: history counts of the same pin) and final_tokens is the
    table's token count, so the run's passes per bucket are the table's weights."""
    from allie.data import mix as chessmix  # the cooldown grammar

    out = {}
    for r in runs:
        parts = []  # (policy name, the cooldown start it runs after, None if it is not a late phase)
        for k in r["policy"].split("+"):
            c, t = chessmix.cool(k), chessmix.recent(k)
            if c:
                parts += [(c[1], None), (c[2], c[0])]
            else:  # a recent-only tail trains on its inner policy's table throughout
                parts += [(t[2] if t else k, None)]
        for k, start in parts:
            if not (k or "").startswith("table:"):
                continue
            raw = (RECIPES / f"{k[6:]}.json").read_bytes()
            t, pin = json.loads(raw), json.loads(w["pin"].read_text())
            if t["kind"] == "anneal":  # a late table: only after its own start
                assert start is not None and float(t["args"]["start"]) == start, k
            assert t["pin_digest"] == pin["sha256"], f"{k} is not on the wave's pin"
            assert set(pin["inventory"]) <= set(t["weights"]), f"{k} misses buckets"
            assert r["final_tokens"] == t["training_tokens"], f"{k}: final_tokens"
            hist = json.loads(w["history"].read_text()).get("pin_digest")
            assert (t["basis"] == "full") == bool(r.get("history")), f"{k}: history"
            assert t["basis"] == "disk" or hist == pin["sha256"], f"{k}: history pin"
            want = sha(w["history"]) if t["basis"] == "full" else None
            assert t.get("history_sha256") == want, f"{k}: history file"
            out[f"recipes/{k[6:]}.json"] = raw
    return out


def plan(key, commit=None):
    w = WAVES[key]
    study = STUDIES / w["study"]
    assert not (study / "plan.json").exists(), "never overwrite a frozen study"
    if commit:  # as a sha; commits of every worktree are in ROOT's object store
        rev = ["git", "-C", Path(__file__).parent, "rev-parse", f"{commit}^{{commit}}"]
        commit = subprocess.check_output(rev, text=True).strip()
    runs = [planned(w, r) for r in w["runs"]]
    files = frozen_files(commit) | tables(w, runs)
    files |= {"round.py": w["round"].read_bytes()}
    files |= {"history-counts.json": w["history"].read_bytes()}
    files |= {"data-pin.json": w["pin"].read_bytes()} if w["pin"] else {}
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    hashes["baseline"] = BASE["meta"]["sha256"]
    shared = lambda h: {k: v for k, v in h.items() if not k.startswith("recipes/")}
    for k, o in WAVES.items():  # a group's waves pool controls only on the same files
        p = STUDIES / o["study"] / "plan.json"
        if k != key and o["group"] == w["group"] and p.exists():
            other = json.loads(p.read_text())["hashes"]
            assert shared(other) == shared(hashes), f"{k} froze other files"
    for d in ("source", "recipes", "logs", "results"):
        (study / d).mkdir(parents=True, exist_ok=True)
    for rel, data in files.items():
        if (study / rel).resolve() != w["round"]:
            (study / rel).parent.mkdir(parents=True, exist_ok=True)
            (study / rel).write_bytes(data)
    tasks = -(-len(runs) // w["pack"])
    (study / "run.sbatch").write_text(
        f"""#!/bin/bash
#SBATCH --job-name={w["prefix"]}
{w["sbatch"]}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-{tasks - 1}{f"%{w['throttle']}" if w["throttle"] else ""}
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output={study}/logs/%x-%A_%a.out
{f"export MODELEXP_LOCAL={LOCAL}" if w["local"] else ""}
PYTHONPATH={study}/source exec {ROOT}/.venv/bin/python -m allie.experiments.modelexp task {study}/round.py {key}
"""
    )
    p = dict(wave=key, purpose=w["purpose"], commit=commit, hashes=hashes, runs=runs)
    (study / "plan.json").write_text(json.dumps(p, indent=2) + "\n")  # the commit point


def write(path, text):
    """Publish a file whole: readers see the old version or the new, never a torn one."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def submitted(token, since):
    """Base ids of the Slurm jobs, live or in accounting, whose submit line carries token."""
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(since - 60))
    acct = ["sacct", "-X", "-n", "-P", "-S", t0, "--format=JobID,SubmitLine"]
    live = ["squeue", "--me", "-h", "-o", "%i|%k"]
    rows = [
        x
        for c in (acct, live)
        for x in subprocess.check_output(c, text=True).split("\n")
    ]
    return sorted({x.split("|")[0].split("_")[0] for x in rows if token in x})


def submit(key):
    """sbatch the study's array once, under a lock. The submission carries a token recorded
    before sbatch; a retry that finds the intent but no receipt looks the token up in Slurm
    and records that job, and fails closed if Slurm shows none (the first sbatch may be
    in flight or unaccounted): check by hand, then delete submitting.json."""
    study = STUDIES / WAVES[key]["study"]
    with open(study / ".submit.lock", "w") as lock:
        fcntl.flock(
            lock, fcntl.LOCK_EX | fcntl.LOCK_NB
        )  # raises if another submit runs
        assert not (study / "submitted.json").exists(), "already submitted"
        intent = study / "submitting.json"
        if intent.exists():
            i = json.loads(intent.read_text())
            jobs = submitted(i["token"], i["at"])
            assert len(jobs) == 1, f"unresolved submit {i['token']}: jobs {jobs}"
        else:
            token = f"{w_token(study)}-{os.getpid()}-{time.time_ns()}"
            write(intent, json.dumps(dict(at=time.time(), token=token)) + "\n")
            cmd = [
                "sbatch",
                "--parsable",
                f"--comment={token}",
                f"--exclude={bad_nodes()}",  # the list as of now, not as of the plan
                str(study / "run.sbatch"),
            ]
            jobs = [subprocess.check_output(cmd, text=True).strip().split(";")[0]]
        write(
            study / "submitted.json",
            json.dumps(dict(at=time.time(), job=jobs[0])) + "\n",
        )
    print(jobs[0])


def w_token(study):
    return hashlib.sha256(str(study).encode()).hexdigest()[:12]


def task_id():
    return f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}"


def seconds_left():
    cmd = ["squeue", "-h", "-o", "%L", "-j", task_id()]
    days, _, hms = subprocess.check_output(cmd, text=True).strip().rpartition("-")
    seconds = 0
    for x in hms.split(":"):
        seconds = seconds * 60 + int(x)
    return seconds + 86400 * int(days or 0)


CHILDREN = {}  # running child -> {pid: start time} of its tree so far


def tree(root):
    """root and its descendants, from /proc parent links (session changes don't break them)."""
    kids = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        kids.setdefault(ppid, []).append(int(d.name))
    out, todo = [], [root]
    while todo:
        out.append(todo.pop())
        todo += kids.get(out[-1], [])
    return out


def started(pid):
    """A live process's start time (clock ticks since boot), telling a pid from its reuse;
    None once it is gone or a zombie."""
    try:
        f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except OSError:
        return None
    return None if f[0] in "ZX" else int(f[19])


def kill(known, grace=60):
    """TERM, then KILL, the known processes {pid: start time} still alive."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        alive = [q for q, t in known.items() if started(q) == t]
        for q in alive:
            try:
                os.kill(q, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + grace
        while alive and time.monotonic() < deadline:
            time.sleep(1)
            alive = [q for q in alive if started(q) == known[q]]
        if not alive:
            return


def session(sid):
    """Processes of a session (the child's own: start_new_session makes its id the pid)."""
    out = []
    for d in Path("/proc").iterdir():
        try:
            if (
                d.name.isdigit()
                and int((d / "stat").read_text().rsplit(")", 1)[1].split()[3]) == sid
            ):
                out.append(int(d.name))
        except (OSError, IndexError, ValueError):
            pass
    return out


def reap(p, known=None, grace=60):
    """Kill a child's tree: its live descendants, everything left in its session, and every
    process seen in the tree earlier, so workers orphaned by a dead wrapper go too."""
    known = dict(known or {})
    known |= {q: started(q) for q in tree(p.pid)} if p.poll() is None else {}
    known |= {q: started(q) for q in session(p.pid)}
    kill({q: t for q, t in known.items() if t is not None}, grace)
    p.wait()


def run(cmd, env, log, watch=None, limit=None):
    """A child in its own session, its tree reaped however this ends. watch: a file that
    must grow, first within 1800 s (compile, sampler warm-up) and then within
    max(1800, 20 x its median growth interval so far); limit: seconds of wall time.
    A stalled or overdue child is killed and raises TimeoutError."""
    cmd = [str(x) for x in cmd]
    with open(log, "a") as f:
        p = subprocess.Popen(
            cmd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
        )
    CHILDREN[p] = known = {}
    start = last = time.monotonic()

    def size(old):
        try:
            return watch.stat().st_size if watch else 0
        except OSError:  # missing, or a transient NFS error (errno 512)
            return old

    seen, gaps = size(0), []
    try:
        while p.poll() is None:
            known |= {q: started(q) for q in tree(p.pid)}
            time.sleep(2)
            now = time.monotonic()
            if (n := size(seen)) != seen:
                seen, gaps, last = n, gaps + [now - last], now
            stall = max(1800, 20 * sorted(gaps)[len(gaps) // 2]) if gaps else 1800
            if watch and now - last > stall or limit and now - start > limit:
                raise TimeoutError(f"{cmd[:4]}... no progress in {now - last:.0f} s")
    finally:
        reap(p, known)
        CHILDREN.pop(p)
    if p.returncode:
        raise subprocess.CalledProcessError(p.returncode, cmd)


STOPPING = []  # the signal that is stopping this task, if any


def stop(signum, frame):
    """Preemption or cancel: take every child's tree down with this task."""
    STOPPING.append(signum)
    for p, known in list(CHILDREN.items()):
        reap(p, known, grace=30)
    raise SystemExit(128 + signum)


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
    signal.signal(signal.SIGTERM, stop)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    rg = w["gpus"] // w["pack"]  # GPUs per run (one torchrun each)
    gpus = lambda j: ",".join(visible[j * rg : (j + 1) * rg])
    with ThreadPoolExecutor(len(mine)) as ex:
        ok = list(ex.map(lambda j: arm(study, mine[j], gpus(j)), range(len(mine))))
    failed = [r for r, o in zip(mine, ok) if not o]
    loops = [r for r in failed if looping(study, r)]
    for r in loops:
        alert(study, r)
    if failed and not loops and all(len(failures(study, r)) < TRIES for r in failed):
        subprocess.run(["scontrol", "requeue", task_id()], check=True)
    sys.exit(1 if failed else 0)


TRIES = 3  # failed attempts of a run before its task stops requeueing


def looping(study, r):
    """Its last two failures resumed from the checkpoint it is still at: a crash loop, which requeueing repeats."""
    step = checkpoint_step(r)
    return (
        step is not None
        and [x.get("step") for x in failures(study, r)[-2:]] == [step] * 2
    )


def alert(study, r):
    """Stop requeueing r: an ALERT file next to its logs and a row in ALERTS."""
    f = failures(study, r)
    row = dict(at=time.time(), run=r["name"], study=str(study), step=f[-1]["step"], failures=f[-2:],
               why="two failures at the same checkpoint step: not requeued")  # fmt: skip
    write(study / "logs" / f"{r['name']}.ALERT", json.dumps(row, indent=1) + "\n")
    with open(ALERTS, "a") as out:
        out.write(json.dumps(row) + "\n")


def checkpoint_step(r):
    """The step last.pt points at (0: none; None: unreadable, which never counts as a loop)."""
    import torch

    f = pretrained(r) / "last.pt"
    try:
        return torch.load(f, weights_only=True)["step"] if f.exists() else 0
    except Exception:
        return None


def failures(study, r):
    f = study / "logs" / f"{r['name']}.failed.json"
    return json.loads(f.read_text()) if f.exists() else []


def arm(study, r, gpu):
    """run_one, with a failure receipt instead of an exception; the other arms of the
    task keep their GPUs."""
    try:
        return run_one(study, r, gpu)
    except BaseException as e:
        if STOPPING:  # preempted or cancelled: not the run's failure
            raise
        f = study / "logs" / f"{r['name']}.failed.json"
        rec = dict(at=time.time(), job=os.environ.get("SLURM_JOB_ID"), reason=repr(e))
        rec["step"] = checkpoint_step(r)
        write(f, json.dumps([*failures(study, r), rec], indent=1) + "\n")
        if not isinstance(e, Exception):
            raise
        return False


def train_args(study, r):
    """train.trainer arguments of a planned run (all but --max-seconds, --stop-after and
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
        "--wsd-end-step", r.get("end_step", r["steps"]), "--wsd-decay-start", r["decay_start"],
    ]  # fmt: skip
    args += ["--mix-history", history(study, r)] * bool(r.get("history"))
    if r.get("fork_steps"):  # a stable mainline's checkpoints kept for decay branches
        args += ["--wsd-fork-steps", ",".join(map(str, r["fork_steps"]))]
    for k, flag in DATA_FLAGS.items():
        args += [flag, r[k]] * bool(r.get(k))
    args += ["--arch", json.dumps(r["arch"], sort_keys=True)]
    return args + r.get("extra_args", [])  # e.g. --attn-kernel, --ckpt eager


def history(study, r):
    """The history-counts file; a fork trains on its mainline's copy (the trainer compares paths)."""
    own = study / "history-counts.json"
    if "fork" not in r:
        return own
    cfg = ROOT / "results/pretrain" / r["fork"]["parent"] / "config.json"
    path = json.loads(cfg.read_text())["args"]["mix_history"]
    assert sha(path) == sha(own), "a fork's history counts differ from its mainline's"
    return path


# train.trainer arguments that change what a run computes, with the defaults of absent ones
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
    """argv as train.trainer's namespace keys (values unconverted, bare flags True)."""
    out, i = {}, 0
    while i < len(argv):
        k = str(argv[i]).removeprefix("--").replace("-", "_")
        more = i + 1 < len(argv) and not str(argv[i + 1]).startswith("--")
        out[k], i = (argv[i + 1], i + 2) if more else (True, i + 1)
    return out


def trainer_sources(study):
    """The files the trainer hashes into config.json's source_sha256, of a study's frozen package."""
    return source_hashes(study / "source/allie")


def pooled_controls(study, controls):
    """Runs of the control studies usable as this study's 'base' runs: equal executed
    numerics, trainer sources, evaluator and baseline, and fresh (no WSD fork or
    continuation; a same-run resume is exact and allowed)."""
    plan = json.loads((study / "plan.json").read_text())
    w = WAVES[plan["wave"]]
    src = trainer_sources(study)
    ev = sha(study / EVALUATOR)
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
                old = cs / EVALUATOR
                why += ["evaluator"] * (not old.exists() or sha(old) != ev)
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


def owner(study, r):
    """The manifest binding a run's global checkpoint and score directories to one planned
    run of one frozen study."""
    plan = study / "plan.json"
    frozen = json.loads(plan.read_text())["hashes"] if plan.exists() else {}
    digest = lambda x: hashlib.sha256(
        json.dumps(x, sort_keys=True).encode()
    ).hexdigest()
    return dict(study=str(study), run=digest(r), frozen=digest(frozen))


def claim(study, r):
    """Take the run's global directories for this study's run, or check that it owns them:
    output of another study (or of no recorded owner) is never resumed, skipped or scored."""
    out, scores = pretrained(r), ROOT / "results/lm-eval" / r["name"]
    manifest = out / "owner.json"
    if manifest.exists():
        got = json.loads(manifest.read_text())
        assert got == owner(study, r), f"{out} belongs to {got['study']}"
        return
    assert not out.exists() and not scores.exists(), f"{out}: output with no owner"
    out.mkdir(parents=True)
    write(manifest, json.dumps(owner(study, r), indent=1) + "\n")


def complete(study, r):
    """The run's result is published and describes this planned run, trained to the end in
    directories it owns."""
    res, done = study / "results" / f"{r['name']}.json", pretrained(r) / "done.json"
    if not res.exists():
        return False
    claim(study, r)
    got = json.loads(res.read_text())
    same = all(got.get(k) == r[k] for k in ("name", "steps", "arch", "policy", "seed"))
    assert same and "strat" in got, f"{res} is not this run's result"
    return json.loads(done.read_text())["stop_reason"] == "steps"


def pretrained(r):
    return ROOT / "results/pretrain" / r["name"]


# node-local project root of local waves: they train there and copy back all but the rank files (model.pt, pointers,
# logs), so ROOT's storage holds no optimizer state; a requeue on another node starts the run over
LOCAL = Path("/scratch/yimingz3/allie/modelexp-root")


def finished(out):
    f = out / "done.json"
    return f.exists() and json.loads(f.read_text())["stop_reason"] == "steps"


def local_root():
    root = os.environ.get("MODELEXP_LOCAL")
    if not root:
        return ROOT
    (Path(root) / "results/pretrain").mkdir(parents=True, exist_ok=True)
    link = Path(root) / "results/original-data.json"
    if not link.exists():
        link.symlink_to(ROOT / "results/original-data.json")
    return Path(root)


def run_one(study, r, gpu):
    """Train, then score one run on original validation and the golden eval."""
    n = r["name"]
    if complete(study, r):
        return True
    locks = ROOT / "results/pretrain/.locks"
    locks.mkdir(exist_ok=True)
    lock = open(locks / f"{n}.lock", "w")  # global: run names are global directories
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # raises if another task runs n
    claim(study, r)
    if complete(study, r):
        return True
    result = study / "results" / f"{n}.json"
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
        PYTHONPATH=f"{study / 'source'}:{G / 'envs/chessmix-overlay'}",
    )
    out, log = pretrained(r), study / "logs" / n
    stage = Path(f"{log}.stage.log")
    run([sys.executable, "-m", "allie.train.runtime_stage"], env, stage, limit=3600)
    py = stage.read_text().split()[-1]
    root = local_root()
    train_env = env | dict(ALLIE_PROJECT_ROOT=str(root))
    tout = root / "results/pretrain" / n
    assert root == ROOT or "fork" not in r, "a fork needs its mainline's rank files"
    if finished(out) and not (tout / "done.json").exists():
        tout = out  # trained and copied back earlier
    done = tout / "done.json"
    for leg in r.get("stop_after", [None]):
        d = json.loads(done.read_text()) if done.exists() else {}
        if d.get("stop_reason") == "steps" or (leg and d.get("step", 0) >= leg):
            continue
        if (study / "data-pin.json").exists():
            datapin.verify(study / "data-pin.json", r["months"])
        cmd = [py, "-m", "torch.distributed.run", "--standalone"]
        cmd += [f"--nproc_per_node={len(gpu.split(','))}"]
        cmd += ["-m", "allie.train.trainer", *train_args(study, r)]
        cmd += [
            "--max-seconds",
            max(600, left - int(time.monotonic() - started) - 1500),
        ]
        cmd += ["--stop-after", leg] * bool(leg)
        if (tout / "last.pt").exists():
            cmd += ["--resume", tout / "last.pt"]
        elif "fork" in r:  # the mainline's --wsd-fork-steps pointer
            f = r["fork"]
            cmd += [
                "--wsd-fork-from",
                ROOT / "results/pretrain" / f["parent"] / f"fork-{f['at']}.pt",
            ]
        run(cmd, train_env, f"{log}.train.log", watch=tout / "train.jsonl")
        if json.loads(done.read_text())["stop_reason"] != (
            "stop_after" if leg else "steps"
        ):
            return False
        if tout != out:  # done.json last, so a partial copy never looks finished
            rsync = ["rsync", "-a", "--exclude", "rank*.pt", "--exclude", "done.json"]
            subprocess.run([*rsync, f"{tout}/", f"{out}/"], check=True)
            subprocess.run(["cp", "-p", done, out / "done.json"], check=True)
    scores = ROOT / "results/lm-eval" / n
    for split, f in (("original_val", "original-val.json"), ("strat", "strat-v1.json")):
        if not (scores / f).exists():
            cmd = [
                py,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=1",
            ]
            cmd += ["-m", "allie.eval.score", "--checkpoint", out / "last.pt"]
            args = [*cmd, "--split", split, "--batch", 16]
            run(args, env, f"{log}.{split}.log", limit=7200)
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
    write(result, json.dumps(r, indent=2) + "\n")
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
    law (dmix.law) shifted through the base mean at the budget."""
    from allie.experiments import dmix
    from allie.experiments.isoflop import best_loss

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
        law = dmix.law(metric)
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
                    dmix.multiplier(curve, loss, c),
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


def main():
    # round files import this module by package path or, like configs/allie-2.0/round.py, as modelexp
    this = sys.modules[__name__]
    sys.modules["allie.experiments.modelexp"] = sys.modules["modelexp"] = this
    global ROUND
    cmd, ROUND, *rest = sys.argv[1:]
    ROUND = Path(ROUND).resolve()
    runpy.run_path(str(ROUND))
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


if __name__ == "__main__":
    main()
