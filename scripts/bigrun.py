"""Big-run launcher for either family (dense or MoE) on one 8-GPU node, in Slurm chunks.

  plan NAME --width W --layers L [--experts E --topk K --shared F] --tokens T [...]
      Freeze COMMIT's (default: this checkout's HEAD) trainer and golden evaluator and
      these tools into bigrun-v2/NAME with the clocked-pool month list and the full
      trainer argv (plan.json). Every trainer flag goes through modded_train's own
      argparse and schedule checks, the pool through the frozen chessmix index, and
      parameters are counted on fake tensors. --dry-run prints the plan and the command
      and keeps nothing.
  submit NAME [SBATCH ARGS]  sbatch NAME/bigrun.sbatch NAME, --signal from the margin
  task NAME                  one Slurm chunk (what bigrun.sbatch runs)
  eval NAME                  golden + original-val scores of last.pt -> result.json
  refresh NAME               re-copy these tools from this checkout (trainer stays frozen)

A chunk runs preflight.py, then supervises the trainer. After a crash, or a stall (none
of the run's jsonl or checkpoint files grows for --stall-min; for 1.5 worst-case saves
while a checkpoint dir of this attempt is unpublished; for --startup-min before the
attempt's first train row), it SIGTERMs the torchrun group, SIGKILLs what is left and
restarts from last.pt, giving up after --restarts failures in a row without a new
checkpoint. At deadline = job end - margin the ranks get SIGUSR1 (checkpoint, exit) and
the job requeues itself; margin = two checkpoint writes at --write-mbps (one may be in
flight) + 5 steps + 5 min. The worst rate defaults to the slowest logged save of a
>= 5 GB checkpoint under results/pretrain. Slurm's --signal USR1 does the same; SIGTERM
or `touch NAME/STOP` checkpoints and exits without requeue. Once every step is done the
chunk scores last.pt, or requeues when less than --eval-min is left or the eval timed
out (once); an eval that exits without its scores ends the chunk nonzero, unrequeued.
"""

import argparse
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from hashlib import sha256
from pathlib import Path

ROOT = Path("/home/yimingz3/src/allie")
HERE = ROOT / "results/recipe10x/bigrun-v2"
SELF = Path(__file__).resolve().parent
PRETRAIN, LMEVAL = ROOT / "results/pretrain", ROOT / "results/lm-eval"
G = Path("/data/group_data/dei-group/yimingz3/allie")
# validation rows only: training rows come from the chessmix stores
DATA = G / "lichess_tokens_v2"
OVERLAY = str(G / "envs/chessmix-overlay")
RUNTIME = G / "envs/modded-torch210-runtime.json"
# a Lichess month lives in the first store whose start it reaches; HF movetext has %clk
# from 2017-03-28, so earlier months are clockless and dropped
STORES = (
    (G / "data-v1", "2025-01"),
    (G / "data-v1-hist", "2023-01"),
    (G / "data-v1-hist2", "2017-04"),
)
FIRST, LAST = "2017-04", "2026-08"
EXT = [
    G / "ext-v1" / m
    for m in ("otb/20xx-broadcast", "otb/20xx-pgnmentor", "otb/20xx-twic")
    + ("engine/20xx-ccrl404", "engine/20xx-ccrl4040", "engine/20xx-tcec")
]
# the strat-eval-v1 golden month, which also holds the dev/test games
HELD_OUT = ("2026-07",)
# the data fork's per-month policy tokens per mix pass
PASS_TOKENS = ROOT / "results/recipe10x/data-v1-hist2"
KEEP = ("lm_data.py", "lm_checkpoint.py", "chessmix.py", "chess_vocab.py")
KEEP += ("board_encode.cpp", "move-table.json")
TOOLS = ("bigrun.py", "bigrun.sbatch", "preflight.py", "health.py")
ROWS, TOKENS_PER_STEP = 512, 512 * 1024
POLICY = "mover_rule+up4+noengine+otb_x4"
SHIP = {"board": "conv", "mlp": "swiglu", "key_offset": False}
ROUTER = {"moe_seq": 1e-3, "moe_init": 0.006, "moe_router_lr_mul": 0.1}
ROUTER |= {"moe_gamma": 1e-2, "moe_update": "prop", "moe_kernel": "scatter-dualgather"}
DATA_ARGS = ["--clock-feats", 3, "--input-lr-mul", 5.0]
DATA_ARGS += ["--aux-time", 0.2, "--aux-wdl", 0.2]
SCHEDULE = {"batch_rows": ROWS, "plateau": 4.0, "final_lr": 0.2}
SCHEDULE["decay_shape"] = "linear"
# the science's schedule knobs: the 3e16 baseline's steps as fractions of training
FRAC = {"warmup": 32 / 2274, "mtp": 64 / 2274, "split": 65 / 2274, "decay": 32 / 2274}
# kill whatever still runs this long before the job end (Slurm TERMs, KILLs 120 s later)
HARD = 150
# torch 2.10's default NCCL timeout (the trainer passes none): ranks 1.. wait in a
# barrier while rank 0 writes model.pt, bf16 weights at 2 B/param
NCCL_S, MODEL_BPP = 600, 2

# modded_train's own parser and post-parse checks on the frozen source, then parameters
# of its create_model on fake tensors: total, and active = total minus the routed weights
# a token skips
CHECK = r"""
import argparse, json, os, sys
src, argv, world = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3])
sys.path.insert(0, src)
import torch.distributed as dist
from torch._subclasses.fake_tensor import FakeTensorMode
import modded_medium as mm, modded_train as mt
from modded_moe import MoE
from modded_wsd import Schedule

class Parsed(Exception):
    pass

parse = argparse.ArgumentParser.parse_args
def grab(self, args=None, namespace=None):
    raise Parsed(parse(self, args, namespace))
argparse.ArgumentParser.parse_args, sys.argv = grab, ["modded_train.py", *argv]
try:
    mt.main()
except Parsed as e:
    a = e.args[0]
argparse.ArgumentParser.parse_args = parse
s = Schedule(**json.loads(a.wsd_schedule))
s.validate()
assert a.initial_batch_rows == s.batch_rows and a.keep_checkpoints >= 2
assert 0 < a.wsd_end_step <= a.steps and s.warmup_steps <= a.wsd_decay_start < a.steps
assert a.initial_batch_rows % (a.micro_batch * world) == 0
os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(29500 + os.getpid() % 500))
dist.init_process_group("gloo", rank=0, world_size=1)
dist.broadcast = lambda *_, **__: None
cfg = mm.Config(width=a.width, layers=a.layers, head_dim=a.head_dim, arch=json.loads(a.arch))
cfg.feats = a.clock_feats
with FakeTensorMode():
    m = mm.create_model(cfg, "cpu")
total = sum(p.numel() for p in m.parameters())
skip = sum((x.up.numel() + x.down.numel()) * (x.experts - x.topk) / x.experts
           for x in m.modules() if isinstance(x, MoE))
flags = sum(x.startswith("--") for x in argv)
print(json.dumps(dict(flags=flags, total=total, active=round(total - skip), args=vars(a))))
"""

# index the months with the frozen chessmix (buckets.json only, no shard reads) and
# resolve the policy's weights
SAMPLER = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from chessmix import Sampler
s = Sampler(sys.argv[2], months=sys.argv[3].split(","), pool_frac=float(sys.argv[4]),
            aux=True, feats=True, history=sys.argv[5] or None)
s._weights(0.0)
print(json.dumps(dict(buckets=len(s.codes), games=int(s.games.sum()))))
"""


class Run:
    proc = None  # the live torchrun agent
    why = None  # requeue | stop, once asked for


def sha(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def say(msg):
    print(f"{time.strftime('%m-%d %H:%M:%S')} {msg}", flush=True)


def append(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def jsonl(path):
    """Rows of a jsonl file a live writer may be appending to (torn lines skipped)."""
    rows = []
    with suppress(OSError):
        for line in path.read_text().splitlines():
            with suppress(ValueError):
                rows.append(json.loads(line))
    return [r for r in rows if isinstance(r, dict)]


def months(first, last):
    y, m = map(int, first.split("-"))
    while f"{y}-{m:02d}" <= last:
        yield f"{y}-{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def pool():
    """Clocked Lichess months minus the held-out one: (built dirs, unbuilt month names,
    built ext dirs). A month is built once its buckets.json and stats.json (published
    last) exist."""
    built = lambda d: (d / "stats.json").exists() and (d / "buckets.json").exists()
    store = lambda m: next(s for s, start in STORES if m >= start)
    want = [store(m) / m for m in months(FIRST, LAST) if m not in HELD_OUT]
    ext = [d for d in EXT if built(d)]
    return [d for d in want if built(d)], [d.name for d in want if not built(d)], ext


def pass_tokens(dirs, policy):
    """Policy tokens per mix pass over dirs: (sum, dirs the data fork left unpriced)."""
    if policy != POLICY:
        return 0.0, len(dirs)
    base = json.loads((PASS_TOKENS / "pass-tokens-base.json").read_text())["months"]
    table = {m: sum(g["pass_tokens"] for g in v.values()) for m, v in base.items()}
    for f in (PASS_TOKENS / "pass-tokens").glob("*.json"):
        table[f.stem] = json.loads(f.read_text())["total"]["pass_tokens"]
    unpriced = sum(d.name not in table for d in dirs)
    return sum(table.get(d.name, 0.0) for d in dirs), unpriced


def schedule(steps):
    warmup = max(1, round(FRAC["warmup"] * steps))
    mtp = max(1, round(FRAC["mtp"] * steps))
    split = max(mtp, round(FRAC["split"] * steps)) | 1
    s = SCHEDULE | {"warmup_steps": warmup, "mtp_steps": mtp, "split_step": split}
    return s, max(warmup, round(FRAC["decay"] * steps))


def frozen(commit):
    """{path in the study: bytes}: the commit's trainer and eval_strat.py, and these
    tools."""
    git = lambda *a: subprocess.check_output(["git", "-C", ROOT, *a])
    names = git("ls-tree", "-r", "--name-only", commit, "scripts").decode().split()
    keep = [n for n in names if re.fullmatch(r"scripts/modded_\w+\.py", n)]
    keep += [n for n in names if Path(n).name in KEEP]
    out = {f"source-ours/{Path(n).name}": git("show", f"{commit}:{n}") for n in keep}
    out["evaluator-ours/eval_strat.py"] = git("show", f"{commit}:scripts/eval_strat.py")
    return out | {f: (SELF / f).read_bytes() for f in TOOLS}


def arch(a):
    out = dict(SHIP)
    if a.experts:
        out |= {"moe": [a.experts, a.topk]} | ROUTER
        if a.shared == 0:
            out["moe_shared"] = False
        elif a.shared != 0.5:
            out["moe_shared_frac"] = a.shared
    return out


def train_args(a, study, steps, dirs, arch, every, eval_every):
    """modded_train arguments of the run, all but --max-seconds and --resume."""
    s, decay = schedule(steps)
    stores = [*(d for d, _ in STORES), *dict.fromkeys(d.parent for d in EXT)]
    args = [
        "--name", a.name, "--width", a.width, "--layers", a.layers,
        "--head-dim", a.head_dim, "--steps", steps,
        "--initial-batch-rows", ROWS, "--micro-batch", a.micro,
        "--lr-scale", a.lr_scale, "--seed", a.seed, "--eval-every", eval_every,
        "--checkpoint-every", every, "--keep-checkpoints", a.keep,
        "--val-rows", 1024, "--deterministic",
        "--data", DATA, "--mix", a.policy, "--mix-pool-frac", a.pool_frac,
        "--mix-stores", ",".join(map(str, stores)),
        "--mix-months", ",".join(map(str, dirs)),
        "--wsd-schedule", json.dumps(s, sort_keys=True), "--wsd-end-step", steps,
        "--wsd-decay-start", decay, *DATA_ARGS,
        "--arch", json.dumps(arch, sort_keys=True), "--ckpt", "eager",
    ]  # fmt: skip
    args += ["--mix-history", study / "history-counts.json"] * bool(a.history)
    return [str(x) for x in [*args, *shlex.split(a.extra)]]


def env_for(name, study):
    scratch = f"/scratch/yimingz3/allie/{name}"
    return {
        "ALLIE_PROJECT_ROOT": str(ROOT),
        "OMP_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "4",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_MODULE_LOADING": "LAZY",
        "NCCL_CUMEM_HOST_ENABLE": "0",
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_DEBUG": "WARN",
        # a collective stuck past torch's fixed 10-min NCCL timeout aborts its rank
        # (torchrun tears the group down, the supervisor restarts it) and dumps the
        # flight recorder next to the logs
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "3",
        "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000",
        "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
        "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": str(study / "logs/nccl-trace-rank"),
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTHONPATH": OVERLAY,
        "TORCHINDUCTOR_CACHE_DIR": f"{scratch}-inductor",
        "TRITON_CACHE_DIR": f"{scratch}-triton",
    }


def probe(code, *argv):
    env = os.environ | {"PYTHONPATH": OVERLAY}
    cmd = [sys.executable, "-c", code, *map(str, argv)]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    assert out.returncode == 0, f"rejected:\n{out.stderr[-3000:]}"
    return json.loads(out.stdout.strip().splitlines()[-1])


def label(p):
    n, s = p["params"], p["shape"]
    words = f"width {s['width']}, {s['layers']} layers"
    if not s["experts"]:
        return f"Dense {n['total'] / 1e9:.2f}B", words
    frac = s["shared"]
    shared = f"shared expert {frac:g} of the active width"
    shared = shared if frac else "no shared expert"
    kind = f"MoE {n['active'] / 1e9:.2f}B active / {n['total'] / 1e9:.2f}B total"
    return kind, f"{words}, {s['experts']} experts, top-{s['topk']}, {shared}"


def save_rates(min_gb=5):
    """[(MB/s, run, step)] of every logged save of a >= min_gb checkpoint under
    results/pretrain, sized by a surviving published dir of the same run. Smaller saves
    are dominated by fixed costs."""
    out = []
    for log in PRETRAIN.glob("*/checkpoints.jsonl"):
        rows = [r for r in jsonl(log) if r.get("directory") and r.get("seconds")]
        size = 0
        for d in (log.parent / r["directory"] for r in reversed(rows)):
            with suppress(OSError):
                if (d / "model.pt").exists():
                    size = sum(f.stat().st_size for f in d.iterdir())
                    break
        if size >= min_gb * 1e9:
            run = log.parent.name
            out += [(size / 1e6 / r["seconds"], run, r["step"]) for r in rows]
    return out


def estimates(a, total):
    """Checkpoint cadence, chunk margin, save and startup stall limits from size and NFS
    speed, planned at the worst rate and reported at the typical one too."""
    step_s, gb = TOKENS_PER_STEP / a.tok_s, a.bytes_per_param * total / 1e9
    est = {"tok_s": a.tok_s, "step_s": step_s, "ckpt_gb": gb}
    est |= {"write_mbps": a.write_mbps, "write_typ_mbps": a.write_mbps_typ}
    if not a.write_mbps:
        rates = sorted(save_rates())
        assert rates, "no logged saves >= 5 GB: pass --write-mbps"
        (mbps, run, step), median = rates[0], rates[len(rates) // 2][0]
        est |= {"write_mbps": mbps, "write_from": f"{run} step {step}"}
        est |= {"write_saves": len(rates), "write_median_mbps": median}
    worst, typ = gb * 1e3 / est["write_mbps"], gb * 1e3 / a.write_mbps_typ
    # Young/Daly: checkpoint every sqrt(2 x typical save time x MTBF)
    every = round(math.sqrt(2 * typ * a.mtbf_h * 3600) / step_s / 64) * 64
    every = a.ckpt_every or min(8192, max(256, every))
    # every rank reads the whole model.pt and its own optimizer shard
    load = (2 * a.gpus + a.bytes_per_param - 2) * total / 1e9
    est |= {"ckpt_s": worst, "ckpt_typ_s": typ, "ckpt_overhead": typ / (every * step_s)}
    est |= {"resume_gb": load, "model_pt_s": MODEL_BPP / a.bytes_per_param * worst}
    margin = lambda save: round(2 * save + 5 * step_s + 300)
    startup = lambda mbps: round(1200 + load * 1e3 / mbps)
    stall = a.stall_min * 60
    est |= {"margin_typ_s": margin(typ), "startup_typ_s": startup(a.write_mbps_typ)}
    est["save_typ_s"] = round(max(stall, 1.5 * typ))
    sup = {"margin_s": round(a.margin_min * 60) or margin(worst)}
    sup["startup_s"] = round(a.startup_min * 60) or startup(est["write_mbps"])
    sup |= {"stall_s": stall, "save_s": round(max(stall, 1.5 * worst))}
    sup |= {"restarts": a.restarts, "poll_s": 30, "grace_s": 90}
    sup |= {"min_chunk_s": a.min_chunk_min * 60, "eval_s": a.eval_min * 60}
    sup["preflight"] = not a.no_preflight
    return every, est, sup


def plan(a):
    assert re.fullmatch(r"[A-Za-z0-9_-]+", a.name), a.name
    study, out = HERE / a.name, PRETRAIN / a.name
    assert a.dry_run or not (study.exists() or out.exists()), "never overwrite a run"
    assert ROWS % (a.micro * a.gpus) == 0, "micro-batch x GPUs must divide 512 rows"
    steps = round(a.tokens / TOKENS_PER_STEP)
    dirs, missing, ext = pool()
    if missing and not (a.allow_missing or a.dry_run):
        sys.exit(
            f"{len(missing)} clocked months unbuilt ({', '.join(missing[:6])}, ...): "
            "the month list is frozen at plan (resume asserts it): wait, or plan "
            "with --allow-missing"
        )
    dirs += ext
    rev = ["git", "-C", SELF, "rev-parse", "--verify", f"{a.commit}^{{commit}}"]
    a.commit = subprocess.check_output(rev, text=True).strip()
    files = frozen(a.commit)
    if a.history:
        files["history-counts.json"] = Path(a.history).read_bytes()
    tmp = Path(tempfile.mkdtemp()) if a.dry_run else HERE / f".{a.name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        for rel, data in files.items():
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp / rel).write_bytes(data)
        src, ar = tmp / "source-ours", arch(a)
        run, tail = (a, study, steps, dirs, ar), ["--max-seconds", "1", "--resume", "x"]
        check = lambda args: probe(CHECK, src, json.dumps(args + tail), a.gpus)
        every, est, sup = estimates(a, check(train_args(*run, 1, 1))["total"])
        args = train_args(*run, every, a.eval_every or every)
        got = check(args)
        months = ",".join(map(str, dirs))
        info = probe(SAMPLER, src, a.policy, months, a.pool_frac, a.history)
    except BaseException:
        shutil.rmtree(tmp)
        raise
    info |= dict(zip(("pass_tokens", "unpriced"), pass_tokens(dirs, a.policy)))
    est["train_hours"] = steps * est["step_s"] / 3600
    s, decay = schedule(steps)
    shape = {"width": a.width, "layers": a.layers, "head_dim": a.head_dim}
    shape |= {"experts": a.experts, "topk": a.topk, "shared": a.shared}
    data = {"policy": a.policy, "pool_frac": a.pool_frac, "history": a.history}
    data |= {"months": list(map(str, dirs)), "missing": missing, "first_clocked": FIRST}
    p = {"name": a.name, "commit": a.commit, "created": time.strftime("%F %T")}
    p |= {"shape": shape, "arch": ar, "micro": a.micro, "gpus": a.gpus}
    p["params"] = {"total": got["total"], "active": got["active"]}
    p |= {"steps": steps, "tokens": steps * TOKENS_PER_STEP}
    p |= {"schedule": s, "decay_start": decay, "estimates": est, "supervise": sup}
    p["data"] = data | {"held_out": list(HELD_OUT)} | info
    p["env"] = env_for(a.name, study)
    p["args"] = args
    # as modded_train will see them, less the per-attempt --max-seconds and --resume
    got["args"].pop("max_seconds"), got["args"].pop("resume")
    p["parsed"] = got["args"]
    p["model"], p["architecture"] = label(p)
    describe(p, study, got["flags"])
    if a.dry_run:
        shutil.rmtree(tmp)
        return
    p["hashes"] = {rel: sha256(data).hexdigest() for rel, data in files.items()}
    (tmp / "logs").mkdir()
    (tmp / "plan.json").write_text(json.dumps(p, indent=2) + "\n")
    tmp.rename(study)
    say(f"planned {study}")


def describe(p, study, flags):
    e, d, s, sup = p["estimates"], p["data"], p["schedule"], p["supervise"]
    by = {}
    for m in d["months"]:
        by[Path(m).parent.name] = by.get(Path(m).parent.name, 0) + 1
    every = int(p["args"][p["args"].index("--checkpoint-every") + 1])
    print(
        f"Model: {p['model']} ({p['architecture']}), micro-batch {p['micro']} rows, "
        f"{p['gpus']} GPUs, trainer commit {p['commit']}"
    )
    print(
        f"Train: {p['tokens'] / 1e9:.2f}B tokens = {p['steps']} x {TOKENS_PER_STEP}; "
        f"warmup {s['warmup_steps']}, MTP {s['mtp_steps']}, split {s['split_step']}, "
        f"linear decay from {p['decay_start']} to {s['final_lr']}x; "
        f"~{e['train_hours']:.0f} h at {e['tok_s'] / 1e3:.0f}K tok/s"
    )
    print(
        f"Data: {d['policy']}, pool_frac {d['pool_frac']}, {len(d['months'])} dirs "
        f"({', '.join(f'{k} {v}' for k, v in by.items())}); held out "
        f"{' '.join(d['held_out'])}; clockless months before {d['first_clocked']} "
        f"dropped; {d['buckets']} buckets, {d['games'] / 1e9:.2f}B games"
    )
    if d["missing"]:
        print(f"  unbuilt clocked months ({len(d['missing'])}):", *d["missing"])
    if d["pass_tokens"]:
        passes = p["tokens"] / d["pass_tokens"]
        print(
            f"  one mix pass ~{d['pass_tokens'] / 1e9:.1f}B policy tokens "
            f"({d['unpriced']} dirs unpriced): {passes:.2f} passes"
        )
    src = "--write-mbps"
    if e.get("write_from"):
        src = (
            f"slowest of {e['write_saves']} logged saves >= 5 GB: {e['write_from']}; "
            f"median {e['write_median_mbps']:.0f} MB/s"
        )
    m = lambda x: f"{x / 60:.0f} min"
    print(
        f"Checkpoint: ~{e['ckpt_gb']:.0f} GB every {every} steps "
        f"({every * e['step_s'] / 3600:.1f} h); a save takes {m(e['ckpt_s'])} at "
        f"{e['write_mbps']:.1f} MB/s worst ({src}), {m(e['ckpt_typ_s'])} at "
        f"{e['write_typ_mbps']:.0f} MB/s typical ({e['ckpt_overhead']:.1%} of the "
        f"run); resume reads ~{e['resume_gb']:.0f} GB"
    )
    print(
        f"Supervise (worst / typical rate): checkpoint signal {m(sup['margin_s'])} / "
        f"{m(e['margin_typ_s'])} before the end; stall {m(sup['stall_s'])} without "
        f"growth, {m(sup['save_s'])} / {m(e['save_typ_s'])} "
        f"while a save is in flight; startup {m(sup['startup_s'])} / "
        f"{m(e['startup_typ_s'])}; {sup['restarts']} restarts in a row, preflight "
        f"{sup['preflight']}"
    )
    frac, gb = e["model_pt_s"] / NCCL_S, MODEL_BPP * p["params"]["total"] / 1e9
    print(
        f"{'WARNING: ' * (frac > 0.5)}model.pt ({gb:.1f} GB) takes "
        f"{e['model_pt_s']:.0f} s at the worst rate while the other ranks wait in an "
        f"NCCL barrier: {frac:.0%} of torch's default {NCCL_S} s timeout"
    )
    print("Env:", *(f"{k}={shlex.quote(v)}" for k, v in p["env"].items()))
    rt = json.loads(RUNTIME.read_text())["sha256"][:16]
    py = f"/scratch/yimingz3/allie/runtimes/torch210-{rt}/bin/python"
    last = PRETRAIN / p["name"] / "last.pt"
    cmd = trainer(p, study, py)(last, 172800 - sup["margin_s"])
    print(f"Command ({flags} flags accepted by modded_train; a resumed 2-day chunk):")
    print(shlex.join(cmd), flush=True)


def trainer(p, study, py):
    base = [py, "-m", "torch.distributed.run", "--standalone"]
    base += [f"--nproc_per_node={p['gpus']}", study / "source-ours/modded_train.py"]
    base = [str(x) for x in base + p["args"]]
    resume = lambda r: ["--resume", str(r)] if r else []
    return lambda r, seconds: [*base, "--max-seconds", str(seconds), *resume(r)]


def tree(pid):
    """Live descendants of pid."""
    out, todo = [], [pid]
    while todo:
        with suppress(OSError, ValueError):
            for t in Path(f"/proc/{todo.pop()}/task").iterdir():
                kids = list(map(int, (t / "children").read_text().split()))
                out += kids
                todo += kids
    return out


def alive(pid):
    with suppress(OSError, IndexError):
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    return False


def workers(pid):
    """The ranks under torchrun agent pid (whose own argv names modded_train.py too)."""
    out = []
    for c in tree(pid):
        with suppress(OSError):
            argv = Path(f"/proc/{c}/cmdline").read_bytes().split(b"\0")
            out += [c] * any(x.endswith(b"modded_train.py") for x in argv)
    return out


def stop(why):
    """Ask every rank to checkpoint and exit (the trainer checks every 5 steps)."""
    Run.why = Run.why or why
    for w in workers(Run.proc.pid) if Run.proc else ():
        with suppress(ProcessLookupError):
            os.kill(w, signal.SIGUSR1)


def on_signal(sig, _):
    stop("requeue" if sig == signal.SIGUSR1 else "stop")


def kill(proc, grace):
    """SIGTERM the torchrun group (the agent SIGTERMs, then SIGKILLs, its ranks), then
    SIGKILL whatever is left. True once every process seen is gone."""
    pids = {proc.pid, *tree(proc.pid)}
    with suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    with suppress(subprocess.TimeoutExpired):
        proc.wait(grace)
    for p in pids | set(tree(proc.pid)):
        with suppress(ProcessLookupError):
            os.kill(p, signal.SIGKILL)
    proc.wait()
    end = time.time() + 120
    while any(map(alive, pids)) and time.time() < end:
        time.sleep(1)
    return not any(map(alive, pids))


def unpublished(out):
    """Checkpoint dirs no checkpoints.jsonl row lists: saves in flight or crashed. The
    launcher never deletes anything under checkpoints/."""
    rows = jsonl(out / "checkpoints.jsonl")
    pub = {Path(r["directory"]).name for r in rows if "directory" in r}
    with suppress(OSError):
        return {d.name for d in (out / "checkpoints").glob("step-*")} - pub
    return set()


def activity(out):
    """Sizes of what a live trainer grows (its jsonl logs and checkpoint shards), and
    its unpublished checkpoint dirs."""
    sizes = [-1] * 4
    for i, f in enumerate(("train.jsonl", "validation.jsonl", "checkpoints.jsonl")):
        with suppress(OSError):
            sizes[i] = (out / f).stat().st_size
    with suppress(OSError):
        ckpts = (out / "checkpoints").iterdir()
        sizes[3] = sum(f.stat().st_size for d in ckpts for f in d.iterdir())
    return tuple(sizes), unpublished(out)


def attempt(cmd, env, log, out, deadline, hard, stop_file, sup):
    """One trainer run: (exit | crash | stall | stuck | requeue | stop, return code).
    A stall is nothing growing for startup_s before the attempt's first train row, for
    save_s while a checkpoint dir it made is unpublished (fsync adds no bytes), else for
    stall_s; the watchdog stays armed after the ranks are asked to stop."""
    last, old = activity(out)
    train0, changed, why = last[0], time.time(), None
    with open(log, "a") as f:
        f.write(f"=== {time.strftime('%F %T')} {shlex.join(cmd)}\n")
        f.flush()
        kw = {"env": env, "stdout": f, "stderr": subprocess.STDOUT}
        Run.proc = proc = subprocess.Popen(cmd, **kw, start_new_session=True)
    try:
        while proc.poll() is None:
            time.sleep(sup["poll_s"])
            now = time.time()
            if Run.why and now > hard:
                kill(proc, sup["grace_s"])
                break
            if not Run.why and now >= deadline:
                stop("requeue")
            elif not Run.why and stop_file.exists():
                stop("stop")
            seen, dirs = activity(out)
            if seen != last:
                last, changed = seen, now
            limit = "save_s" if dirs - old else "stall_s"
            if now - changed > sup["startup_s" if seen[0] == train0 else limit]:
                why = "stall" if kill(proc, sup["grace_s"]) else "stuck"
    finally:
        Run.proc = None
    why = why or Run.why or ("crash" if proc.returncode else "exit")
    return why, proc.returncode


def ckpt_step(out):
    rows = [r for r in jsonl(out / "checkpoints.jsonl") if "step" in r]
    return rows[-1]["step"] if rows else 0


def mtime(f):
    return f.stat().st_mtime_ns if f.exists() else 0


def stop_reason(out, before=None):
    """done.json's stop_reason, if written after mtime `before`."""
    with suppress(OSError, ValueError, KeyError):
        if mtime(out / "done.json") != before:
            return json.loads((out / "done.json").read_text())["stop_reason"]
    return None


def finished(out, steps):
    with suppress(OSError, ValueError):
        d = json.loads((out / "done.json").read_text())
        return d["stop_reason"] == "steps" and d["step"] == steps
    return False


def chunk(p, out, study, argv, env, deadline, hard, log):
    """Train until every step is done, the deadline or a stop: done | requeue | stop |
    failed | short (no time for even one attempt). Crashes and stalls restart."""
    sup, fails, mark, tried = p["supervise"], 0, ckpt_step(out), False
    stop_file = study / "STOP"
    while True:
        if finished(out, p["steps"]):
            return "done"
        if Run.why or stop_file.exists():
            return Run.why or "stop"
        left = deadline - time.time()
        if left < sup["min_chunk_s"]:
            return "requeue" if tried else "short"
        resume, tried = (out / "last.pt").exists(), True
        t0, step0, before = time.time(), ckpt_step(out), mtime(out / "done.json")
        cmd = argv(out / "last.pt" if resume else None, int(left))
        why, rc = attempt(cmd, env, log, out, deadline, hard, stop_file, sup)
        if why == "exit":  # a clean exit leaves a fresh done.json
            ends = {"steps": "exit", "wall_clock_cap": "requeue", "signal": "stop"}
            why = ends.get(stop_reason(out, before), "crash")
            why = "crash" if why == "exit" and not finished(out, p["steps"]) else why
        step = ckpt_step(out)
        row = {"job": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname()}
        row |= {"start": t0, "end": time.time(), "why": why, "rc": rc}
        row |= {"resumed": resume, "ckpt_before": step0, "ckpt_after": step}
        append(study / "attempts.jsonl", row)
        say(f"attempt ended: {why} (rc {rc}), checkpoint step {step0} -> {step}")
        if why in ("requeue", "stop", "stuck"):
            return Run.why or ("requeue" if why == "stuck" else why)
        if why in ("crash", "stall"):
            fails, mark = (1 if step > mark else fails + 1), step
            if lost := sorted(unpublished(out)):
                say(f"unpublished checkpoint dirs from crashed saves: {' '.join(lost)}")
            if fails > sup["restarts"]:
                say(f"{fails} failures in a row without a new checkpoint: giving up")
                return "failed"


def job_end():
    job = os.environ["SLURM_JOB_ID"]
    with suppress(subprocess.CalledProcessError, ValueError, OSError):
        cmd = ["squeue", "-h", "-o", "%L", "-j", job]
        days, _, hms = subprocess.check_output(cmd, text=True).strip().rpartition("-")
        s = 0
        for x in hms.split(":"):
            s = s * 60 + int(x)
        return time.time() + s + 86400 * int(days or 0)
    return float(os.environ["SLURM_JOB_END_TIME"])


def stage(study, env):
    cmd = [sys.executable, study / "source-ours/modded_runtime_stage.py"]
    return subprocess.check_output(cmd, env=env, text=True).strip()


def requeue(job):
    subprocess.run(["scontrol", "requeue", job], check=True)


def load(study):
    p = json.loads((study / "plan.json").read_text())
    bad = sorted(k for k, v in p["hashes"].items() if sha(study / k) != v)
    assert not bad, f"frozen files changed since plan: {bad}"
    return p


def task(name):
    study, out = HERE / name, PRETRAIN / name
    p = load(study)
    for s in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(s, on_signal)
    job, node, sup = os.environ["SLURM_JOB_ID"], socket.gethostname(), p["supervise"]
    end = job_end()
    env = os.environ | p["env"]
    py = stage(study, env)
    say(f"chunk: job {job} on {node}, {(end - time.time()) / 3600:.2f} h left")
    record = {"job": job, "node": node}
    if sup["preflight"]:
        pf = study / "logs" / f"preflight-{job}-{int(time.time())}.json"
        cmd = [py, study / "preflight.py", "--out", pf]
        if subprocess.run(cmd, env=env, check=False).returncode:
            say(f"preflight failed on {node} ({pf}): resubmit with --exclude={node}")
            append(study / "chunks.jsonl", record | {"why": "preflight"})
            sys.exit(3)
    log = study / "logs" / f"{name}.train.log"
    deadline, hard = end - sup["margin_s"], end - HARD
    why = chunk(p, out, study, trainer(p, study, py), env, deadline, hard, log)
    if why == "done" and end - time.time() < sup["eval_s"]:
        why = "requeue"
    elif why == "done" and (ev := evaluate(name, env, py, hard)) != "ok":
        why = f"eval-{ev}"
    # a timed-out eval gets one fresh chunk; a second timeout is a hang
    past = [r.get("why") for r in jsonl(study / "chunks.jsonl")]
    retry = why == "requeue" or why == "eval-timeout" and why not in past
    append(study / "chunks.jsonl", record | {"end": time.time(), "why": why})
    say(f"chunk end: {why}, checkpoint step {ckpt_step(out)}")
    if retry and not (study / "STOP").exists():
        requeue(job)
    elif not retry and why not in ("done", "stop"):
        sys.exit(f"chunk {why}: not requeued")


def evaluate(name, env=None, py=None, until=math.inf):
    """Golden and original-val scores of last.pt on one GPU, skipping scored splits;
    then result.json: ok, timeout (at `until`) or failed (an eval exited unscored)."""
    study, out = HERE / name, PRETRAIN / name
    p = load(study)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    env = (env or os.environ | p["env"]) | {"CUDA_VISIBLE_DEVICES": gpu}
    py = py or stage(study, env)
    ev, src = study / "evaluator-ours/eval_strat.py", study / "source-ours"
    for split, f in (("strat", "strat-v1.json"), ("original_val", "original-val.json")):
        if (LMEVAL / name / f).exists():
            continue
        cmd = [py, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1"]
        cmd += [ev, "--checkpoint", out / "last.pt", "--source", src, "--split", split]
        cmd = [str(x) for x in [*cmd, "--batch", 16]]
        log = study / "logs" / f"eval-{split}.log"
        t = max(60, until - time.time())
        with open(log, "a") as o:
            try:
                kw = {"env": env, "stdout": o, "stderr": o, "timeout": t}
                rc = subprocess.run(cmd, **kw, check=False).returncode
            except subprocess.TimeoutExpired:
                say(f"eval {split} timed out after {t:.0f} s: see {log}")
                return "timeout"
        if not (LMEVAL / name / f).exists():
            say(f"eval {split} exited {rc} without {f}: see {log}")
            return "failed"
    read = lambda f: json.loads((LMEVAL / name / f).read_text())
    ov, sv = read("original-val.json"), read("strat-v1.json")
    last = json.loads((out / "train.jsonl").read_text().splitlines()[-1])
    result = {k: p[k] for k in ("name", "model", "architecture", "params", "steps")}
    result |= {k: last[k] for k in ("tokens", "useful_training_flops")}
    result["strat"] = {k: sv[k] for k in ("macro", "expert_macro", "cells")}
    result["ce"] = {k: ov[k + "_ce"] for k in ("move", "expert2400", "expert2600")}
    (study / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    say(f"golden macro {sv['macro']:.4f}, expert macro {sv['expert_macro']:.4f}")
    return "ok"


def submit(name, extra):
    study = HERE / name
    p = load(study)
    cmd = ["sbatch", "--parsable", f"--job-name={name}"]
    cmd += [f"--output={study}/logs/%x-%j.out"]
    cmd += [f"--signal=B:USR1@{p['supervise']['margin_s']}", *extra]
    cmd += [str(study / "bigrun.sbatch"), name]
    say(shlex.join(cmd))
    job = subprocess.check_output(cmd, text=True).strip()
    append(study / "jobs.jsonl", {"at": time.time(), "job": job, "cmd": cmd})
    print(job)


def refresh(name):
    study = HERE / name
    p = json.loads((study / "plan.json").read_text())
    for f in TOOLS:
        shutil.copy2(SELF / f, study / f)
        p["hashes"][f] = sha(study / f)
    (study / "plan.json").write_text(json.dumps(p, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("plan").add_argument
    add("name")
    add("--width", type=int, required=True)
    add("--layers", type=int, required=True)
    add("--head-dim", type=int, default=64)
    add("--experts", type=int, default=0, help="routed experts; 0 = dense")
    add("--topk", type=int, default=4)
    add("--shared", type=float, default=0.5, help="shared expert's active-width share")
    add("--tokens", type=float, required=True, help="steps = tokens / 524288")
    add("--micro", type=int, default=64, help="rows per rank per micro-batch")
    add("--gpus", type=int, default=8)
    add("--seed", type=int, default=42)
    add("--lr-scale", type=float, default=1.0)
    add("--commit", default="HEAD", help="trainer source to freeze (this checkout's)")
    add("--policy", default=POLICY)
    add("--pool-frac", type=float, default=1.0)
    add("--history", default="", help="all-history bucket counts (default: stores')")
    add("--extra", default="", help="more trainer flags, shell-quoted (parser-checked)")
    add("--ckpt-every", type=int, default=0, help="0 = Young/Daly from the estimates")
    add("--eval-every", type=int, default=0, help="0 = validate at every checkpoint")
    add("--keep", type=int, default=2, help="checkpoints kept (plus best)")
    add("--tok-s", type=float, required=True, help="measured tokens/s of the node")
    add("--bytes-per-param", type=float, default=10.0, help="checkpoint bytes/param")
    worst = "worst NFS MB/s (0 = slowest logged save >= 5 GB): margin, stall limits"
    add("--write-mbps", type=float, default=0, help=worst)
    add("--write-mbps-typ", type=float, default=140.0, help="typical NFS MB/s: cadence")
    add("--mtbf-h", type=float, default=12.0, help="hours between crashes: cadence")
    add("--margin-min", type=float, default=0, help="0 = from the estimates")
    add("--stall-min", type=float, default=20)
    add("--startup-min", type=float, default=0, help="0 = 20 min + worst resume read")
    add("--restarts", type=int, default=3)
    add("--min-chunk-min", type=float, default=60, help="least time for a new attempt")
    add("--eval-min", type=float, default=90, help="least time left for the final eval")
    add("--no-preflight", action="store_true")
    add("--allow-missing", action="store_true", help="plan despite unbuilt months")
    add("--dry-run", action="store_true")
    for c in ("task", "eval", "refresh", "submit"):
        sub.add_parser(c).add_argument("name")
    a, rest = ap.parse_known_args()
    if rest and a.cmd != "submit":
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    match a.cmd:
        case "plan":
            plan(a)
        case "submit":
            submit(a.name, rest)
        case "task":
            task(a.name)
        case "eval":
            sys.exit(evaluate(a.name) != "ok")
        case "refresh":
            refresh(a.name)


if __name__ == "__main__":
    main()
