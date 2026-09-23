#!/usr/bin/env python3
"""runindex.py: read-only, incremental index of Allie runs for the runs dashboard.

  runindex.py compute        one queue snapshot (babel squeue / sinfo, the dei governor's
                             state, the orchard mirror's squeue) appended to compute.jsonl,
                             and the sacct GPU-hour cache (sacct.json)
  runindex.py index          study plans + run logs + scores + replays + the orchard mirror
                             -> runs.json, curves.json, sweep.json, compute.json
  runindex.py page OUT.html  scripts/runindex.html with the index embedded

Everything lands in results/runindex/. Reads small files only (plan.json, the run dir's
jsonl / json files, results/*.json, replay / perply json, train.log), growing files by
byte offset (state.json); never data shards or data/test*. Derived, not logged:
  MFU         d(useful_training_flops) / d(seconds) / (GPUs x dense BF16 datasheet peak)
  LR          the run's frozen WSD schedule (its study's source-ours/modded_wsd.py)
  ckpt stall  checkpoints.jsonl seconds: the training loop's block per save (it includes
              the wait for the previous async save)
  restarts    trainer starts that trained ('{"args"' metadata line, then a train row, in
              the run's train.log) - 1
  local CM    moe-knobs/knobs.py cm(): the frozen golden law through the paired control
  sweep law   scripts/isoflop_fit.py additive() per family, CMs by dmix_fit.multiplier()
"""

import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/yimingz3/src/allie")
R, P = ROOT / "results/recipe10x", ROOT / "results/pretrain"
OUT = ROOT / "results/runindex"
MIRROR = OUT / "orchard"
GOV = Path("/data/group_data/dei-group/yimingz3/allie/controller/dei-governor")
RACES = Path("/data/group_data/dei-group/yimingz3/allie/race")
PHASE = time.mktime(time.strptime("2026-09-21", "%Y-%m-%d"))  # studies / GPU-h since
# dense BF16 tensor FLOP/s (datasheets) and display names, by normalized GPU name
PEAK = {
    "h100": 989.4e12,
    "l40s": 362.05e12,
    "rtxpro6000": 503.8e12,
    "6000ada": 364.25e12,
    "a6000": 154.8e12,
}
GPUS = {
    "h100": "H100",
    "l40s": "L40S",
    "rtxpro6000": "RTX PRO 6000",
    "6000ada": "RTX 6000 Ada",
    "a6000": "A6000",
}
CAPS = {"general": 8, "preempt": 24, "dei-group": 16, "orchard": 32}
BUDGET = dict(model=570, data=300, sweep=230)  # GPU-h planning targets (soft)
TRACKS = (
    ("sweep", "sw"),
    ("model", "mk|qb|audit|perf|fixe|rhealth|racetest|moeperf|maia"),
    ("data", "dr|dc3|dt|perply"),
)
NODE_WEEK = 5e20  # training FLOPs of a node-week (moe-v1 ledger)
BATCH = 524288  # tokens per optimizer step
FMTS = ("bullet", "blitz", "rapid", "classical")
BANDS = ("<1400", "1400-2000", "2000-2400", ">=2400")
MIXFMTS = (
    "ultrabullet",
    "bullet",
    "blitz",
    "rapid",
    "classical",
    "correspondence",
    "other",  # no format tag; OTB games carry their own format digit
)
STALL = 1800  # s without a train row before a running run is flagged
RECHECK = 6 * 3600  # s between looks at a scored run's files
POINTS, LAYER_POINTS = 120, 32
CACHE = 2  # per-run cache format; a cache of another format is rebuilt from the files
ST, MODULES = {}, {}
KNOBS = LAWS = None


def fin(x, d=4):
    """x to d significant digits; None unless a finite number."""
    ok = isinstance(x, (int, float)) and math.isfinite(x)
    return float(f"{x:.{d}g}") if ok else None


def clean(x):
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def jload(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError, TypeError):
        return default


def save(p, x):
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(clean(x), separators=(",", ":"), allow_nan=False))
    tmp.replace(p)


def sh(*cmd):
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=120, check=True
    ).stdout


def stat(p):
    try:
        return os.stat(p)
    except (OSError, TypeError):
        return None


def tail(p, offs):
    """(reset, new complete lines) of a growing file since the offset kept in offs;
    reset when it shrank, vanished or was replaced, so the caller drops what it parsed
    before. Offsets live with what was parsed from them (a run's cache, state.json)."""
    s, key = stat(p), str(p)
    if s is None:
        return offs.pop(key, None) is not None, []
    ino, off = offs.get(key, (s.st_ino, 0))
    reset = ino != s.st_ino or s.st_size < off
    off = 0 if reset else off
    if s.st_size == off:
        return reset, []
    with open(p, "rb") as f:
        f.seek(off)
        b = f.read(s.st_size - off)
    n = b.rfind(b"\n") + 1
    offs[key] = (s.st_ino, off + n)
    return reset, b[:n].splitlines()


def last_lines(p, n, size=2048):
    try:
        with open(p, "rb") as f:
            f.seek(max(0, f.seek(0, 2) - size))
            return f.read().decode(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def parse(lines):
    for line in lines:
        if line.startswith(b"{"):
            try:
                yield json.loads(line)
            except ValueError:
                pass


def module(path, name):
    """A frozen study's own copy of a torch-free module (modded_arch, modded_wsd)."""
    key = str(path)
    if key not in MODULES:
        try:
            spec = importlib.util.spec_from_file_location(
                f"ri{len(MODULES)}{name}", path
            )
            MODULES[key] = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(MODULES[key])
        except Exception:
            MODULES[key] = None
    return MODULES[key]


def gpu_key(gpu):
    g = re.sub(r"[\s_-]", "", (gpu or "").lower())
    return next((k for k in PEAK if k in g), None)


def gres(s):
    """(GPU type, count) of a Slurm gres / tres string."""
    m = re.search(r"gpu(?::(\w*[A-Za-z]\w*))?[:=](\d+)", s or "")
    return (m[1], int(m[2])) if m else (None, 0)


def params(layers, width, dims):
    """Whole-model parameters (memmodel.Shape's count less 128 d: config.json's
    'parameters' to within a few scalars) and the active count (the routed experts a
    token skips removed); dims = modded_arch.moe_dims[:4] or None for dense."""
    L, d, V = layers, width, 2432
    H, nve = round(8 * d / 3 / 16) * 16, min(5, L // 2)
    e, k, routed, shared = dims or (0, 0, 0, 0)
    board = 32 * 13 * 9 + 2 * 32 * 32 * 9 + 8 * 32 + 32 * 32 + 544 * d
    rest = (3 + nve) * V * d + 64 * d + 64 + 2 * L + board + (L - 1) * e * d
    rest += (L + 2 * nve) * d // 4 + 3 * L + 5 + (-(3 * L + 5)) % 8
    mlp = 3 * d * H + (L - 1) * 3 * d * (shared + e * routed) if dims else L * 3 * d * H
    total = 4 * L * d * d + mlp + rest
    return total, total - skipped(layers, width, dims)


def skipped(layers, width, dims):
    return (layers - 1) * (dims[0] - dims[1]) * 3 * width * dims[2] if dims else 0


# ------------------------------------------------------------------ studies

LABELS = (
    (r"sweep-check3-(.+)", "check", "pre-launch check, babel ({})"),
    (r"sweep-checkh-(.+)", "check", "pre-launch check, orchard ({})"),
    (r"sweep-(\d+e\d+)-(.+)", "sweep", "sweep {} ({})"),
    (r"moe-knobs/r(\w+?)-\d+e\d+d?", "model", "model round {}"),
    (r"data-recipe-3e17w?", "data", "data 3e17 confirm"),
    (r"data-recipe-r(\w+?)(?:-.*)?", "data", "data round {}"),
    (r"data-recipe-(.+)", "data", "data {}"),
    (r"moe-v1-(.+)", "model", "moe-v1 {}"),
)
PLAN_DROP = ("months", "stores", "extra_args", "baseline", "group", "arch")


def label(key):
    """(track, round label) of a study key (its dir under results/recipe10x)."""
    for pat, track, fmt in LABELS:
        if m := re.fullmatch(pat, key):
            return track, fmt.format(*m.groups())
    return "other", key


def recipe(policy):
    return re.sub(r"table:([\w.]+?)-[0-9a-f]{12}", r"\1", policy) if policy else None


def plan_summary(d, key, plan, mtime):
    sb = (d / "run.sbatch").read_text() if (d / "run.sbatch").exists() else ""
    g = re.search(r"--gres=gpu:(?:([A-Za-z0-9_]+):)?(\d+)", sb)
    arr = re.search(r"--array=(\d+)-(\d+)", sb)
    arch = module(d / "modded_arch.py", "arch")
    runs = []
    for i, r in enumerate(plan["runs"]):
        a = r.get("arch") or {}
        try:  # experts, top-k, routed width, shared width; the declared switches
            dims = list(arch.moe_dims(r["width"], a)[:4]) if a.get("moe") else None
            full = arch.resolve(a)
        except Exception:
            dims, full = None, a
        share = full.get("moe_shared_frac", 0.5) if full.get("moe_shared", True) else 0
        x = {k: v for k, v in r.items() if k not in PLAN_DROP}
        x |= dict(i=i, dims=dims, moe=a.get("moe"), update=full.get("moe_update"))
        runs.append(x | dict(share=share, mround=full.get("moe_round")))
    track, rnd = label(key)
    tasks = int(arr[2]) - int(arr[1]) + 1 if arr else len(runs)
    return dict(
        key=key,
        wave=plan.get("wave"),
        purpose=plan.get("purpose"),
        commit=plan.get("commit"),
        mtime=mtime,
        track=track,
        round=rnd,
        gpu=g and g[1],
        gpus=g and int(g[2]),
        pack=max(1, math.ceil(len(runs) / tasks)),
        runs=runs,
    )


def studies():
    """Study plans frozen since PHASE (superseded / moved / never-run copies skipped),
    re-parsed only when plan.json changes."""
    out = {}
    for p in sorted([*R.glob("*/plan.json"), *R.glob("moe-knobs/*/plan.json")]):
        key = str(p.parent.relative_to(R))
        s = stat(p)
        if re.search(r"superseded|moved|never-ran|\.bak|\.old", key) or not s:
            continue
        if s.st_mtime < PHASE:
            continue
        sig = [s.st_ino, s.st_size, s.st_mtime_ns]
        if (ST["plans"].get(key) or {}).get("sig") != sig:
            x = jload(p)
            if not x or "runs" not in x:
                continue
            ST["plans"][key] = plan_summary(p.parent, key, x, s.st_mtime) | dict(
                sig=sig
            )
        out[key] = ST["plans"][key]
    return out


# ------------------------------------------------------------------ per-run ingest

TRAIN_KEYS = (
    "step", "train_ce", "tokens", "tokens_per_second", "sampler_wait", "seconds",
    "useful_training_flops", "max_memory_gb", "time_ce", "wdl_ce", "moe_imbalance",
    "moe_min_load", "moe_starved", "moe_bias_range", "moe_margin",
)  # fmt: skip
SMALL = (
    "config.json", "resume-config.json", "done.json", "orchard.json", "result",
    "failed", "replay", "perply",
)  # fmt: skip
CFG_ARGS = (
    "wsd_schedule", "wsd_decay_start", "wsd_end_step", "lr_scale", "steps",
    "stop_after", "micro_batch", "row_tokens", "checkpoint_every", "max_seconds",
)  # fmt: skip


def add(rows, row):
    """Append a row; a step at or below the last one is a resumed trainer relogging, so
    the rows it replaces go."""
    while rows and rows[-1]["step"] >= row["step"]:
        rows.pop()
    rows.append(row)


def small_cfg(x):
    a = x.get("args", {})
    return dict(
        world_size=x.get("world_size"),
        parameters=x.get("parameters"),
        job_id=x.get("job_id"),
        args={k: a.get(k) for k in CFG_ARGS},
    )


def sources(pr, study, perply):
    """The files a run's record is derived from."""
    d, name, s = P / pr["name"], pr["name"], R / study["key"]
    f = {k: d / k for k in ("train.jsonl", "validation.jsonl", "checkpoints.jsonl")}
    f |= {k: d / k for k in SMALL[:4]}
    f["train.log"] = s / "logs" / f"{name}.train.log"
    f["failed"] = s / "logs" / f"{name}.failed.json"
    f["result"] = s / "results" / f"{name}.json"
    f["mirror"] = MIRROR / "studies" / Path(study["key"]).name / f"{name}.train.log"
    f["replay"] = next(
        (
            x
            for n in ("", "p")
            if (x := s / "replay" / f"run{pr['i']}{n}.json").exists()
        ),
        None,
    )
    f["perply"] = next(
        (x for x in (p / f"{name}.json" for p in perply) if x.exists()), None
    )
    return f


def started(c, meta, row):
    """Count trainer incarnations that trained: a metadata line, then a train row."""
    if meta:
        c["armed"] = True
    elif row and c.get("armed"):
        c["starts"], c["armed"] = c.get("starts", 0) + 1, False


def ingest(c, f):
    """Bring cache c up to date with the run's files f. The orchard mirror's train.log
    stands in for the run dir until the hand-back (orchard.json + done.json) lands.
    incof maps a train row's step to its trainer incarnation (train.log metadata lines)."""
    handed = f["orchard.json"].exists() and f["done.json"].exists()
    src = "orchard" if f["mirror"].exists() and not handed else "babel"
    if c.get("v") != CACHE or c.get("src") != src:
        c.clear()
        c.update(v=CACHE, src=src, off={}, train=[], val=[], ckpt=[], incof={})
        c.update(starts=0, armed=False, metas=0)
    offs, incof = c["off"], c["incof"]
    if src == "babel":
        for key, name in (
            ("train", "train.jsonl"),
            ("val", "validation.jsonl"),
            ("ckpt", "checkpoints.jsonl"),
        ):
            reset, lines = tail(f[name], offs)
            if reset:
                c[key] = []
            for x in parse(lines):
                if "step" in x:
                    add(
                        c[key],
                        {k: x[k] for k in TRAIN_KEYS if k in x}
                        if key == "train"
                        else x,
                    )
        reset, lines = tail(f["train.log"], offs)
        if reset:
            incof.clear()
            c.update(starts=0, armed=False, metas=0)
        for line in lines:
            meta, row = line.startswith(b'{"args"'), b'"train_ce"' in line
            started(c, meta, row)
            c["metas"] += meta
            if row and (m := re.search(rb'"step": (\d+)', line)):
                incof[m[1].decode()] = c["metas"]
    else:
        reset, lines = tail(f["mirror"], offs)
        if reset:
            incof.clear()
            c.update(train=[], val=[], ckpt=[], starts=0, armed=False, metas=0)
        for x in parse(lines):
            started(c, "args" in x, "train_ce" in x)
            if "args" in x:
                c["metas"] += 1
                c["cfg" if c["metas"] == 1 else "rcfg"] = small_cfg(x)
            elif "train_ce" in x:
                add(c["train"], {k: x[k] for k in TRAIN_KEYS if k in x})
                incof[str(x["step"])] = c["metas"]
            elif "move_ce" in x and "step" in x:
                add(c["val"], x)
            elif "durable_seconds" in x:
                add(c["ckpt"], x)
    names = ("cfg", "rcfg", "done", "orchard", "result", "failed", "replay", "perply")
    for key, k in zip(SMALL, names):
        if src == "orchard" and k in ("cfg", "rcfg"):
            continue
        s = stat(f[key])
        x = jload(f[key]) if s else None
        c[k] = small_cfg(x) if x and k in ("cfg", "rcfg") else x
        c[k + "_t"] = s and s.st_mtime
    s = stat(f["train.jsonl" if src == "babel" else "mirror"])
    c["last_t"] = s and s.st_mtime


def signature(f):
    return [
        [s.st_ino, s.st_size, s.st_mtime_ns] if (s := stat(p)) else None
        for p in f.values()
    ]


# ------------------------------------------------------------------ derived record


def thin(n, k=POINTS):
    """Indices of at most k of n points, first and last kept."""
    return (
        list(range(n))
        if n <= k
        else sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    )


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return statistics.median(xs) if xs else None


def pct(xs, q):
    xs = sorted(x for x in xs if isinstance(x, (int, float)) and math.isfinite(x))
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def lr_curve(pr, c, study):
    """LR multiplier over the run's frozen WSD schedule (its config args, else the
    plan), evaluated by its study's modded_wsd.Schedule at 80 steps."""
    wsd = module(R / study["key"] / "source-ours/modded_wsd.py", "wsd")
    a = (c.get("cfg") or {}).get("args") or {}
    sched = (
        json.loads(a["wsd_schedule"]) if a.get("wsd_schedule") else pr.get("schedule")
    )
    if not (wsd and sched):
        return []
    start = a.get("wsd_decay_start", pr.get("decay_start"))
    end = a.get("wsd_end_step") or pr["steps"]
    scale = a.get("lr_scale") or pr.get("lr") or 1
    try:
        s = wsd.Schedule(**sched)
        xs = {
            1,
            s.warmup_steps,
            start,
            end,
            *(round(1 + i * (end - 1) / 79) for i in range(80)),
        }
        return [
            [x, fin(scale * s.lr(x - 1, start, end))]
            for x in sorted(xs)
            if 1 <= x <= end
        ]
    except Exception:
        return []


def moe_row(x, experts):
    st = x.get("moe_starved")
    if not st or not experts:
        return None
    return dict(
        starved=sum(st) / (experts * len(st)),
        imb=max(x["moe_imbalance"]),
        minload=min(x["moe_min_load"]),
        bias=max(x["moe_bias_range"]),
        margin=statistics.fmean(x["moe_margin"]),
    )


def health_of(mr, alerts):
    """Starvation / balance at step 500 and now, by step parity (the router's Adam
    steps on odd steps only), plus the sweep's step-500 bar: > 5% starved or rising."""
    near = lambda par, s0: min(
        (p for p in mr if p[0] % 2 == par), key=lambda p: abs(p[0] - s0), default=None
    )
    at = {k: near(par, 500) for k, par in (("even", 0), ("odd", 1))}
    at = {k: v if v and abs(v[0] - 500) <= 50 else None for k, v in at.items()}
    now = {k: near(par, mr[-1][0]) for k, par in (("even", 0), ("odd", 1))}
    after = [m for s, m in mr if s >= 500]
    s500 = max((v[1]["starved"] for v in at.values() if v), default=None)
    slast = max(v[1]["starved"] for v in now.values() if v)
    rising = s500 is not None and mr[-1][0] > 750 and slast > max(0.01, 1.5 * s500)
    if mr[-1][0] >= 500 and slast > 0.05:
        alerts.append(
            dict(kind="starved", text=f"{100 * slast:.1f}% of routed experts starved")
        )
    if rising:
        alerts.append(
            dict(
                kind="starved",
                text=f"starvation rising: {100 * s500:.1f}% at step 500, {100 * slast:.1f}% now",
            )
        )
    row = lambda v: v and dict(step=v[0], **v[1])
    return dict(
        at500={k: row(v) for k, v in at.items()},
        now={k: row(v) for k, v in now.items()},
        peak_starved=max((m["starved"] for m in after), default=None),
        peak_imb=max((m["imb"] for m in after), default=None),
        rising=rising,
        reached=mr[-1][0] >= 500,
    )


def derive(pr, c, study):
    """The part of a run's record that depends on its files only, and its curves."""
    tr, va, ck = c.get("train") or [], c.get("val") or [], c.get("ckpt") or []
    cfg, steps = c.get("cfg") or {}, pr["steps"]
    total, active = params(pr["layers"], pr["width"], pr["dims"])
    planned = not cfg.get("parameters")
    if not planned:
        total = cfg["parameters"]
        active = total - skipped(pr["layers"], pr["width"], pr["dims"])
    last = tr[-1] if tr else None
    # rates come from windows inside one trainer incarnation (a resume restores the
    # cumulative seconds and FLOPs), the latest one, on the hardware the record names; a
    # row of unknown incarnation counts for none, a window under half the median rate
    # holds a compile or restart. No train.log and never resumed: one incarnation.
    tags = c.get("incof") or {}
    one = not tags and not c.get("rcfg")
    inc = lambda x: 0 if one else tags.get(str(x["step"]))
    mine = [x for x in tr if inc(x) is not None and inc(x) == inc(last)] if last else []
    rate = med(x["tokens_per_second"] for x in mine)
    fast = lambda x: rate is None or x["tokens_per_second"] >= 0.5 * rate
    steady = [x for x in mine if x["step"] > steps / 4 and fast(x)] or mine[2:] or mine
    wins = [
        (a, b)
        for a, b in zip(tr, tr[1:])
        if inc(a) is not None
        and inc(a) == inc(b)
        and b["seconds"] > a["seconds"]
        and fast(b)
    ]
    fps = {
        b["step"]: (b["useful_training_flops"] - a["useful_training_flops"])
        / (b["seconds"] - a["seconds"])
        for a, b in wins
    }
    stall = [x.get("seconds", 0) for x in ck]  # save()'s timer spans its flush wait too
    ces = [x.get("train_ce") for x in tr]
    alerts = []
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in ces):
        alerts.append(dict(kind="nan", text="non-finite train CE"))
    warm = (pr.get("schedule") or {}).get("warmup_steps", 0)
    spikes = []
    for i in range(8, len(tr)):
        m = med(ces[i - 8 : i])
        if tr[i]["step"] > 2 * warm and m is not None and (ces[i] or 0) - m > 0.1:
            spikes.append((ces[i] - m, tr[i]["step"]))
    for dv, s in sorted(spikes, reverse=True)[:3]:
        alerts.append(
            dict(kind="spike", step=s, text=f"train CE spike +{dv:.2f} at step {s}")
        )
    late = [v for x, v in zip(tr, ces) if x["step"] > 0.1 * steps and v]
    if (
        len(late) > 6
        and last["step"] > 0.2 * steps
        and med(late[-4:]) - min(late) > 0.15
    ):
        alerts.append(
            dict(
                kind="diverging",
                text=f"train CE {med(late[-4:]) - min(late):+.2f} above its low",
            )
        )
    E = (pr["moe"] or [0])[0]
    mr = [(x["step"], m) for x in tr if (m := moe_row(x, E))]
    health = health_of(mr, alerts) if mr else None
    sw = [x.get("sampler_wait") for x in steady]
    if (m := med(sw)) is not None and m > 0.1:
        alerts.append(
            dict(
                kind="data-bound", text=f"sampler_wait {m:.2f} of step time: data-bound"
            )
        )
    val = va[-1] if va else None
    failed = c.get("failed") or []
    rec = dict(
        params_total=total,
        params_active=active,
        params_planned=planned,
        world=cfg.get("world_size") or (c.get("orchard") or {}).get("gpus"),
        job_ids=[
            j for j in (cfg.get("job_id"), (c.get("rcfg") or {}).get("job_id")) if j
        ],
        step=last and last["step"],
        tokens=last and last["tokens"],
        train_ce=last and fin(last["train_ce"]),
        seconds=last and last["seconds"],
        starts=c.get("starts") or (1 if tr else 0),
        failures=len(failed),
        fail_reason=failed and failed[-1].get("reason"),
        done=c.get("done"),
        t_start=c.get("cfg_t"),
        t_done=c.get("done_t"),
        t_last=c.get("last_t"),
        src=c.get("src"),
        orchard=bool(c.get("orchard")) or c.get("src") == "orchard",
        eff=dict(
            tps=med(x["tokens_per_second"] for x in steady),
            fps=med(fps.get(x["step"]) for x in steady),
            mem=max((x.get("max_memory_gb", 0) for x in tr), default=None),
            sw=med(sw),
            sw90=pct(sw, 0.9),
            swlast=last and last.get("sampler_wait"),
            ckpt_n=len(ck),
            stall=sum(stall),
            stall_max=max(stall, default=None),
            stall_frac=sum(stall) / last["seconds"]
            if last and last["seconds"]
            else None,
            durable=med(x.get("durable_seconds") for x in ck),
            window=med(b["seconds"] - a["seconds"] for a, b in wins),
        ),
        val=val
        and dict(
            step=val["step"],
            move=val.get("move_ce"),
            e2400=val.get("expert2400_ce"),
            e2600=val.get("expert2600_ce"),
        ),
        health=health,
        alerts=alerts,
    )
    pick = [tr[i] for i in thin(len(tr))]
    col = lambda k, d=4: [fin(x.get(k), d) for x in pick]
    curve = dict(
        s=[x["step"] for x in pick],
        tok=[fin(x["tokens"] / 1e9) for x in pick],
        ce=col("train_ce"),
        tce=col("time_ce"),
        wce=col("wdl_ce"),
        tps=col("tokens_per_second", 3),
        fps=[fin(fps.get(x["step"]), 3) for x in pick],
        sw=col("sampler_wait", 3),
        mem=col("max_memory_gb", 3),
        val=[
            [
                x["step"],
                fin(x.get("move_ce")),
                fin(x.get("expert2400_ce")),
                fin(x.get("expert2600_ce")),
            ]
            for x in va
        ],
        lr=lr_curve(pr, c, study),
        ckpt=[
            [
                x["step"],
                fin(x.get("seconds", 0), 3),
                fin(x.get("durable_seconds"), 3),
            ]
            for x in ck
        ],
    )
    if mr:
        by = dict(mr)
        curve["moe"] = {
            k: [fin((by.get(x["step"]) or {}).get(k)) for x in pick]
            for k in ("starved", "imb", "minload", "bias", "margin")
        }
        li = [tr[i] for i in thin(len(tr), LAYER_POINTS)]
        curve["layers"] = dict(
            s=[x["step"] for x in li],
            imb=[[fin(v, 3) for v in x.get("moe_imbalance") or []] for x in li],
            starved=[x.get("moe_starved") or [] for x in li],
        )
    if (res := c.get("result")) and "strat" in res:
        s = res["strat"]
        rec["res"] = dict(
            macro=s["macro"],
            expert=s["expert_macro"],
            B=(s["macro"] + s["expert_macro"]) / 2,
            cells=s["cells"],
            ce=res.get("ce"),
            gpu=res.get("gpu_name"),
            node=res.get("node"),
            job=res.get("job"),
            flops=res.get("useful_training_flops"),
            t=c.get("result_t"),
        )
    if pp := c.get("perply"):
        rec["perply"] = {
            k: {b: v["ce"] for b, v in pp[k].items()}
            for k in ("bins", "expert", "rest")
            if k in pp
        }
    if rp := c.get("replay"):
        rec["replay"] = replay_of(rp)
    return rec, curve


def replay_of(rp):
    """The sampler replay (replay.py): repeats, passes and token share by format x band."""
    cells = {}
    for k, v in rp.get("cells", {}).items():
        f, b = k.split("/")
        name = f"{MIXFMTS[int(f)] if f.isdigit() else f}/{BANDS[int(b)]}"
        cells[name] = dict(
            passes=fin(v["passes"]),
            share=fin(v["share"]),
            late=fin(v.get("late_passes")),
            rep=fin(v["repeats"] / v["seen"]) if v.get("seen") else None,
        )
    keep = (
        "repeats",
        "seen",
        "max_passes",
        "wrapped_buckets",
        "infeasible",
        "repeats_observed_only",
        "phases",
    )
    return {k: rp.get(k) for k in keep} | dict(cells=cells)


def recipe_plan(study, policy):
    """The recipe table's planned passes / token shares by format x band (a phased
    policy's first table)."""
    for t in re.findall(r"table:([\w.]+-[0-9a-f]{12})", policy or ""):
        x = jload(R / study["key"] / "recipes" / f"{t}.json")
        if not (x and "summary" in x):
            continue
        s = x["summary"]
        cells = {
            f"{f}/{b}": dict(passes=fin(v["passes"]), share=fin(v["share"]))
            for f, fv in s.get("formats", {}).items()
            for b, v in fv.get("cells", {}).items()
        }
        keep = (
            "engine_share",
            "otb_share",
            "max_passes",
            "unique_tokens",
            "drawn_tokens",
        )
        return dict(table=t, cells=cells) | {k: s.get(k) for k in keep}
    return None


# ------------------------------------------------------------------ queues -> runs


def race_origin(script):
    """The study script a race copy runs (race_submit.sh records it first in sha256)."""
    d = Path(script).parent
    if d.parent != RACES:
        return script
    if str(d) not in ST["races"]:
        try:
            ST["races"][str(d)] = (
                (d / "sha256").read_text().split("\n")[0].split(None, 1)[1]
            )
        except (OSError, IndexError):
            ST["races"][str(d)] = script
    return ST["races"][str(d)]


def job_runs(snap, plans):
    """{run name: [job, ...]} for our queued / running jobs: babel by the job's script
    (race copies resolved to the study's) and array task, orchard by wave (its job name,
    a study mirrored there first) and task; and {job id: job}."""
    by_dir = {str((R / k).resolve()): p for k, p in plans.items()}
    root = R.resolve()
    mirrored = {d.name for d in (MIRROR / "studies").glob("*")}
    out, ids = {}, {}

    def put(p, tasks, j):
        runs = p["runs"]
        if tasks is None:
            names = [r["name"] for r in runs] if len(runs) == 1 else []
        else:
            names = [
                r["name"]
                for t in tasks
                for r in runs[t * p["pack"] : (t + 1) * p["pack"]]
            ]
        for n in names:
            out.setdefault(n, []).append(j)

    for j in (snap or {}).get("babel", {}).get("jobs", []):
        ids[j["id"]] = j
        d = Path(race_origin(j.get("cmd") or "/")).resolve().parent
        while root in d.parents and str(d) not in by_dir:
            d = d.parent
        if str(d) in by_dir:
            put(by_dir[str(d)], j["tasks"], j)
    for j in (snap or {}).get("orchard", {}).get("jobs", []):
        cands = sorted(
            (p for p in plans.values() if p["wave"] == j["name"]),
            key=lambda p: -p["mtime"],
        )
        cands = [p for p in cands if Path(p["key"]).name in mirrored] or cands
        if cands:
            put(cands[0], j["tasks"], j)
    return out, ids


def orun_events():
    """Per run, from the mirrored orun job logs: jobs, first / last timestamp, done, and
    the tracebacks in its job logs."""
    ev, line = {}, re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (\S+) (.*)$", re.M)
    for f in sorted((MIRROR / "logs").glob("*.out")):
        job = re.sub(r"^.*?-(\d+(?:_\d+)?)\.out$", r"\1", f.name)
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        names = set()
        for m in line.finditer(text):
            t = time.mktime(time.strptime(m[1], "%Y-%m-%d %H:%M:%S"))
            e = ev.setdefault(m[2], dict(jobs=[], t0=t, t1=t, done=False, errors=0))
            e["t1"] = t
            e["jobs"] += [job] if job not in e["jobs"] else []
            e["done"] |= (
                "handed off" in m[3]
                or m[3].startswith("published")
                and m[3].endswith(" done")
            )
            names.add(m[2])
        for n in names:
            ev[n]["errors"] += text.count("Traceback")
    return ev


def remote_mtimes():
    """{run name: orchard mtime} of the train.logs the mirror listed."""
    out = {}
    try:
        for line in (MIRROR / "files.txt").read_text().splitlines():
            path, _, mt = line.rsplit(" ", 2)
            if path.endswith(".train.log"):
                out[Path(path).name[: -len(".train.log")]] = float(mt)
    except (OSError, ValueError):
        pass
    return out


# ------------------------------------------------------------------ the index


def status(pr, r, jobs, now):
    done, legs = r["done"] or {}, pr.get("stop_after") or []
    why = done.get("stop_reason")
    finished = (
        why == "steps"
        or bool(legs)
        and why == "stop_after"
        and done.get("step", 0) >= max(legs)
    )
    if r.get("res"):
        return "scored"
    if finished or r.get("orchard_done"):
        return "scoring"
    states = {j["state"] for j in jobs}
    if "RUNNING" in states:
        return "running"
    if "PENDING" in states:  # requeued or waiting, even if its log moved lately
        return "queued"
    if r["t_last"] and now - r["t_last"] < STALL:  # a job we could not map
        return "running"
    if r["failures"] >= 3:
        return "failed"
    if r["step"]:
        return "stopped"
    return "planned" if now - r["_mtime"] < 86400 else "not run"


def family(pr):
    """The declared design: S16 = 256 experts, top-16, shared 1/4, widths x16."""
    if not pr["moe"]:
        return "dense"
    s16 = pr["moe"] == [256, 16] and pr["share"] == 0.25 and pr["mround"] == 16
    return "S16" if s16 else "MoE (older)"


def words(pr, r):
    """'MoE 0.132B active / 0.93B total' and the architecture in words."""
    size = lambda n: f"{n / 1e9:.3g}B"
    arch = f"width {pr['width']}, {pr['layers']} layers"
    if not pr["moe"]:
        return f"Dense {size(r['params_total'])}", arch
    (e, k), frac = pr["moe"], pr["share"]
    names = ((0.25, "1/4"), (0.5, "1/2"), (1 / 3, "1/3"))
    sh = (
        next((s for v, s in names if abs(frac - v) < 1e-6), f"{frac:.0%}")
        if frac
        else "none"
    )
    model = f"MoE {size(r['params_active'])} active / {size(r['params_total'])} total"
    return model, f"{arch}, {e} experts, top-{k}, shared expert {sh}"


def control_of(r, runs):
    """The paired control. A data arm: the 'control' arm with the same budget, seed, pool
    tag, shape, steps and MoE design (its own study's, else the latest such study planned
    no later). A model arm: its reference in knobs.py CONTRASTS (rC -> C, rbase -> base,
    ...), else the rung's base, of the same budget and seed (own study first). A sweep
    S16 run: the dense run of the same shape."""
    same = lambda x, keys: all(x[k] == r[k] for k in keys)
    if r["track"] == "data" and r["v"] != "control":
        keys = ("budget", "seed", "tag", "layers", "width", "steps", "dims", "update")
        c = [
            x
            for x in runs
            if x["track"] == "data" and x["v"] == "control" and same(x, keys)
        ]
    elif r["track"] == "model" and r["v"] not in ("base", "dense"):
        pairs = KNOBS.CONTRASTS if KNOBS else ()
        ref = next((y for x, y in pairs if x == r["v"]), "base")
        c = [
            x
            for x in runs
            if x["track"] == "model" and x["v"] == ref and same(x, ("budget", "seed"))
        ]
    elif r["track"] == "sweep" and r["family"] != "dense":
        c = [
            x
            for x in runs
            if x["study"] == r["study"]
            and x["family"] == "dense"
            and same(x, ("layers", "width", "seed"))
        ]
    else:
        c = []
    own = [x for x in c if x["study"] == r["study"]]
    older = sorted(
        (x for x in c if x["_mtime"] <= r["_mtime"]), key=lambda x: -x["_mtime"]
    )
    pick = own or older or sorted(c, key=lambda x: -x["_mtime"])
    return pick[0]["name"] if pick else None


def local_cm(field, ctl, loss, budget):
    key = f"{field}|{ctl:.6f}|{loss:.6f}|{budget}"
    if key not in ST["cm"]:
        try:
            ST["cm"][key] = fin(KNOBS.cm(LAWS[field], ctl, loss, float(budget)))
        except Exception:
            ST["cm"][key] = None
    return ST["cm"][key]


def deltas(runs):
    """Paired deltas vs the control (B, macro, expert, 16 cells, bands, formats,
    per-ply) and the local law-conditional CM."""
    by = {r["name"]: r for r in runs}
    for r in runs:
        r["control"] = control_of(r, runs)
        c, x = by.get(r["control"]), r.get("res")
        if not (x and c and c.get("res")):
            continue
        y = c["res"]
        d = {k: x[k] - y[k] for k in ("macro", "expert", "B")}
        d["cells"] = {
            k: v - y["cells"][k] for k, v in x["cells"].items() if k in y["cells"]
        }
        d["better"] = sum(v < 0 for v in d["cells"].values())
        pick = lambda f: [v for k, v in d["cells"].items() if f(k)]
        d["bands"] = {
            b: statistics.fmean(pick(lambda k: k.endswith("/" + b)) or [math.nan])
            for b in BANDS
        }
        d["formats"] = {
            f: statistics.fmean(pick(lambda k: k.startswith(f + "/")) or [math.nan])
            for f in FMTS
        }
        if KNOBS:
            d["cm"] = {
                f: local_cm(f, y[k], x[k], r["budget"])
                for f, k in (("macro", "macro"), ("expert_macro", "expert"))
            }
        a, b = r.get("perply"), c.get("perply")
        if a and b:
            d["perply"] = {
                k: {i: v - b[k][i] for i, v in a[k].items() if i in b[k]}
                for k in a
                if k in b
            }
        r["delta"] = d


def load_knobs():
    """moe-knobs/knobs.py (its cm() and frozen law, input hashes checked)."""
    global KNOBS, LAWS
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location("knobs", R / "moe-knobs/knobs.py")
        KNOBS = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(KNOBS)
        LAWS = KNOBS.laws()
    except Exception as e:
        print("knobs unavailable:", repr(e), file=sys.stderr)
        KNOBS = LAWS = None


def record(pr, study, f, curves, now):
    """A run's cached record, refreshed from its files when their signature moved
    (a scored run's files are looked at every RECHECK s only)."""
    name, meta = pr["name"], ST["runs"].get(pr["name"]) or {}
    if meta.get("final") and now - meta["checked"] < RECHECK and name in curves:
        return meta["rec"]
    sig = signature(f)
    if sig != meta.get("sig") or name not in curves or "rec" not in meta:
        cp = OUT / "cache" / f"{name}.json"
        c = jload(cp, {}) or {}
        ingest(c, f)
        meta["rec"], curves[name] = derive(pr, c, study)
        save(cp, c)
    ST["runs"][name] = meta | dict(sig=sig, checked=now)
    return meta["rec"]


def index():
    now = time.time()
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    load_knobs()
    perply = [d for d in R.glob("*/perply") if d.is_dir()]
    plans = studies()
    jobs, ids = job_runs(ST.get("snap"), plans)
    ev, listing = orun_events(), remote_mtimes()
    curves = jload(OUT / "curves.json", {}) or {}
    runs, seen = [], set()
    for key, study in plans.items():
        for pr in study["runs"]:
            if pr["name"] in seen:
                continue
            seen.add(pr["name"])
            f = sources(pr, study, perply)
            rec = record(pr, study, f, curves, now)
            runs.append(assemble(pr, study, dict(rec), jobs, ids, ev, listing, now))
            ST["runs"][pr["name"]]["final"] = runs[-1]["status"] == "scored"
    deltas(runs)
    for r in runs:
        r.pop("_mtime")
    save(OUT / "runs.json", dict(generated=now, runs=runs))
    save(OUT / "curves.json", {k: v for k, v in curves.items() if k in seen})
    save(OUT / "sweep.json", sweep(runs))
    save(OUT / "compute.json", compute_summary())


def assemble(pr, study, r, jobs, ids, ev, listing, now):
    """The record plus plan identity, live status, jobs, hardware and derived rates."""
    name = pr["name"]
    r |= dict(
        name=name, study=study["key"], track=study["track"], round=study["round"],
        wave=study["wave"], _mtime=study["mtime"], v=pr["v"], budget=pr["budget"],
        seed=pr["seed"], tag=pr.get("tag"), layers=pr["layers"], width=pr["width"],
        experts=(pr["moe"] or [None])[0], topk=(pr["moe"] or [None, None])[1],
        family=family(pr), steps=pr["steps"], stop_after=pr.get("stop_after"),
        dims=pr["dims"], update=pr.get("update"),
        tokens_plan=pr["steps"] * BATCH, law_n=12 * pr["layers"] * pr["width"] ** 2,
        recipe=recipe(pr.get("policy")), pool_frac=pr.get("pool_frac"),
    )  # fmt: skip
    r["t_last"] = listing.get(name, r["t_last"])
    acct = [ST["sacct"][j] for j in r["job_ids"] if j in ST["sacct"]]
    if (
        r["orchard"] and acct
    ):  # hand-back mtimes are copy times; the orchard jobs' are real
        r["t_start"] = time.mktime(time.strptime(acct[0]["start"], "%Y-%m-%dT%H:%M:%S"))
    if e := ev.get(name):
        r["orchard_jobs"], r["orchard_done"] = e["jobs"], e["done"]
        r["t_start"] = r["t_start"] or e["t0"]
        r["t_done"] = r["t_done"] or (e["t1"] if e["done"] else None)
        if e["errors"] and not e["done"]:
            n = e["errors"]
            r["alerts"] = r["alerts"] + [
                dict(
                    kind="orun",
                    text=f"{n} traceback{'s' * (n > 1)} in its orchard job logs",
                )
            ]
    js = list(jobs.get(name, []))
    js += [ids[j] for j in r["job_ids"] if j in ids and ids[j] not in js]
    r["model"], r["arch"] = words(pr, r)
    r["tpp"] = r["tokens_plan"] / r["params_active"] if r["params_active"] else None
    r["status"] = status(pr, r, js, now)
    keep = (
        "id",
        "aid",
        "state",
        "part",
        "cluster",
        "reason",
        "gpus",
        "gpu",
        "node",
        "elapsed",
    )
    r["jobs"] = [{k: j.get(k) for k in keep} for j in js]
    run = next((j for j in js if j["state"] == "RUNNING"), None)
    acct = acct[-1] if acct else None
    if r["orchard"] or (run or {}).get("cluster") == "orchard":
        r["gpu"], r["queue"], r["cluster"] = "H100", "orchard", "orchard"
    elif run or acct:
        x = run or acct
        r["gpu"], r["queue"], r["cluster"] = (
            x.get("gpu") or x.get("type"),
            x["part"],
            "babel",
        )
    else:
        r["gpu"] = (r.get("res") or {}).get("gpu") or study["gpu"]
        r["queue"] = ",".join(sorted({j["part"] for j in js})) or None
        r["cluster"] = "babel"
    k = gpu_key(r["gpu"])
    r["gpu"] = GPUS.get(k, r["gpu"])
    r["world"] = r["world"] or study["gpus"]
    r["peak"] = PEAK[k] * r["world"] if k and r["world"] else None
    r["eff"] = r["eff"] | dict(
        mfu=r["eff"]["fps"] / r["peak"] if r["eff"]["fps"] and r["peak"] else None
    )
    r["gpu_h"] = (
        r["world"] * r["seconds"] / 3600 if r["seconds"] and r["world"] else None
    )
    r["restarts"] = max(0, r["starts"] - 1)
    quiet = now - (r["t_last"] or now)
    if r["status"] == "running" and quiet > max(STALL, 8 * (r["eff"]["window"] or 0)):
        r["alerts"] = r["alerts"] + [
            dict(kind="stalled", text=f"no train row for {quiet / 60:.0f} min")
        ]
    if r["track"] == "data" and pr.get("policy"):
        mk = f"{study['key']}|{pr['policy']}"
        if mk not in ST["mix"]:
            ST["mix"][mk] = recipe_plan(study, pr["policy"])
        r["plan_mix"] = ST["mix"][mk]
    return r


# ------------------------------------------------------------------ sweep


def sweep(runs):
    """isoFLOP points (law N = 12 layers width^2, D = tokens, C = N D), per-budget
    parabola minima, the L(N, D) refit per family (isoflop_fit.additive), N*(C), and the
    S16-over-dense CM, at a node-week with bootstrap errors (200 resamples)."""
    pts = [r for r in runs if r["track"] == "sweep"]
    res = lambda r, k: (r.get("res") or {}).get(k)
    out = dict(
        points=[
            dict(
                name=r["name"],
                family=r["family"],
                budget=r["budget"],
                n=r["law_n"],
                d=r["tokens_plan"],
                active=r["params_active"],
                total=r["params_total"],
                status=r["status"],
                shape=f"{r['layers']}x{r['width']}",
                step=r["step"],
                steps=r["steps"],
                train_ce=r["train_ce"],
                macro=res(r, "macro"),
                expert=res(r, "expert"),
                flops=res(r, "flops"),
            )  # fmt: skip
            for r in pts
        ]
    )
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        import dmix_fit
        import isoflop_fit as fit
        import numpy as np
        from scipy.optimize import minimize_scalar
    except Exception as e:
        out["error"] = repr(e)
        return out
    grid = np.logspace(16, 21, 41)

    def nopt(p, c):
        f = lambda x: fit.predict(p, np.exp(x), c / np.exp(x))
        b = (np.log(1e4), np.log(c / 1e4))
        return float(np.exp(minimize_scalar(f, bounds=b, method="bounded").x))

    if LAWS:
        th = np.array(LAWS["macro"]["theta"])
        out["reference"] = dict(
            note="frozen isoflop-v1 golden macro law (old bundle, dense): orientation only",
            c=grid.tolist(),
            loss=[fit.best_loss(th, c) for c in grid],
            n=[nopt(th, c) for c in grid],
        )
    scored = [p for p in out["points"] if p["macro"] is not None]
    ratio = [p["flops"] / (p["n"] * p["d"]) for p in scored if p["flops"]]
    out["flops_per_nd"] = statistics.median(ratio) if ratio else None
    cnw = out["node_week_c"] = NODE_WEEK / (out["flops_per_nd"] or 6.2)
    for metric in ("macro", "expert"):
        fams = {
            f: np.array(
                [
                    (p["n"], p["d"], float(p["budget"]), p[metric])
                    for p in scored
                    if p["family"] == f
                ]
            )
            for f in ("S16", "dense")
        }
        fams = {f: a for f, a in fams.items() if len(a)}
        m = out[metric] = dict(minima=fit.isoflop(fams) if fams else {})
        ok = {f: a for f, a in fams.items() if len(a) >= 6 and len(set(a[:, 2])) >= 2}
        if not ok:
            continue
        key = json.dumps({f: a.tolist() for f, a in ok.items()})
        if (ST["fits"].get(metric) or {}).get("key") != key:
            ST["fits"][metric] = dict(
                key=key, out=refit(ok, grid, cnw, nopt, fit, dmix_fit, np)
            )
        m |= ST["fits"][metric]["out"]
    return out


def refit(ok, grid, cnw, nopt, fit, dmix_fit, np):
    data = {("ours" if f == "S16" else "qwen"): a for f, a in ok.items()}
    theta = fit.additive(data, shared=False)
    ps = fit.unpack(theta, False)
    law = {f: ps["ours" if f == "S16" else "qwen"] for f in ok}
    out = dict(laws={}, residuals=[])
    for f, p in law.items():
        a = ok[f]
        r = fit.predict(p, a[:, 0], a[:, 1]) - a[:, 3]
        out["laws"][f] = dict(
            E=float(np.exp(p[0])), A=float(np.exp(p[1])), alpha=float(p[2]),
            B=float(np.exp(p[3])), beta=float(p[4]), rmse=float(np.sqrt(np.mean(r**2))),
            points=len(a), c=grid.tolist(), loss=[fit.best_loss(p, c) for c in grid],
            nopt=[nopt(p, c) for c in grid],
        )  # fmt: skip
        out["residuals"] += [
            dict(family=f, n=x[0], budget=x[2], r=v)
            for x, v in zip(a.tolist(), r.tolist())
        ]
    if len(law) < 2:
        return out
    cm = lambda pm, pd, c: dmix_fit.multiplier(
        lambda x: fit.best_loss(pd, x), fit.best_loss(pm, c), c
    )
    out["cm"] = dict(
        c=grid.tolist(),
        value=[cm(law["S16"], law["dense"], c) for c in grid],
        node_week=cm(law["S16"], law["dense"], cnw),
    )
    rng, boot = np.random.default_rng(0), []
    for _ in range(200):
        b = {k: a[rng.integers(0, len(a), len(a))] for k, a in data.items()}
        try:
            q = fit.unpack(fit.additive(b, shared=False, starts=[theta]), False)
            boot.append(cm(q["ours"], q["qwen"], cnw))
        except Exception:
            pass
    boot = sorted(x for x in boot if x is not None and math.isfinite(x))
    if boot:
        q = lambda f: boot[min(len(boot) - 1, int(f * len(boot)))]
        out["cm"]["boot"] = dict(
            n=len(boot), p16=q(0.16), p84=q(0.84), p2=q(0.025), p97=q(0.975)
        )
    return out


# ------------------------------------------------------------------ compute

SQ = ("id", "aid", "name", "task", "state", "part", "qos", "tres", "nodes", "node",
      "start", "elapsed", "reason", "comment", "cmd")  # fmt: skip
SQFMT = (
    "%A|%i|%j|%K|%T|%P|%q|%b|%D|%N|%S|%M|%r|%k|%o"  # orchard_mirror.sh uses the same
)


def tasks_of(s):
    """Array task ids of squeue's %K ('3', '0-3%8', '0,2'), None for a non-array job."""
    s = s.split("%")[0]
    if not re.fullmatch(r"[\d,\-]+", s):
        return None
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def jobs_of(text, cluster):
    out = []
    for line in text.splitlines():
        x = dict(zip(SQ, line.split("|", len(SQ) - 1)))
        if len(x) < len(SQ):
            continue
        t, n = gres(x["tres"])
        x |= dict(
            gpu=t,
            gpus=n * int(x["nodes"] or 1),
            tasks=tasks_of(x["task"]),
            cluster=cluster,
        )
        out.append(x)
    return out


def compute():
    """One snapshot of our queues, the dei labmate flag (the governor's rule: another
    user's dei-group job pending on Resources / Priority), the governor's held list and
    the idle GPUs per partition and type -> compute.jsonl; then the sacct GPU-h cache."""
    me = os.environ.get("USER", "yimingz3")
    snap = dict(
        at=time.time(),
        babel=dict(jobs=jobs_of(sh("squeue", "-h", "-u", me, "-o", SQFMT), "babel")),
    )
    fmt = "UserName:0|,JobArrayID:0|,Reason:0|,tres-alloc:0"
    pd = [
        line.split("|")
        for line in sh(
            "squeue", "-h", "-r", "-p", "dei-group", "-t", "PD", "-O", fmt
        ).splitlines()
    ]
    snap["dei_waiting"] = [
        dict(user=x[0], id=x[1], reason=x[2])
        for x in pd
        if len(x) > 2 and x[0] != me and x[2] in ("Resources", "Priority")
    ]
    alive = (
        subprocess.run(
            ["pgrep", "-f", "dei_governor.sh"], capture_output=True
        ).returncode
        == 0
    )
    held = (GOV / "held").read_text().split() if (GOV / "held").exists() else []
    snap["governor"] = dict(alive=alive, held=held, log=last_lines(GOV / "log", 4))
    idle = {}
    fmt = "NodeHost:40,Partition:20,Gres:60,GresUsed:80,StateCompact:12"
    for line in sh(
        "sinfo", "-h", "-N", "-p", "general,preempt,dei-group", "-O", fmt
    ).splitlines():
        f = line.split()
        (t, n), (_, u) = (
            gres(f[2] if len(f) > 4 else ""),
            gres(f[3] if len(f) > 4 else ""),
        )
        if t and n > u and re.match(r"(idle|mix)", f[4]):
            k = f"{f[1].rstrip('*')}|{t}"
            idle[k] = idle.get(k, 0) + n - u
    snap["idle"] = idle
    if (s := stat(MIRROR / "squeue.txt")) and time.time() - s.st_mtime < 1800:
        snap["orchard"] = dict(
            at=s.st_mtime, jobs=jobs_of((MIRROR / "squeue.txt").read_text(), "orchard")
        )
    with open(OUT / "compute.jsonl", "a") as f:
        f.write(json.dumps(snap, separators=(",", ":")) + "\n")
    acct = jload(OUT / "sacct.json", {}) or {}
    jobs = acct.get("jobs", {})
    since = time.strftime(
        "%Y-%m-%dT%H:%M:%S",
        time.localtime(max(PHASE, acct.get("last", PHASE) - 3 * 3600)),
    )
    cols = "JobIDRaw,JobID,JobName,Partition,AllocTRES,ElapsedRaw,State,Start,End"
    text = sh("sacct", "-u", me, "-S", since, "-X", "-D", "-n", "-P", "-o", cols)
    orchard = (
        (MIRROR / "sacct.txt").read_text() if (MIRROR / "sacct.txt").exists() else ""
    )
    keys = ("raw", "id", "name", "part", "tres", "sec", "state", "start", "end")
    for cluster, t in (("babel", text), ("orchard", orchard)):
        for line in t.splitlines():
            x = dict(zip(keys, line.split("|")))
            typ, n = gres(x.get("tres"))
            if n and re.match(r"\d{4}-", x.get("start") or ""):
                typ = typ or ("H100" if cluster == "orchard" else None)
                jobs[f"{cluster}:{x['raw']}@{x['start']}"] = x | dict(
                    cluster=cluster, gpus=n, type=typ, sec=int(x["sec"] or 0)
                )
    save(OUT / "sacct.json", dict(last=time.time(), jobs=jobs))


def track_of(name):
    return next((t for t, pat in TRACKS if re.match(pat, name or "")), "other")


def pool_of(j):
    return (
        "orchard"
        if j.get("cluster") == "orchard"
        else (j.get("part") or "").split(",")[0]
    )


def compute_summary():
    """Pools vs caps now, 48 h of our GPUs per pool, GPU-h per track and GPU type since
    PHASE, preempted / requeued incarnations, and approved work waiting while GPUs of
    its type sit idle within our cap."""
    snap = ST.get("snap") or {}
    pools = {
        k: dict(cap=v, used=0, pending=0, held=0, jobs=[]) for k, v in CAPS.items()
    }
    for j in snap.get("babel", {}).get("jobs", []) + snap.get("orchard", {}).get(
        "jobs", []
    ):
        p = pools.get(pool_of(j))
        if not p or not j["gpus"]:
            continue
        held = "Held" in (j.get("reason") or "")
        kind = "used" if j["state"] == "RUNNING" else "held" if held else "pending"
        p[kind] += j["gpus"] if j["state"] in ("RUNNING", "PENDING") else 0
        keep = (
            "aid",
            "name",
            "state",
            "gpus",
            "gpu",
            "reason",
            "elapsed",
            "node",
            "part",
        )
        p["jobs"].append({k: j.get(k) for k in keep} | dict(track=track_of(j["name"])))
    labmate = bool(snap.get("dei_waiting"))
    if labmate:
        pools["dei-group"]["cap"] = 8
    flags, idle = [], snap.get("idle", {})
    for k, p in pools.items():
        for j in p["jobs"]:
            why = j["reason"] or ""
            if (
                j["state"] != "PENDING"
                or "Held" in why
                or "Dependency" in why
                or k == "orchard"
            ):
                continue
            ok = lambda key: (
                key.startswith(k + "|")
                and (not j["gpu"] or key.split("|")[1].lower() == j["gpu"].lower())
            )
            free = sum(v for key, v in idle.items() if ok(key))
            if free >= j["gpus"] and p["used"] + j["gpus"] <= p["cap"]:
                flags.append(
                    dict(
                        pool=k,
                        text=f"{j['name']} ({j['aid']}) waits on {why} while {free} {j['gpu'] or ''} GPUs sit idle on {k}",
                    )
                )
        if (
            k == "general"
            and p["used"] < p["cap"]
            and not any(j["state"] == "PENDING" for j in p["jobs"])
        ):
            flags.append(
                dict(
                    pool=k,
                    text=f"general: {p['cap'] - p['used']} GPUs of our cap unused, nothing queued there",
                )
            )
    gh, pre = {}, {}
    for x in ((jload(OUT / "sacct.json", {}) or {}).get("jobs", {})).values():
        try:
            if time.mktime(time.strptime(x["start"], "%Y-%m-%dT%H:%M:%S")) < PHASE:
                continue
        except (ValueError, TypeError, KeyError):
            continue
        t, g = track_of(x["name"]), GPUS.get(gpu_key(x["type"]), x["type"] or "?")
        gh.setdefault(t, {})[g] = gh.get(t, {}).get(g, 0) + x["gpus"] * x["sec"] / 3600
        pre[t] = pre.get(t, 0) + (x["state"].startswith(("PREEMPTED", "REQUEUE")))
    return dict(
        at=snap.get("at"),
        orchard_at=(snap.get("orchard") or {}).get("at"),
        pools=pools,
        labmate=labmate,
        dei_waiting=snap.get("dei_waiting", []),
        governor=snap.get("governor"),
        idle=idle,
        flags=flags,
        series=usage((jload(OUT / "sacct.json", {}) or {}).get("jobs", {}).values()),
        gpu_h=gh,
        preempted=pre,
        budget=BUDGET,
        since=PHASE,
    )


def usage(jobs):
    """Our GPUs in use per pool every 10 min over the last 48 h from sacct's intervals
    (a running job ends now), with the labmate flag of the latest earlier snapshot."""
    now, iv, fmt = time.time(), [], "%Y-%m-%dT%H:%M:%S"
    for x in jobs:
        try:
            t0 = time.mktime(time.strptime(x["start"], fmt))
            end = x.get("end") or ""
            t1 = (
                time.mktime(time.strptime(end, fmt))
                if re.match(r"\d{4}-", end)
                else now + 60
            )
        except (ValueError, TypeError, KeyError):
            continue
        if (pool := pool_of(x)) in CAPS and t1 > now - 48 * 3600:
            iv.append((t0, t1, pool, x["gpus"]))
    snaps, out = ST.get("series", []), []
    for i in range(289):
        t, u = now - 48 * 3600 + 600 * i, dict.fromkeys(CAPS, 0)
        for a, b, pool, g in iv:
            u[pool] += g if a <= t < b else 0
        lab = next((x[-1] for x in reversed(snaps) if x[0] <= t), 0)
        out.append([t, *u.values(), lab])
    return out


def snapshots(now):
    """Fold new compute.jsonl lines into the 48 h per-pool series; keep the latest."""
    reset, lines = tail(OUT / "compute.jsonl", ST["off"])
    if reset:
        ST["series"] = []
    for x in parse(lines):
        ST["snap"] = x
        u = dict.fromkeys(CAPS, 0)
        for j in x.get("babel", {}).get("jobs", []) + x.get("orchard", {}).get(
            "jobs", []
        ):
            if j["state"] == "RUNNING" and pool_of(j) in u:
                u[pool_of(j)] += j["gpus"]
        ST.setdefault("series", []).append(
            [x["at"], *u.values(), int(bool(x.get("dei_waiting")))]
        )
    ST["series"] = [s for s in ST.get("series", []) if now - s[0] < 48 * 3600]


def page(out):
    data = {
        k: jload(OUT / f"{k}.json", {}) for k in ("runs", "curves", "sweep", "compute")
    }
    blob = json.dumps(data | dict(caps=CAPS), separators=(",", ":")).replace(
        "</", "<\\/"
    )
    html = (
        (ROOT / "scripts/runindex.html")
        .read_text()
        .replace("/*RUNINDEX_DATA*/{}", blob)
    )
    tmp = Path(out).with_name(Path(out).name + ".tmp")
    tmp.write_text(html)
    tmp.replace(out)


def main():
    sys.dont_write_bytecode = (
        True  # importing frozen study modules must not write there
    )
    os.nice(max(0, 19 - os.nice(0)))
    cmd = sys.argv[1] if len(sys.argv) > 1 else "index"
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "page":
        return page(sys.argv[2] if len(sys.argv) > 2 else OUT / "allie-runs.html")
    lock = open(OUT / "index.lock", "w")
    fcntl.flock(
        lock, fcntl.LOCK_EX
    )  # one writer of state / sacct / compute.jsonl at a time
    if cmd == "compute":
        return compute()
    ST.update(jload(OUT / "state.json", {}) or {})
    for k in ("off", "plans", "runs", "races", "cm", "fits", "mix"):
        ST.setdefault(k, {})
    ST["off"] = {k: v for k, v in ST["off"].items() if k.endswith("compute.jsonl")}
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if ST.get("code") != code:  # new code: re-derive every record from its cache
        ST.update(code=code, runs={}, plans={}, fits={}, mix={})
    acct = (jload(OUT / "sacct.json", {}) or {}).get("jobs", {})
    ST["sacct"] = {x["raw"]: x for x in acct.values()}
    snapshots(time.time())
    index()
    del ST["sacct"]
    save(OUT / "state.json", ST)


if __name__ == "__main__":
    main()
