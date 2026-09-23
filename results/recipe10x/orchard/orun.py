"""GCS-backed, preemption-safe orchard leg of one frozen babel run (omake.py). Usage: orun.py <study>.
Checks the study mirror against plan.json's hashes, stages the run's months from gs://cmu-gpucloud-yimingz3/data/v1 into
the job's /tmp, resumes from runs/<name>/LATEST.json, checkpoints every ~OCKPT_MIN (15) minutes of training, publishes
each checkpoint (files, then the pointer, forward only, generation precondition), requeues until done, then hands the
final run dir to babel results/pretrain/<name> for scoring. Run = plan[SLURM_ARRAY_TASK_ID] unless ORUN_NAME is set.
Tests: OLEGS="200,400" trains --stop-after legs with a requeue between them; OKILL_STEP=K requeues the job (as a
preemption does: SIGTERM, then SIGKILL after 30 s) OKILL_DELAY s after the first published checkpoint >= K, once."""

import base64, hashlib, json, math, os, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch

BUCKET = "gs://cmu-gpucloud-yimingz3"
DATA = ("v1", "v1-f0.19")  # GCS data prefixes in order of preference: whole months, then the f <= 0.19 subset
study = Path.home() / "allie/studies" / sys.argv[1]
spec = json.loads((study / "orchard/args.json").read_text())
plan = json.loads((study / "plan.json").read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
# every file the plan hashes must be in the mirror and match ('baseline' is a digest, not a file)
bad = [
    k
    for k, v in plan["hashes"].items()
    if k != "baseline" and not ((study / k).is_file() and sha(study / k) == v)
]
bad += [k for k, v in spec["source"].items() if sha(study / k) != v]
assert not bad, f"study mirror differs from the frozen plan: {bad}"
assert spec.get("plan_sha256") == sha(study / "plan.json"), "orchard/args.json was not made from this plan.json"
name = (
    os.environ.get("ORUN_NAME")
    or plan["runs"][int(os.environ["SLURM_ARRAY_TASK_ID"])]["name"]
)
run = spec["runs"][name]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import oargs

assert Path(spec["study"]).name == study.name, f"args.json is for {spec['study']}"
oargs.check(run["args"], next(r for r in plan["runs"] if r["name"] == name), spec["study"], spec["store"])
# an array task's SLURM_JOB_ID can equal the array's id (the last task gets it), which squeue/scontrol take as every task
aid = os.environ.get("SLURM_ARRAY_JOB_ID")
job = f"{aid}_{os.environ['SLURM_ARRAY_TASK_ID']}" if aid else os.environ["SLURM_JOB_ID"]
host = os.uname().nodename
gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
assert gpus == run["gpus"], f"{name} is frozen for {run['gpus']} GPUs, job has {gpus}"
LOCAL = Path(os.environ["LOCAL"])
store, root = LOCAL / "g", LOCAL / "root"
out, RUN = root / "results/pretrain" / name, f"{BUCKET}/runs/{name}"
legs = [int(x) for x in os.environ.get("OLEGS", "").split(",") if x]
log = lambda *a: print(time.strftime("%F %T"), name, *a, flush=True)


def gs(*a, check=True):
    return subprocess.run(
        ["gcloud", "storage", *map(str, a)], check=check, text=True, capture_output=True
    )


def stage():
    """Months named by the run's args (plus the val rows), one tar each, extracted once per job."""
    a = run["args"]
    months = a[a.index("--mix-months") + 1].split(",")
    rels = sorted({m[len(spec["store"]) + 1 :] for m in months} | {"lichess_tokens_v2"})

    def one(rel):
        if (store / rel / ".staged").exists():
            return 0
        for pre in DATA:  # the whole month, else a subset (oupload.sh SUBSET): check_staged() tests it covers the run
            r = gs("cat", f"{BUCKET}/data/{pre}/{rel}.tar.json", check=False)
            if r.returncode == 0:
                break
        else:
            raise FileNotFoundError(f"{rel} is in none of {DATA}")
        meta = json.loads(r.stdout)
        tar = LOCAL / "dl" / f"{rel}.tar"
        tar.parent.mkdir(parents=True, exist_ok=True)
        gs("cp", f"{BUCKET}/data/{pre}/{rel}.tar", tar)  # crc32c-validated
        assert tar.stat().st_size == meta["bytes"], rel
        (store / rel).mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xf", tar, "-C", store], check=True)
        tar.unlink()
        (store / rel / ".staged").write_text(json.dumps(dict(prefix=pre, subset_f=meta.get("subset_f"))) + "\n")
        return meta["bytes"]

    t = time.time()
    with ThreadPoolExecutor(8) as ex:
        n = sum(ex.map(one, rels))
    log(f"staged {len(rels)} months, {n / 1e9:.0f} GB new, {time.time() - t:.0f}s")


def latest():
    r = gs(
        "objects",
        "describe",
        f"{RUN}/LATEST.json",
        "--format=value(generation)",
        check=False,
    )
    if r.returncode:
        return 0, None
    gen = int(r.stdout.strip())
    return gen, json.loads(gs("cat", f"{RUN}/LATEST.json#{gen}").stdout)


def owner():
    r = gs("cat", f"{RUN}/OWNER.json", check=False)
    return json.loads(r.stdout)["job"] if r.returncode == 0 else None


def step_seconds():
    """Median seconds per step over this run's logged windows (compile excluded), else None."""
    p = out / "train.jsonl"
    rows = (
        [
            json.loads(l)
            for l in p.read_text().splitlines()
            if '"tokens_per_second"' in l
        ]
        if p.exists()
        else []
    )
    dt = sorted(
        (b["seconds"] - a["seconds"]) / (b["step"] - a["step"])
        for a, b in zip(rows[1:], rows[2:])
        if b["step"] > a["step"]
    )
    return dt[len(dt) // 2] if dt else None


RUNFILES = (
    "train.jsonl",
    "validation.jsonl",
    "checkpoints.jsonl",
    "config.json",
    "resume-config.json",
    "done.json",
    "orchard.json",
)


def publish(pointer, done=False):
    """Upload checkpoint dir + a run-file snapshot, then move LATEST forward (never back)."""
    d = pointer["directory"].split("/")[-1]
    pub = LOCAL / "pub" / d
    shutil.rmtree(pub, ignore_errors=True)
    pub.mkdir(parents=True)
    try:  # hard links survive the trainer pruning the directory mid-upload
        for f in (out / pointer["directory"]).iterdir():
            os.link(f, pub / f.name)
    except FileNotFoundError:
        shutil.rmtree(pub)
        return False
    subprocess.run(
        [
            "tar",
            "-cf",
            pub / "run.tar",
            "-C",
            out,
            *[f for f in RUNFILES if (out / f).exists()],
        ],
        check=True,
    )
    files = {f.name: f.stat().st_size for f in pub.iterdir()}
    gs("cp", *sorted(pub.iterdir()), f"{RUN}/ckpt/{d}/")
    ls = gs("ls", "-l", f"{RUN}/ckpt/{d}/").stdout.splitlines()
    listed = {
        Path(l.split()[-1]).name: int(l.split()[0])
        for l in ls
        if l.strip().endswith(tuple(files))
    }
    assert listed == files, (listed, files)
    shutil.rmtree(pub)
    body = dict(
        step=pointer["step"],
        directory=d,
        world_size=pointer["world_size"],
        done=done,
        files=files,
        step_seconds=step_seconds(),
        job=job,
        host=host,
        at=time.time(),
    )
    while True:
        gen, cur = latest()
        if cur and (cur["step"], cur["done"]) >= (body["step"], done):
            log(
                f"LATEST already at step {cur['step']} (done={cur['done']}); not moving it back"
            )
            return False
        assert owner() == job, f"run now owned by job {owner()}; stopping publication"
        p = LOCAL / "LATEST.json"
        p.write_text(json.dumps(body) + "\n")
        if (
            gs(
                "cp",
                f"--if-generation-match={gen}",
                p,
                f"{RUN}/LATEST.json",
                check=False,
            ).returncode
            == 0
        ):
            break
    dirs = gs("ls", f"{RUN}/ckpt/").stdout.split()
    keep = sorted(Path(l.rstrip("/")).name for l in dirs)[-2:]
    for l in dirs:
        if Path(l.rstrip("/")).name not in keep:
            gs("rm", "-r", l, check=False)
    log(
        f"published step {body['step']} ({sum(files.values()) / 1e9:.1f} GB){' done' if done else ''}"
    )
    return True


def restore(cur):
    d = out / "checkpoints" / cur["directory"]
    d.mkdir(parents=True, exist_ok=True)
    gs("cp", *[f"{RUN}/ckpt/{cur['directory']}/{f}" for f in cur["files"]], d)
    subprocess.run(["tar", "-xf", d / "run.tar", "-C", out], check=True)
    (d / "run.tar").unlink()
    for f in (
        "train.jsonl",
        "validation.jsonl",
    ):  # rows logged past the checkpoint are retrained
        if (out / f).exists():
            rows = (out / f).read_text().splitlines(keepends=True)
            (out / f).write_text(
                "".join(l for l in rows if json.loads(l).get("step", 0) <= cur["step"])
            )
    pointer = dict(
        format="allie-modded-medium-1",
        directory=f"checkpoints/{cur['directory']}",
        step=cur["step"],
        world_size=cur["world_size"],
    )
    torch.save(pointer, out / "last.pt")
    log(f"restored step {cur['step']} from {RUN}")


def handoff():
    """Final run dir (last checkpoint + run files) -> babel results/pretrain/<name> for scoring."""
    if (
        os.environ.get("OBABEL_DEST") == "none"
        or gs("ls", f"{RUN}/HANDOFF.json", check=False).returncode == 0
    ):
        return
    dest = (
        f"{os.environ.get('OBABEL_DEST', spec['store'] + '/results/pretrain')}/{name}"
    )
    ox = lambda c: ["ssh", "-o", "BatchMode=yes", os.environ["BABEL"], "~/src/allie/.orchard/oxcmd.sh " + base64.b64encode(c.encode()).decode()]
    free = int(
        subprocess.check_output(
            ox(f"df --output=avail -B1 {spec['store']} | tail -1"), text=True
        )
    )
    pointer = torch.load(out / "last.pt", weights_only=False)
    # scoring reads model.pt only; the optimizer/RNG/data states (rank*.pt) stay in GCS
    items = ["last.pt", f"{pointer['directory']}/model.pt", *[f for f in RUNFILES if (out / f).exists()]]
    size = sum((out / i).stat().st_size for i in items)
    floor = float(os.environ.get("OBABEL_FLOOR", 400e9))  # main, 2026-09-21: 400 GB (was 1.2 TB)
    assert free - size >= floor, f"babel /data/group_data would drop below {floor / 1e9:.0f} GB free"
    ours = (
        subprocess.run(
            ox(f"test ! -e {dest} || test -e {dest}/orchard.json")
        ).returncode
        == 0
    )
    assert ours, f"{dest} exists on babel and is not an orchard hand-off"
    tar = subprocess.Popen(
        ["tar", "-cf", "-", "-C", out, *items], stdout=subprocess.PIPE
    )
    subprocess.run(
        ox(f"mkdir -p {dest} && tar -xf - -C {dest}"), stdin=tar.stdout, check=True
    )
    assert tar.wait() == 0
    p = LOCAL / "HANDOFF.json"
    p.write_text(
        json.dumps(
            dict(
                dest=dest,
                step=pointer["step"],
                bytes=size,
                full_checkpoint=f"{RUN}/ckpt/{pointer['directory'].split('/')[-1]}/",
                job=job,
                at=time.time(),
            )
        )
        + "\n"
    )
    gs("cp", p, f"{RUN}/HANDOFF.json")
    log(f"handed off to babel {dest} ({size / 1e9:.1f} GB)")


gen, cur = latest()
if cur and cur["done"]:
    log(f"done at step {cur['step']}")
    if os.environ.get("OBABEL_DEST") != "none" and gs("ls", f"{RUN}/HANDOFF.json", check=False).returncode:
        if not (out / "last.pt").exists():
            restore(cur)
        handoff()
    sys.exit(0)
p = LOCAL / "OWNER.json"
p.write_text(json.dumps(dict(job=job, host=host, at=time.time())) + "\n")
gs("cp", p, f"{RUN}/OWNER.json")
stage()


def verify_pin():
    """datapin.verify on the staged copies: pin unedited, the run's months are the pin's, pinned files unchanged."""
    if "data-pin.json" not in plan["hashes"]:  # a pin the plan declares is never optional (hash check above)
        return
    sys.path.insert(0, str(study))
    import datapin

    p = json.loads((study / "data-pin.json").read_text())
    assert p.pop("sha256") == datapin.digest(p), "data-pin.json edited after pinning"
    a = run["args"]
    assert sorted(a[a.index("--mix-months") + 1].split(",")) == sorted(p["months"]), "run months differ from the pin"
    local = lambda m: store / m[len(spec["store"]) + 1 :]
    bad = [m for m, h in p["files"].items() if {f: datapin.sha(local(m) / f) for f in h} != h]
    assert not bad, f"staged pinned months differ from the pin: {bad}"
    log(f"data pin verified: {len(p['months'])} months")


verify_pin()


def check_staged():
    """Every shard the run's sampler can open is on local disk: chessmix._index at the run's pool_frac and history
    counts (per bucket the first ceil(min(1, scale) x games) games, 50000 per shard)."""
    a = run["args"]
    months = [store / m[len(spec["store"]) + 1 :] for m in a[a.index("--mix-months") + 1].split(",")]
    f = float(a[a.index("--mix-pool-frac") + 1])
    hist = json.loads((study / "history-counts.json").read_text())["counts"] if "--mix-history" in a else None
    buckets = [(m, b) for m in months for b in json.loads((m / "buckets.json").read_text())]
    games = {}
    for _, b in buckets:
        games[b["code"]] = games.get(b["code"], 0) + b["games"]
    scale = {c: f * hist.get(str(c), 0) / n for c, n in games.items()} if hist else dict.fromkeys(games, f)
    missing, n = [], 0
    for m, b in buckets:
        left = math.ceil(min(1, scale[b["code"]]) * b["games"])
        for k, s in enumerate(b["shards"]):
            rows = min(50000, b["games"] - k * 50000, left)
            if rows <= 0:
                break
            n += 1
            missing += [str(m / s)] * (not (m / s).is_file())
            left -= rows
    assert not missing, f"{len(missing)} of {n} shards the sampler can open are not staged, e.g. {missing[:3]}"
    log(f"staged shards cover the run: {n} shards at pool_frac {f}")


check_staged()
args = [
    a.replace(spec["study"], str(study)).replace(spec["store"], str(store))
    for a in run["args"]
]
(root / "results").mkdir(parents=True, exist_ok=True)
shutil.copy(
    Path.home() / "allie/src/original-data.json", root / "results/original-data.json"
)
if cur and not (out / "last.pt").exists():
    assert cur["world_size"] == gpus, (
        f"checkpoint world size {cur['world_size']} != {gpus} GPUs"
    )
    restore(cur)
step0 = cur["step"] if cur else 0
out.mkdir(parents=True, exist_ok=True)
(out / "orchard.json").write_text(
    json.dumps(dict(study=study.name, run=name, bucket=RUN, gpus=gpus)) + "\n"
)
leg = next((l for l in legs if l > step0), None)
assert leg or not legs, f"all test legs {legs} already done at step {step0}"
# checkpoint cadence is operational (not in the resume asserts or the numerics): ~OCKPT_MIN minutes of training
sps = (
    (cur or {}).get("step_seconds") or run.get("step_seconds") or float(os.environ.get("OSTEP_SECONDS", 0)) or None
)
every = (
    max(16, round(60 * float(os.environ.get("OCKPT_MIN", 15)) / sps))
    if sps
    else int(args[args.index("--checkpoint-every") + 1])
)
args[args.index("--checkpoint-every") + 1] = str(every)

stop, seen, kill = threading.Event(), [0], int(os.environ.get("OKILL_STEP", 0))


def watch():
    while not stop.wait(20):
        if not (out / "last.pt").exists():
            continue
        pointer = torch.load(out / "last.pt", weights_only=False)
        if pointer["step"] > max(seen[0], step0):
            try:
                publish(pointer)
                seen[0] = pointer["step"]
            except (
                Exception
            ) as e:  # keep training; the next checkpoint or the final publish retries
                log(f"publish of step {pointer['step']} failed: {e!r}")
        if (
            kill
            and seen[0] >= kill
            and gs("ls", f"{RUN}/KILLED.json", check=False).returncode
        ):
            p = LOCAL / "KILLED.json"
            p.write_text(
                json.dumps(
                    dict(
                        after_step=seen[0],
                        delay=int(os.environ.get("OKILL_DELAY", 60)),
                        job=job,
                        at=time.time(),
                    )
                )
                + "\n"
            )
            gs("cp", p, f"{RUN}/KILLED.json")
            time.sleep(int(os.environ.get("OKILL_DELAY", 60)))
            log(
                f"test preemption: requeueing job {job} mid-run (last published step {seen[0]})"
            )
            subprocess.run(["scontrol", "requeue", job])


watcher = threading.Thread(target=watch, daemon=True)
watcher.start()
left = subprocess.check_output(
    ["squeue", "-h", "-o", "%L", "-j", job], text=True
).split()[0]
d, _, hms = left.rpartition("-")
secs = int(d or 0) * 86400 + sum(
    int(x) * 60**i for i, x in enumerate(reversed(hms.split(":")))
)
cache = LOCAL / "cache"
env = os.environ | dict(
    ALLIE_PROJECT_ROOT=str(root),
    PYTHONUNBUFFERED="1",
    OMP_NUM_THREADS="4",
    TORCHINDUCTOR_COMPILE_THREADS="4",
    CUBLAS_WORKSPACE_CONFIG=":4096:8",
    CUDA_MODULE_LOADING="LAZY",
    NCCL_CUMEM_HOST_ENABLE="0",
    ALLIE_BOARD_CACHE=str(cache / "board"),
    TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
    TRITON_CACHE_DIR=str(cache / "triton"),
)
trainer = [
    "-m",
    "torch.distributed.run",
    "--standalone",
    f"--nproc_per_node={gpus}",
    study / "source-ours/modded_train.py",
]
cmd = [
    sys.executable,
    *map(str, [os.environ["OTRAINER"]] if os.environ.get("OTRAINER") else trainer),
    *args,
]  # OTRAINER: CPU dry runs
cmd += [
    "--max-seconds",
    str(max(300, secs - min(1800, secs // 4))),
]  # leave time for the final publish
cmd += ["--stop-after", str(leg)] if leg else []
cmd += ["--resume", str(out / "last.pt")] if (out / "last.pt").exists() else []
(study / "logs").mkdir(exist_ok=True)
log(
    f"training from step {step0}{f' to {leg}' if leg else ''} on {gpus} GPUs, checkpoint every {every} steps, {secs}s left"
)
with open(study / f"logs/{name}.train.log", "a") as f:
    rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
stop.set()
watcher.join()
assert rc == 0, f"trainer exited {rc}; not requeueing a crash"
done = json.loads((out / "done.json").read_text())
finished = done["stop_reason"] == "steps" or (
    leg and leg == legs[-1] and done["stop_reason"] == "stop_after"
)
publish(torch.load(out / "last.pt", weights_only=False), done=finished)
if finished:
    handoff()
else:
    log(f"stopped at step {done['step']} ({done['stop_reason']}); requeueing")
    subprocess.run(["scontrol", "requeue", job], check=True)
