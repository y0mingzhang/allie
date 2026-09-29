"""Recoverable training of the pinned medium recipe on chessmix-sampled chess rows."""

import argparse
import hashlib
import json
import os
import random
import signal
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import triton

import modded_arch
import modded_medium
from lm_checkpoint import restore_rng, rng_state
from lm_data import Packed
from modded_arch import extra_flops
from modded_checkpoints import AsyncSaver, RowLog, publish
from modded_checkpoints import prune as prune_checkpoints
from modded_medium import (
    RUNTIME_SOURCE_KEY,
    Config,
    TrainingManager,
    aux_losses,
    config_dict,
    create_model,
    make_context,
    move_losses,
    ratings,
)
from modded_medium_core import sync_params
from modded_moe import STATS
from modded_wsd import Schedule

ROOT = Path(os.environ.get("ALLIE_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
SOURCE = Path(__file__).resolve().parent
# arguments a resume, fork or continuation shares with its checkpoint: they change what is computed
SAME = (
    "width", "layers", "head_dim", "initial_batch_rows", "micro_batch", "lr_scale", "seed",
    "deterministic", "wsd_schedule", "mix", "mix_stores", "mix_pool_frac", "mix_history",
    "mix_months", "clock_feats", "input_lr_mul", "aux_time", "aux_wdl", "arch", "attn_kernel",
)  # fmt: skip


@torch.inference_mode()
def evaluate(model, manager, rows, batch):
    model.eval()
    sums = torch.zeros(6, dtype=torch.float64, device="cuda")
    for lo in range(0, len(rows), batch):
        data = rows[lo : lo + batch]
        ids = torch.as_tensor(data, device="cuda")
        x, y = ids[:, :-1], ids[:, 1:]
        context = make_context(x, manager.ws_short * 128, manager.ws_long * 128)
        logits = model(x.flatten(), y.flatten(), context, manager.get_forward_args())
        values = logits.flatten(0, 1)[:, 378:2346].float()
        targets = y.flatten()
        losses = (
            values.logsumexp(-1)
            - values.gather(1, (targets - 378).clamp(0, 1967)[:, None])[:, 0]
        )
        valid = (targets >= 378) & (targets < 2346)
        elo = torch.as_tensor(ratings(data), device="cuda").flatten()
        for i, mask in enumerate((valid, valid & (elo >= 2400), valid & (elo >= 2600))):
            sums[2 * i] += losses[mask].double().sum()
            sums[2 * i + 1] += mask.sum()
    model.train()
    vals = sums.cpu().tolist()
    return {
        key: vals[2 * i] / max(1, vals[2 * i + 1])
        for i, key in enumerate(("move_ce", "expert2400_ce", "expert2600_ce"))
    } | {
        key: int(vals[2 * i + 1])
        for i, key in enumerate(("move_count", "expert2400_count", "expert2600_count"))
    }


def to_gpu(x):
    """Pinned asynchronous host-to-device copy, so no micro-batch input syncs the host."""
    return torch.from_numpy(x).pin_memory().to("cuda", non_blocking=True)


def moe_stats(manager):
    """Per MoE layer, over the last step (modded_moe.STATS)."""
    if not manager.moe:
        return {}
    stats = torch.stack([m.stats for m in manager.moe]).tolist()
    return {f"moe_{k}": [round(x[i], 4) for x in stats] for i, k in enumerate(STATS)}


def useful_flops(rows, cfg, short_window, long_window):
    x = rows[:, :-1]
    pos = np.arange(x.shape[1])[None, :]
    starts = np.maximum.accumulate(np.where(x == 2348, pos, 0), axis=1)
    lengths = pos - starts + 1
    short_pairs = int(np.minimum(lengths, short_window + 1).sum())
    long_pairs = int(np.minimum(lengths, long_window + 1).sum())
    d, heads, layers = cfg.width, cfg.width // cfg.head_dim, cfg.layers
    # Forward projection + head + attention gates (one per layer) and value-embedding gates,
    # 3 skip gates and the smear gate, over the actual long/short window layer pattern
    # (identical to the 16-layer count used before). Embedding lookups are not matmuls.
    long = len({round(i * (layers - 1) / 15) for i in (0, 4, 11, 15)})
    gates = layers + 2 * min(5, layers // 2)
    per_token = 24 * layers * d * d + 2 * d * 2432 + 2 * (gates * heads * 16 + 64)
    per_token += extra_flops(cfg.arch, d, layers)  # board, SwiGLU and MoE
    pairs = (layers - long) * short_pairs + long * long_pairs
    return 3 * (x.size * per_token + 4 * d * pairs)


def continuation(shared, local, args, config, output, pointer):
    """Strict stable-prefix horizon continuation (--wsd-continue-from): the rank state with only
    the manager's configured horizon migrated, and its provenance."""
    assert shared["args"]["wsd_decay_start"] == -1, "Continue only a stable prefix"
    old = shared["config"]
    assert old == local["manager"]["config"], "Rank/model configuration mismatch"
    assert set(old) == set(config)
    for key in old:
        if key == "scheduled_steps":
            assert config[key] >= old[key], "Cannot shorten the configured horizon"
        else:
            assert old[key] == config[key], f"Continuation changes {key}"
    assert old["scheduled_steps"] == shared["args"]["steps"]
    assert config["scheduled_steps"] == args["steps"]
    assert 0 < shared["step"] < args["wsd_end_step"] <= args["steps"]
    assert args["wsd_decay_start"] == -1 or args["wsd_decay_start"] >= shared["step"], (
        "Cannot change past LR updates"
    )
    assert shared["tokens"] == local["data"]["seen"] * args["row_tokens"]
    assert shared["step"] <= shared["args"]["wsd_end_step"]
    assert local["manager"]["schedule_step"] == shared["step"] - 1
    assert Path(pointer).resolve().parent != Path(output).resolve()
    assert not (Path(output) / "last.pt").exists(), (
        "Continue into a fresh run; use resume thereafter"
    )
    # Tensor states, moments, gradients, RNG and loader state are unchanged.
    migrated = dict(local, manager=dict(local["manager"], config=dict(config)))
    provenance = dict(
        parent_pointer=str(Path(pointer).resolve()),
        parent_step=shared["step"],
        parent_tokens=shared["tokens"],
        parent_useful_training_flops=shared["useful_training_flops"],
        parent_config=old,
        parent_source_sha256=shared["source_sha256"],
        previous=shared.get("continuation_provenance"),
        changed_configuration_fields=["scheduled_steps"]
        if old["scheduled_steps"] != config["scheduled_steps"]
        else [],
        accounting="Inherited tokens/model FLOPs include the already-paid prefix; new allocation elapsed time starts at zero.",
    )
    return migrated, provenance


def migratable(key, old, new):
    """--resume-new-source: another study's byte-identical history counts, or an arch that differs only
    in switches that guard numerics (modded_arch.RESUMABLE)."""
    if key == "mix_history":
        return bool(old and new) and Path(old).read_bytes() == Path(new).read_bytes()
    if key == "arch":
        o, n = (modded_arch.resolve(json.loads(x)) for x in (old, new))
        return all(o[k] == n[k] for k in o if k not in modded_arch.RESUMABLE)
    return False


def init_triton_signals():
    """Let LLVM register its handlers before installing the trainer's callbacks.

    Triton 3.6's first MLIR pass pipeline installs process-wide native signal
    handlers, including USR1 and TERM. Lazy compilation otherwise replaces our
    Python handlers. An empty pipeline triggers that one-time registration now;
    later pipelines leave the handlers installed below intact. No GPU is used.
    """
    from triton._C.libtriton import ir

    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = ir.builder(ctx).create_module()
    ir.pass_manager(ctx).run(mod, "initialize-signal-handlers")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--steps", type=int, default=4700, help="Scheduled steps")
    p.add_argument("--initial-batch-rows", type=int, default=32, help="Global rows")
    p.add_argument("--micro-batch", type=int, default=8, help="Rows per micro-batch")
    p.add_argument(
        "--row-tokens",
        type=int,
        default=1024,
        help="Training row length; --initial-batch-rows stays in 1024-token rows",
    )
    p.add_argument("--lr-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument(
        "--keep-checkpoints",
        type=int,
        default=0,
        help="0 keeps all; otherwise retain latest N>=2 plus best",
    )
    p.add_argument("--val-rows", type=int, default=1024)
    p.add_argument("--max-seconds", type=int, default=3600)
    p.add_argument("--stop-after", type=int, default=0)
    p.add_argument(
        "--profile",
        type=int,
        default=0,
        help="profile this step (CUDA kernels) into the log",
    )
    p.add_argument("--resume")
    p.add_argument(
        "--resume-new-source",
        action="store_true",
        help="resume on another training source, arch switches in modded_arch.RESUMABLE or another study's "
        "copy of the history counts (recorded as source_changes)",
    )
    p.add_argument(
        "--data",
        default="/scratch/yimingz3/allie/lichess_tokens_v2",
        help="packed corpus of the validation rows",
    )
    p.add_argument("--mix", required=True, help="chessmix policy")
    p.add_argument("--mix-stores", default="", help="comma-separated data-v1 stores")
    p.add_argument("--mix-pool-frac", type=float, default=1.0)
    p.add_argument(
        "--mix-history", default="", help="all-history bucket counts for repetition"
    )
    p.add_argument(
        "--mix-months", default="", help="comma-separated month dirs (else all built)"
    )
    p.add_argument(
        "--attn-kernel",
        action="store_true",
        help="causal window/game attention as a Triton FlashAttention-2 kernel (modded_attn), not FlexAttention",
    )
    p.add_argument(
        "--ckpt",
        default="none",
        choices=("none", "eager"),
        help="eager: each block and its residual blend recomputed in backward",
    )
    p.add_argument(
        "--ckpt-frac",
        type=float,
        default=1.0,
        help="with --ckpt eager: checkpoint only the first ceil(frac * layers) blocks",
    )
    p.add_argument(
        "--arch", default="{}", help="model-track switches, JSON (modded_arch.DEFAULTS)"
    )
    p.add_argument(
        "--clock-feats",
        type=int,
        default=0,
        choices=(0, 1, 3),
        help="continuous clock features: 1 = own time left, 3 = + opponent's, previous think",
    )
    p.add_argument(
        "--input-lr-mul", type=float, default=75.0, help="clock feature table lr mul"
    )
    p.add_argument("--aux-time", type=float, default=0.0, help="think-time CE weight")
    p.add_argument("--aux-wdl", type=float, default=0.0, help="game-outcome CE weight")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument(
        "--wsd-schedule", required=True, help="Frozen JSON absolute schedule"
    )
    p.add_argument("--wsd-end-step", type=int, required=True)
    p.add_argument("--wsd-decay-start", type=int, default=-1)
    p.add_argument("--wsd-fork-from")
    p.add_argument("--wsd-continue-from")
    p.add_argument("--wsd-fork-steps", default="")
    a = p.parse_args()
    schedule = Schedule(**json.loads(a.wsd_schedule))
    schedule.validate()
    assert a.initial_batch_rows == schedule.batch_rows
    assert 0 < a.wsd_end_step <= a.steps
    assert sum(bool(x) for x in (a.resume, a.wsd_fork_from, a.wsd_continue_from)) <= 1
    assert a.resume or not a.resume_new_source
    fork_steps = (
        set(map(int, a.wsd_fork_steps.split(","))) if a.wsd_fork_steps else set()
    )
    assert all(0 < s <= a.wsd_end_step for s in fork_steps)
    if a.wsd_decay_start >= 0:
        assert schedule.warmup_steps <= a.wsd_decay_start < a.wsd_end_step
        assert not fork_steps
    start = time.monotonic()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    snap = os.environ.get("MEMSNAP")
    if snap:
        from modded_memsnap import start as start_memory_trace

        start_memory_trace(snap, rank)
    dist.init_process_group("nccl")
    # host-side flags: never queue behind in-flight NCCL work
    cpu = dist.new_group(backend="gloo")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(a.deterministic)
    # deterministic mode NaN-fills every fresh allocation (~1.5K fills/step); skipping it was bit-identical on the
    # reference config; new kernels keep a poisoned-allocation test
    torch.utils.deterministic.fill_uninitialized_memory = False
    modded_medium.ATTENTION = "triton" if a.attn_kernel else "flex"
    torch.backends.cuda.matmul.allow_tf32 = True
    assert a.initial_batch_rows * 1024 % (a.micro_batch * world * a.row_tokens) == 0
    assert a.micro_batch * a.row_tokens % 1024 == 0
    assert min(a.eval_every, a.checkpoint_every, a.val_rows, a.micro_batch) > 0
    assert a.keep_checkpoints == 0 or a.keep_checkpoints >= 2
    assert a.name and all(c.isalnum() or c in "-_" for c in a.name)
    out = ROOT / "results/pretrain" / a.name
    if rank == 0:
        out.mkdir(exist_ok=True, parents=True)
    dist.barrier()
    cfg = Config(
        width=a.width,
        layers=a.layers,
        head_dim=a.head_dim,
        max_tokens=a.micro_batch * a.row_tokens,
        scheduled_steps=a.steps,
        lr_scale=a.lr_scale,
        feats=a.clock_feats,
        input_lr_mul=a.input_lr_mul,
        ckpt=a.ckpt,
        ckpt_frac=a.ckpt_frac,
        arch=json.loads(a.arch),
    )
    aux = bool(a.aux_time or a.aux_wdl)
    torch.manual_seed(a.seed)
    model = create_model(cfg)
    manager = TrainingManager(model, cfg, schedule, a.wsd_decay_start, a.wsd_end_step)
    net = torch.compile(model, dynamic=False, fullgraph=a.ckpt != "eager")
    from chessmix import (
        ROW,
        Prefetch,
        Sampler,
    )  # its data dependencies only where it samples

    kw = dict(pool_frac=a.mix_pool_frac, aux=aux, feats=bool(a.clock_feats))
    kw["row"] = a.row_tokens + 1
    if a.mix_stores:
        kw["stores"] = a.mix_stores.split(",")
    if a.mix_history:
        kw["history"] = a.mix_history
    if a.mix_months:
        kw["months"] = a.mix_months.split(",")
    train = Prefetch(
        Sampler(
            a.mix,
            a.seed,
            total_rows=a.steps * a.initial_batch_rows * 1024 // a.row_tokens,
            **kw,
        )
    )
    val = Packed(a.data, "val")
    val_idx = np.random.default_rng(20260910).choice(
        int(val.ends[-1]), min(a.val_rows, int(val.ends[-1])), replace=False
    )
    vrows = val.rows(val_idx) if rank == 0 else None
    first, best, elapsed_prior, flops_local = 0, float("inf"), 0.0, 0
    runtime = dict(
        torch=torch.__version__,
        triton=triton.__version__,
        cuda=torch.version.cuda,
        torch_source_key=RUNTIME_SOURCE_KEY,
    )
    hashed = ("lm_data.py", "lm_checkpoint.py", "chessmix.py", "chess_vocab.py")
    source_hashes = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [*SOURCE.glob("modded_*.py"), *(SOURCE / n for n in hashed)]
    }
    torch.manual_seed(a.seed + rank)
    np.random.seed(a.seed + rank)
    random.seed(a.seed + rank)
    continuation_provenance, source_changes = None, []
    load_path = a.resume or a.wsd_fork_from or a.wsd_continue_from
    if load_path:
        pointer = torch.load(load_path, weights_only=False, map_location="cpu")
        assert pointer["format"] == "allie-modded-medium-1"
        directory = Path(load_path).resolve().parent / pointer["directory"]
        shared = torch.load(
            directory / "model.pt", weights_only=False, map_location="cpu"
        )
        local = torch.load(
            directory / f"rank{rank}.pt", weights_only=False, map_location="cpu"
        )
        for key in SAME:
            same = shared["args"][key] == vars(a)[key]
            if not same and a.resume_new_source:
                same = migratable(key, shared["args"][key], vars(a)[key])
            assert same, f"Resume changes {key}"
        old_row = shared["args"].get("row_tokens", 1024)
        assert old_row == a.row_tokens, "Resume changes row_tokens"
        source_changes = shared.get("source_changes", [])
        if shared["source_sha256"] != source_hashes or shared["args"]["arch"] != a.arch:
            assert a.resume_new_source, "Resume requires the frozen training source"
            source_changes = [
                *source_changes,
                dict(
                    step=shared["step"],
                    parent_source_sha256=shared["source_sha256"],
                    parent_arch=shared["args"]["arch"],
                    parent_mix_history=shared["args"]["mix_history"],
                ),
            ]
            # the rank state's configuration differs from this run's only in RESUMABLE arch switches
            local["manager"]["config"] = dict(local["manager"]["config"], arch=cfg.arch)
        assert shared["runtime"] == runtime, (
            "Exact continuation requires the same PyTorch/Triton/CUDA runtime"
        )
        if a.wsd_continue_from:
            local, continuation_provenance = continuation(
                shared, local, vars(a), asdict(cfg), out, load_path
            )
        else:
            assert shared["args"]["steps"] == a.steps, "Resume changes steps"
            continuation_provenance = shared.get("continuation_provenance")
            if a.wsd_fork_from:
                assert shared["args"]["wsd_decay_start"] == -1, (
                    "Only stable mainlines can be forked"
                )
                assert shared["step"] == a.wsd_decay_start, (
                    "Fork must begin at its exact decay boundary"
                )
                assert not (out / "last.pt").exists(), (
                    "Cannot fork over an existing run"
                )
                assert Path(load_path).resolve().parent != out.resolve()
            else:
                for key in (
                    "name",
                    "wsd_end_step",
                    "wsd_decay_start",
                    "wsd_fork_steps",
                ):
                    assert shared["args"][key] == vars(a)[key], f"Resume changes {key}"
        model.load_state_dict(shared["model"])
        manager.load_rank_state_dict(local["manager"])
        train.load_state_dict(local["data"])
        restore_rng(local["rng"])
        first, best = shared["step"], shared["best_move_ce"]
        elapsed_prior, flops_local = (
            shared["elapsed_seconds"],
            local["useful_training_flops"],
        )
        if a.wsd_fork_from or a.wsd_continue_from:
            # Endpoint model compute includes prefix; branch allocation time does not.
            elapsed_prior, best = 0.0, float("inf")
    elif (out / "last.pt").exists():
        raise ValueError("Existing checkpoint requires explicit resume")
    termination = [False]

    def stop_handler(*_):
        termination[0] = True

    init_triton_signals()
    signal.signal(signal.SIGUSR1, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    log = RowLog(out) if rank == 0 else None

    def append(file, row):
        if rank == 0:
            log.put(file, json.dumps(row) + "\n")

    metadata = dict(
        args=vars(a),
        config=config_dict(cfg),
        world_size=world,
        parameters=sum(p.numel() for p in model.parameters()),
        source_sha256=source_hashes,
        runtime=runtime,
        dataset=json.loads((ROOT / "results/original-data.json").read_text()),
        train_rows=int(train.ends[-1]),
        train_shards=len(train.paths),
        mix_infeasible_buckets=list(train.infeasible),
        mix_missing_history_games=train.missing,
        mix_months=list(train.months),
        val_indices=val_idx.tolist(),
        continuation_provenance=continuation_provenance,
        source_changes=source_changes,
        job_id=os.environ.get("SLURM_JOB_ID"),
        compute_accounting="Useful model matmul FLOPs; excludes optimizer, elementwise and padded attention kernel work",
        gradient_normalization="Global summed objective /8, matching upstream fixed grad_accum_steps=8/world_size; physical microbatch accumulation does not change normalization",
    )
    if rank == 0:
        (out / ("resume-config.json" if a.resume else "config.json")).write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print(json.dumps(metadata | {"val_indices": len(val_idx)}), flush=True)

    # checkpoints are written from a thread; pointers move once every rank's files are durable
    saver = AsyncSaver(cpu)

    def save(step, metrics):
        checkpoint_start = time.monotonic()
        waited = saver.flush()
        total_flops = torch.tensor(flops_local, device="cuda", dtype=torch.float64)
        dist.all_reduce(total_flops)
        identifier = [f"step-{step:08d}-{uuid.uuid4().hex[:8]}" if rank == 0 else None]
        dist.broadcast_object_list(identifier, 0)
        directory = out / "checkpoints" / identifier[0]
        if rank == 0:
            directory.mkdir(parents=True)
        dist.barrier()
        # Save each rank independently: its sharded moments and pending grads
        # cannot be reconstructed from rank0's optimizer state.
        saver.add(
            dict(
                manager=manager.rank_state_dict(saver.snapshot),
                rng=rng_state(),
                data=train.state_dict(),
                useful_training_flops=flops_local,
            ),
            directory / f"rank{rank}.pt",
        )
        sync_params()
        if rank == 0:
            inference = dict(
                split_embed=model.split_embed,
                ws_short=manager.ws_short,
                ws_long=manager.ws_long,
                yarn=dict(
                    angular_freq=model.yarn.angular_freq,
                    cos=model.yarn.cos,
                    sin=model.yarn.sin,
                    attn_scale=model.yarn.attn_scale,
                ),
            )
            saver.add(
                dict(
                    model=saver.snapshot(model.state_dict()),
                    config=asdict(cfg),
                    args=vars(a),
                    source_sha256=source_hashes,
                    runtime=runtime,
                    step=step,
                    best_move_ce=best,
                    metrics=metrics,
                    tokens=train.seen * a.row_tokens,
                    useful_training_flops=total_flops.item(),
                    inference=saver.snapshot(inference),
                    continuation_provenance=continuation_provenance,
                    source_changes=source_changes,
                    elapsed_seconds=elapsed_prior + time.monotonic() - start,
                ),
                directory / "model.pt",
            )
        names = ["last.pt"]
        if step in fork_steps:
            names.append(f"fork-{step}.pt")
        if metrics is not None and metrics["move_ce"] <= best:
            names.append("best.pt")
        torch.cuda.synchronize()  # completes the snapshot's pinned copies
        seconds = time.monotonic() - checkpoint_start

        def commit():
            publish(out, directory, step, world, names)
            removed = prune_checkpoints(out, a.keep_checkpoints, directory)
            row = {
                "step": step,
                "seconds": seconds,
                "wait_seconds": waited,
                "durable_seconds": time.monotonic() - checkpoint_start,
                "host_bytes": saver.nbytes,
                "directory": str(directory.relative_to(out)),
                "removed": removed,
            }
            # written here, not by the log thread: prune() finds directories through this
            # file, and no checkpoint write is in flight once commit runs
            line = json.dumps(row) + "\n"
            with (out / "checkpoints.jsonl").open("a") as f:
                f.write(line)
            log.put(None, line)

        saver.start(commit if rank == 0 else None)

    total_steps = a.wsd_end_step
    window_start, window_tokens, wait0 = time.monotonic(), 0, train.waited
    primary_sum = torch.zeros((), device="cuda")
    count_sum = torch.zeros((), device="cuda")
    aux_sum = torch.zeros(4, device="cuda")  # time NLL, count, wdl NLL, count
    stop_reason = "steps"
    step = first
    # MEMSNAP also dumps early OOMs; successful diagnostics stop before step3.
    for index in range(first, total_steps):
        if snap and index == first + 2:
            if rank == 0:
                torch.cuda.memory._dump_snapshot(snap)
            dist.barrier()
            os._exit(0)
        if a.profile and index + 1 == a.profile:
            shapes = bool(os.environ.get("ALLIE_SHAPES"))
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
                + [torch.profiler.ProfilerActivity.CPU] * shapes,
                record_shapes=shapes,
            )
            prof.__enter__()
        manager.advance_schedule(index)
        accum = manager.batch_size // (world * a.micro_batch * a.row_tokens)
        assert accum * world * a.micro_batch * a.row_tokens == manager.batch_size > 0
        for micro in range(accum):
            rows = train.batch(a.micro_batch, rank, world)
            flops_local += useful_flops(
                rows, cfg, manager.ws_short * 128, manager.ws_long * 128
            )
            data = to_gpu(rows)
            x, y = data[:, :-1], data[:, 1:]
            context = make_context(
                x,
                manager.ws_short * 128,
                manager.ws_long * 128,
                host=rows[:, :-1],
                span=ROW,
            )
            if micro == accum - 1:
                manager.activate_hooks(index)
            feat = train.last.get("feat")
            if feat is not None:
                feat = to_gpu(feat[:, :-1].astype(np.int64)).flatten(0, 1)
            logits = net(
                x.flatten(),
                y.flatten(),
                context,
                manager.get_forward_args(),
                feat_seq=feat,
            )
            mask = to_gpu(train.last["mask"][:, 1:])
            loss, primary, count = move_losses(
                logits, x, y, context, manager.mtp_weights, mask
            )
            if aux:
                t, w = (
                    to_gpu(train.last[k][:, :-1].astype(np.int64)).flatten()
                    for k in ("time", "wdl")
                )
                parts = aux_losses(logits, t, w, mask.flatten())
                loss = loss + a.aux_time * parts[0] + a.aux_wdl * parts[2]
                aux_sum += torch.stack([p.detach().float() for p in parts])
            # Upstream uses fixed grad_accum_steps=8/world_size while growing
            # the physical microbatch. We grow the number of microbatches to
            # bound activation memory, but must retain its gradient scaling.
            (loss * (world / 8)).backward()
            primary_sum += primary.detach()
            count_sum += count
            window_tokens += rows.shape[0] * world * a.row_tokens
        manager.step_optimizers(index)
        step = index + 1
        saver.poll()
        if a.profile and step == a.profile:
            torch.cuda.synchronize()
            prof.__exit__(None, None, None)
            if rank == 0:
                print(
                    prof.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=40
                    ),
                    flush=True,
                )
                if os.environ.get("ALLIE_TRACE"):
                    prof.export_chrome_trace(os.environ["ALLIE_TRACE"])
                if shapes:
                    print(
                        prof.key_averages(group_by_input_shape=True).table(
                            sort_by="cuda_time_total",
                            row_limit=60,
                            max_name_column_width=40,
                        ),
                        flush=True,
                    )
        if step % 25 == 0 or step == total_steps:
            stats = torch.cat(
                (
                    primary_sum.double()[None],
                    count_sum.double()[None],
                    torch.tensor([flops_local], device="cuda", dtype=torch.float64),
                    aux_sum.double(),
                )
            )
            dist.all_reduce(stats)
            torch.cuda.synchronize()
            assert torch.isfinite(stats[0]), "Nonfinite move CE"
            dt = time.monotonic() - window_start
            append(
                "train.jsonl",
                dict(
                    step=step,
                    train_ce=(stats[0] / stats[1]).item(),
                    tokens=train.seen * a.row_tokens,
                    tokens_per_second=window_tokens / dt,
                    sampler_wait=(train.waited - wait0) / dt,
                    seconds=elapsed_prior + time.monotonic() - start,
                    useful_training_flops=stats[2].item(),
                    global_batch_tokens=manager.batch_size,
                    split_embed=model.split_embed,
                    windows=[manager.ws_short * 128, manager.ws_long * 128],
                    max_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
                    **moe_stats(manager),
                    **(
                        dict(
                            time_ce=(stats[3] / stats[4]).item(),
                            wdl_ce=(stats[5] / stats[6]).item(),
                        )
                        if aux
                        else {}
                    ),
                ),
            )
            window_start, window_tokens, wait0 = time.monotonic(), 0, train.waited
            primary_sum.zero_()
            count_sum.zero_()
            aux_sum.zero_()
        stop_code = 0
        if (
            step % 5 == 0
            or step % a.eval_every == 0
            or step % a.checkpoint_every == 0
            or step == total_steps
            or step == a.stop_after
        ):
            stop_code = (
                2
                if termination[0]
                else int(
                    time.monotonic() - start >= a.max_seconds
                    or (a.stop_after and step >= a.stop_after)
                )
            )
            flag = torch.tensor(stop_code, device="cpu")
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=cpu)
            stop_code = int(flag)
        metrics = None
        if stop_code != 2 and (
            step % a.eval_every == 0 or step == total_steps or stop_code
        ):
            total_flops = torch.tensor(flops_local, device="cuda", dtype=torch.float64)
            dist.all_reduce(total_flops)
            if rank == 0:
                metrics = evaluate(
                    net, manager, vrows, a.micro_batch * a.row_tokens // 1024
                )
                best = min(best, metrics["move_ce"])
                append(
                    "validation.jsonl",
                    dict(
                        step=step,
                        seconds=elapsed_prior + time.monotonic() - start,
                        tokens=train.seen * a.row_tokens,
                        useful_training_flops=total_flops.item(),
                        **metrics,
                    ),
                )
            dist.barrier()
        if (
            metrics is not None
            or step % a.eval_every == 0
            or step % a.checkpoint_every == 0
            or step == total_steps
            or step in fork_steps
            or stop_code
        ):
            save(step, metrics)
            window_start, window_tokens, wait0 = time.monotonic(), 0, train.waited
            primary_sum.zero_()
            count_sum.zero_()
            aux_sum.zero_()
        if stop_code:
            stop_reason = (
                "signal"
                if stop_code == 2
                else "stop_after"
                if a.stop_after and step >= a.stop_after
                else "wall_clock_cap"
            )
            break
    saver.flush()
    if rank == 0:
        log.close()
        (out / "done.json").write_text(
            json.dumps(
                dict(
                    step=step,
                    stop_reason=stop_reason,
                    best_move_ce=best,
                    seconds=elapsed_prior + time.monotonic() - start,
                    tokens_processed=train.seen * a.row_tokens,
                ),
                indent=2,
            )
            + "\n"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if os.environ.get("MEMSNAP"):
            from modded_memsnap import dump_on_error

            dump_on_error()
        raise
