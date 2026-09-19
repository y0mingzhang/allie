"""Recoverable training of the pinned medium recipe on original chess rows."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
import uuid

import numpy as np
import torch
import torch.distributed as dist
import triton
from modded_arch import attn_factor, extra_flops
from modded_medium import (
    Config,
    TrainingManager,
    create_model,
    make_context,
    move_losses,
    aux_losses,
    elo_buckets,
    ratings,
    cpu_copy,
    config_dict,
    core,
    RUNTIME_SOURCE_KEY,
)
from lm_data import Packed
from lm_checkpoint import atomic_save, rng_state, restore_rng
from modded_checkpoints import prune as prune_checkpoints

ROOT = Path(os.environ.get("ALLIE_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
SOURCE = Path(__file__).resolve().parent


@torch.inference_mode()
def evaluate(model, manager, rows, batch):
    model.eval()
    sums = torch.zeros(6, dtype=torch.float64, device="cuda")
    for lo in range(0, len(rows), batch):
        data = rows[lo : lo + batch]
        ids = torch.as_tensor(data, device="cuda")
        x, y = ids[:, :-1], ids[:, 1:]
        context = make_context(x, manager.ws_short * 128, manager.ws_long * 128)
        elo = (
            torch.as_tensor(elo_buckets(data), device="cuda").flatten()
            if getattr(model, "use_elo", False)
            else None
        )
        logits = model(
            x.flatten(), y.flatten(), context, manager.get_forward_args(), elo_seq=elo
        )
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
    """Per MoE layer: last step's max / mean and min / mean tokens per expert, experts under 10% of
    the mean, dropped fraction, bias norm."""
    if not manager.moe:
        return {}
    stats = torch.stack([m.stats for m in manager.moe]).tolist()
    return dict(
        moe_imbalance=[round(x[0], 3) for x in stats],
        moe_min_load=[round(x[1], 3) for x in stats],
        moe_starved=[int(x[2]) for x in stats],
        moe_dropped=[round(x[3], 4) for x in stats],
        moe_bias_norm=[round(m.bias.norm().item(), 4) for m in manager.moe],
    )


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
    per_token += extra_flops(cfg.arch, d, layers)  # model-track branches (0 by default)
    pairs = (layers - long) * short_pairs + long * long_pairs
    return 3 * (x.size * per_token + 4 * d * pairs * attn_factor(cfg.arch))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument(
        "--steps", type=int, default=4700, help="Scheduled steps before extension"
    )
    p.add_argument("--extension-steps", type=int, default=40)
    p.add_argument(
        "--initial-batch-rows", type=int, default=32, help="Global rows; ramps 1:2:3:4"
    )
    p.add_argument("--micro-batch", type=int, default=8)
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
    p.add_argument("--resume")
    p.add_argument("--data", default="/scratch/yimingz3/allie/lichess_tokens_v2")
    p.add_argument(
        "--mix", default="", help="chessmix policy; empty trains on the packed corpus"
    )
    p.add_argument("--mix-stores", default="", help="comma-separated data-v1 stores")
    p.add_argument("--mix-pool-frac", type=float, default=1.0)
    p.add_argument(
        "--mix-history", default="", help="all-history bucket counts for repetition"
    )
    p.add_argument(
        "--mix-months", default="", help="comma-separated month dirs (else all built)"
    )
    p.add_argument(
        "--clock",
        action="store_true",
        help="feed each mover's remaining clock (needs --mix)",
    )
    p.add_argument("--elo", action="store_true", help="feed each mover's rating bucket")
    p.add_argument("--no-value-embeds", action="store_true")
    p.add_argument(
        "--no-skips", action="store_true", help="drop skip connections and backout"
    )
    p.add_argument("--no-smear", action="store_true")
    p.add_argument(
        "--wd-scale", type=float, default=1.0, help="scales every weight decay"
    )
    p.add_argument(
        "--doc-rope", action="store_true", help="rotary position within game"
    )
    p.add_argument(
        "--rope-fp32", action="store_true", help="FP32 rotary cos/sin tables"
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
        "--input-lr-mul", type=float, default=75.0, help="clock/Elo table lr mul"
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
    from modded_wsd import Schedule, install

    schedule = Schedule(**json.loads(a.wsd_schedule))
    schedule.validate()
    assert a.extension_steps == 0 and a.initial_batch_rows == schedule.batch_rows
    assert 0 < a.wsd_end_step <= a.steps
    assert sum(bool(x) for x in (a.resume, a.wsd_fork_from, a.wsd_continue_from)) <= 1
    fork_steps = (
        set(map(int, a.wsd_fork_steps.split(","))) if a.wsd_fork_steps else set()
    )
    assert all(0 < s <= a.wsd_end_step for s in fork_steps)
    if a.wsd_decay_start >= 0:
        assert schedule.warmup_steps <= a.wsd_decay_start < a.wsd_end_step
        assert not fork_steps
    install(core, schedule, a.wsd_decay_start, a.wsd_end_step)
    start = time.monotonic()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(a.deterministic)
    torch.backends.cuda.matmul.allow_tf32 = True
    assert a.initial_batch_rows % (a.micro_batch * world) == 0
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
        max_tokens=a.micro_batch * 1024,
        scheduled_steps=a.steps,
        extension_steps=a.extension_steps,
        initial_batch_rows=a.initial_batch_rows,
        lr_scale=a.lr_scale,
    )
    aux = bool(a.aux_time or a.aux_wdl)
    assert a.mix or not (a.clock or aux), (
        "--clock and aux heads need the chessmix sampler"
    )
    cfg.clock, cfg.elo, cfg.input_lr_mul = a.clock, a.elo, a.input_lr_mul
    cfg.feats = a.clock_feats
    cfg.doc_rope, cfg.rope_fp32 = a.doc_rope, a.rope_fp32
    cfg.arch = json.loads(a.arch)
    cfg.value_embeds, cfg.skips, cfg.smear = (
        not a.no_value_embeds,
        not a.no_skips,
        not a.no_smear,
    )
    torch.manual_seed(a.seed)
    model = create_model(cfg)
    manager = TrainingManager(model, cfg)
    manager.split_step = schedule.split_step
    if a.wd_scale != 1:
        for opt in manager.optimizers:
            for group in opt.param_groups:
                group["weight_decay"] *= a.wd_scale
    net = torch.compile(model, dynamic=False, fullgraph=True)
    if a.mix:
        from chessmix import Prefetch, Sampler

        kw = dict(
            pool_frac=a.mix_pool_frac, clock=a.clock, aux=aux, feats=bool(a.clock_feats)
        )
        if a.mix_stores:
            kw["stores"] = a.mix_stores.split(",")
        if a.mix_history:
            kw["history"] = a.mix_history
        if a.mix_months:
            kw["months"] = a.mix_months.split(",")
        train = Prefetch(
            Sampler(a.mix, a.seed, total_rows=a.steps * a.initial_batch_rows, **kw)
        )
    else:
        train = Packed(a.data, "train", a.seed)
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
    source_hashes = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [
            *SOURCE.glob("modded_*.py"),
            SOURCE / "lm_data.py",
            SOURCE / "lm_checkpoint.py",
            *([SOURCE / "chessmix.py", SOURCE / "chess_vocab.py"] if a.mix else []),
        ]
    }
    torch.manual_seed(a.seed + rank)
    np.random.seed(a.seed + rank)
    random.seed(a.seed + rank)
    continuation_provenance = None
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
        for key in (
            "width",
            "layers",
            "head_dim",
            "extension_steps",
            "initial_batch_rows",
            "micro_batch",
            "lr_scale",
            "seed",
            "deterministic",
            "wsd_schedule",
        ):
            assert shared["args"][key] == vars(a)[key], f"Resume changes {key}"
        for key in (  # added with the data/model tracks; absent in older checkpoints
            "mix",
            "mix_stores",
            "mix_pool_frac",
            "mix_history",
            "mix_months",
            "clock",
            "elo",
            "clock_feats",
            "input_lr_mul",
            "aux_time",
            "aux_wdl",
            "no_value_embeds",
            "no_skips",
            "no_smear",
            "wd_scale",
            "doc_rope",
            "rope_fp32",
            "arch",
        ):
            assert shared["args"].get(key, vars(a)[key]) == vars(a)[key], (
                f"Resume changes {key}"
            )
        if a.wsd_continue_from:
            from modded_continuation import prepare_continuation

            local, continuation_provenance = prepare_continuation(
                shared,
                local,
                vars(a),
                asdict(cfg),
                source_hashes,
                runtime,
                out,
                load_path,
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
            assert shared["source_sha256"] == source_hashes, (
                "Resume requires the frozen training source"
            )
            assert shared["runtime"] == runtime, (
                "Exact continuation requires the same PyTorch/Triton/CUDA runtime"
            )
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

    signal.signal(signal.SIGUSR1, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    def append(file, row):
        if rank == 0:
            with (out / file).open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

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
        mix_infeasible_buckets=list(getattr(train, "infeasible", [])),
        mix_missing_history_games=getattr(train, "missing", 0),
        mix_months=list(getattr(train, "months", [])),
        val_indices=val_idx.tolist(),
        continuation_provenance=continuation_provenance,
        job_id=os.environ.get("SLURM_JOB_ID"),
        compute_accounting="Useful model matmul FLOPs; excludes optimizer, elementwise and padded attention kernel work",
        gradient_normalization="Global summed objective /8, matching upstream fixed grad_accum_steps=8/world_size; physical microbatch accumulation does not change normalization",
    )
    if rank == 0:
        (out / ("resume-config.json" if a.resume else "config.json")).write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print(json.dumps(metadata | {"val_indices": len(val_idx)}), flush=True)

    def save(step, metrics):
        checkpoint_start = time.monotonic()
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
        atomic_save(
            dict(
                manager=manager.rank_state_dict(),
                rng=rng_state(),
                data=train.state_dict(),
                useful_training_flops=flops_local,
            ),
            directory / f"rank{rank}.pt",
        )
        if rank == 0:
            atomic_save(
                dict(
                    model=cpu_copy(model.state_dict()),
                    config=asdict(cfg),
                    args=vars(a),
                    source_sha256=source_hashes,
                    runtime=runtime,
                    step=step,
                    best_move_ce=best,
                    metrics=metrics,
                    tokens=train.seen * 1024,
                    useful_training_flops=total_flops.item(),
                    inference=cpu_copy(
                        dict(
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
                    ),
                    continuation_provenance=continuation_provenance,
                    elapsed_seconds=elapsed_prior + time.monotonic() - start,
                ),
                directory / "model.pt",
            )
        dist.barrier()
        if rank == 0:
            pointer = dict(
                format="allie-modded-medium-1",
                directory=str(directory.relative_to(out)),
                step=step,
                world_size=world,
            )
            atomic_save(pointer, out / "last.pt")
            if step in fork_steps:
                atomic_save(pointer, out / f"fork-{step}.pt")
            if metrics is not None and metrics["move_ce"] <= best:
                atomic_save(pointer, out / "best.pt")
        dist.barrier()
        removed = (
            prune_checkpoints(out, a.keep_checkpoints, directory) if rank == 0 else []
        )
        dist.barrier()
        append(
            "checkpoints.jsonl",
            dict(
                step=step,
                seconds=time.monotonic() - checkpoint_start,
                directory=str(directory.relative_to(out)),
                removed=removed,
            ),
        )

    total_steps = a.wsd_end_step
    train_wait = lambda: getattr(train, "waited", 0.0)
    window_start, window_tokens, wait0 = time.monotonic(), 0, train_wait()
    primary_sum = torch.zeros((), device="cuda")
    count_sum = torch.zeros((), device="cuda")
    aux_sum = torch.zeros(4, device="cuda")  # time NLL, count, wdl NLL, count
    stop_reason = "steps"
    step = first
    for index in range(first, total_steps):
        manager.advance_schedule(index)
        accum = manager.batch_size // (world * a.micro_batch * 1024)
        assert accum > 0 and accum * world * a.micro_batch * 1024 == manager.batch_size
        core.grad_accum_steps = accum
        for micro in range(accum):
            rows = train.batch(a.micro_batch, rank, world)
            flops_local += useful_flops(
                rows, cfg, manager.ws_short * 128, manager.ws_long * 128
            )
            data = to_gpu(rows)
            x, y = data[:, :-1], data[:, 1:]
            context = make_context(
                x, manager.ws_short * 128, manager.ws_long * 128, host=rows[:, :-1]
            )
            if micro == accum - 1:
                manager.activate_hooks(index)
            last = getattr(train, "last", {})
            clock = last.get("clock")
            feat = last.get("feat")
            if feat is not None:
                feat = to_gpu(feat[:, :-1].astype(np.int64)).flatten(0, 1)
            extra = (
                ()
                if clock is None
                else (to_gpu(clock[:, :-1].astype(np.int64)).flatten(),)
            )
            elo = to_gpu(elo_buckets(rows)).flatten() if a.elo else None
            logits = net(
                x.flatten(),
                y.flatten(),
                context,
                manager.get_forward_args(),
                *extra,
                elo_seq=elo,
                feat_seq=feat,
            )
            mask = last.get("mask")
            mask = None if mask is None else to_gpu(mask[:, 1:])
            loss, primary, count = move_losses(
                logits, x, y, context, manager.mtp_weights, mask
            )
            if aux:
                t, w = (
                    to_gpu(last[k][:, :-1].astype(np.int64)).flatten()
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
            window_tokens += rows.shape[0] * world * 1024
        manager.step_optimizers(index)
        step = index + 1
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
                    tokens=train.seen * 1024,
                    tokens_per_second=window_tokens / dt,
                    sampler_wait=(train_wait() - wait0) / dt,
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
            window_start, window_tokens, wait0 = time.monotonic(), 0, train_wait()
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
            flag = torch.tensor(stop_code, device="cuda")
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            stop_code = int(flag.item())
        metrics = None
        if stop_code != 2 and (
            step % a.eval_every == 0 or step == total_steps or stop_code
        ):
            total_flops = torch.tensor(flops_local, device="cuda", dtype=torch.float64)
            dist.all_reduce(total_flops)
            if rank == 0:
                metrics = evaluate(net, manager, vrows, a.micro_batch)
                best = min(best, metrics["move_ce"])
                append(
                    "validation.jsonl",
                    dict(
                        step=step,
                        seconds=elapsed_prior + time.monotonic() - start,
                        tokens=train.seen * 1024,
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
            window_start, window_tokens, wait0 = time.monotonic(), 0, train_wait()
            primary_sum.zero_()
            count_sum.zero_()
        if stop_code:
            stop_reason = (
                "signal"
                if stop_code == 2
                else "stop_after"
                if a.stop_after and step >= a.stop_after
                else "wall_clock_cap"
            )
            break
    if rank == 0:
        (out / "done.json").write_text(
            json.dumps(
                dict(
                    step=step,
                    stop_reason=stop_reason,
                    best_move_ce=best,
                    seconds=elapsed_prior + time.monotonic() - start,
                    tokens_processed=train.seen * 1024,
                ),
                indent=2,
            )
            + "\n"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
