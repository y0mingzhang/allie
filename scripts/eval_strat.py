"""Original validation and aligned development CE for a medium-track checkpoint.

Uses frozen model code from the run and its saved inference schedule/state.
Final test splits are deliberately not exposed by this entry point.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist
from modded_runtime import prime_source_key, ForwardTimer
from ce_alignment import KEYS as ALIGNMENT_KEYS, verify_alignment

prime_source_key()

ROOT = Path(os.environ.get("ALLIE_PROJECT_ROOT", Path(__file__).resolve().parents[1]))


def read_hashed(path):
    with path.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
        f.seek(0)
        state = torch.load(f, map_location="cpu", weights_only=False)
    return state, sha


def main(final=False):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--split",
        choices=["test", "test_expert"]
        if final
        else ["original_val", "dev", "dev_expert", "strat"],
        required=True,
    )
    if final:
        p.add_argument("--selection", required=True)
        p.add_argument("--selection-sha256", required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--source",
        help="Explicit frozen source directory; every file hash must match the checkpoint",
    )
    p.add_argument(
        "--match-training-validation",
        action="store_true",
        help="Verify the saved training validation selection, not full original validation",
    )
    p.add_argument(
        "--diagnose-restore",
        action="store_true",
        help="Report full optimizer/RNG/schedule restoration against saved CE; no pass claim",
    )
    p.add_argument(
        "--diagnose-evaluator-paths",
        action="store_true",
        help="Compare repeated forwards and input/reduction layouts; diagnostics only",
    )
    p.add_argument(
        "--audit-evaluator-paths",
        action="store_true",
        help="Record forward/input-mode diagnostics, then enforce the usual evaluation gates",
    )
    a = p.parse_args()
    selected = None
    shards = 4
    if final:
        from final_selection import load_selection

        _, selected, shards = load_selection(
            a.selection, a.selection_sha256, ROOT, a.checkpoint, a.split, a.batch
        )
        assert not any(
            (
                a.match_training_validation,
                a.diagnose_restore,
                a.diagnose_evaluator_paths,
                a.audit_evaluator_paths,
            )
        )
        if a.source is None:
            a.source = selected["source"]
    assert not a.diagnose_restore or (
        a.match_training_validation and a.split == "original_val"
    )
    assert not a.diagnose_evaluator_paths or (
        a.match_training_validation
        and a.split == "original_val"
        and not a.diagnose_restore
    )
    assert not a.audit_evaluator_paths or (
        a.match_training_validation
        and a.split == "original_val"
        and not a.diagnose_restore
        and not a.diagnose_evaluator_paths
    )
    print(json.dumps(dict(phase="load_checkpoint", split=a.split)), flush=True)
    checkpoint = Path(a.checkpoint).resolve()
    pointer, pointer_sha = read_hashed(checkpoint)
    assert pointer["format"] == "allie-modded-medium-1"
    run = checkpoint.parent
    state, model_sha = read_hashed(run / pointer["directory"] / "model.pt")
    assert "inference" in state, "Checkpoint needs saved inference schedule/buffers"
    source = Path(a.source).resolve() if a.source else run / "launch-source"
    assert source.exists(), "Evaluate the actual frozen run implementation"
    for name, expected in state["source_sha256"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, (
            name
        )
    if final:
        from final_selection import verify_selected_checkpoint

        verify_selected_checkpoint(
            selected, pointer_sha, model_sha, state, source, run / pointer["directory"]
        )
    sys.path.insert(0, str(source))
    from modded_medium import Config, create_model, make_context, core
    import modded_medium

    elo_buckets = getattr(
        modded_medium, "elo_buckets", None
    )  # absent in pre-Elo sources
    from lm_data import Packed
    from modded_train import ratings

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.use_deterministic_algorithms(state["args"].get("deterministic", False))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 1, (
        "This evaluator uses one GPU and no distributed optimizer"
    )
    cfg = Config(**state["config"])
    model = create_model(cfg)
    # Preserve padding parameters when evaluating an N-GPU checkpoint on one
    # GPU. The extra scalar entries are unused, but strict loading/counting
    # must retain the original tensor shape.
    model.scalars.data = state["model"]["scalars"].to(
        device="cuda", dtype=model.scalars.dtype
    )
    model.load_state_dict(state["model"])
    inference = state["inference"]
    model.split_embed = inference["split_embed"]
    for key in ("angular_freq", "cos", "sin"):
        getattr(model.yarn, key).copy_(inference["yarn"][key].to("cuda"))
    model.yarn.attn_scale = inference["yarn"]["attn_scale"]
    model.eval()
    print(json.dumps(dict(phase="model_restored", step=state["step"])), flush=True)
    net = torch.compile(model, dynamic=False, fullgraph=True)
    schedule = core.ForwardScheduleConfig(
        None, inference["ws_short"], inference["ws_long"]
    )
    batch = min(a.batch, cfg.max_tokens // 1024)
    assert batch > 0
    common = dict(
        checkpoint=str(checkpoint),
        checkpoint_sha256=pointer_sha,
        model_sha256=model_sha,
        step=state["step"],
        parameters=sum(p.numel() for p in model.parameters()),
        training_runtime=state["runtime"],
        inference_torch=torch.__version__,
        batch=batch,
        normalization="1968 original move IDs378..2345; legal CE additionally renormalizes the identical legal lists",
        forward_protocol="Compiled BF16 full-sequence forward with saved split/YaRN/window state; no cached-inference performance claim",
        input_tensor_protocol="Inputs constructed inside inference mode, matching the frozen training evaluator",
    )
    out = ROOT / "results/lm-eval" / run.name
    if final:
        common.update(
            final_selection_sha256=a.selection_sha256, evaluation_shards=shards
        )
        out = ROOT / "results/final-evaluation" / a.selection_sha256 / run.name
    out.mkdir(parents=True, exist_ok=True)

    forward_timer = ForwardTimer(enabled=a.split != "original_val")

    @torch.inference_mode()
    def forward(ids, scored_positions=0, clock=None, elo=None, feat=None):
        # Keep the last partial batch at the same compiled shape. BOS padding
        # forms isolated one-token documents and cannot affect real histories.
        count = len(ids)
        if count < batch:
            ids = torch.cat(
                (
                    ids,
                    torch.full(
                        (batch - count, ids.shape[1]),
                        2348,
                        dtype=ids.dtype,
                        device=ids.device,
                    ),
                )
            )
            if clock is not None:
                clock = torch.cat(
                    (clock, clock.new_zeros(batch - count, clock.shape[1]))
                )
            if elo is not None:
                elo = torch.cat((elo, elo.new_zeros(batch - count, elo.shape[1])))
            if feat is not None:
                feat = torch.cat(
                    (feat, feat.new_full((batch - count, *feat.shape[1:]), -1))
                )
        context = make_context(ids, schedule.ws_short * 128, schedule.ws_long * 128)
        extra = () if clock is None else (clock.flatten(),)
        kw = (
            {} if elo is None else dict(elo_seq=elo.flatten())
        )  # only Elo-input sources take it
        if feat is not None:
            kw["feat_seq"] = feat.flatten(0, 1)
        with forward_timer.measure(ids.shape, scored_positions):
            logits = net(ids.flatten(), ids.flatten(), context, schedule, *extra, **kw)
        return logits.reshape(*ids.shape, -1)[:count]

    if a.split == "strat":
        cached = Path("/data/group_data/dei-group/yimingz3/allie/strat-eval-v1")
        manifest = json.loads((cached / "manifest.json").read_text())
        with (cached / "strat.npz").open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == manifest["sha256"]
        with np.load(cached / "strat.npz") as z:
            srows = z["rows"].astype(np.int64)
            slabels = z["labels"][:, 1:].astype(np.int64)
        with (cached / "clocks.npz").open("rb") as f:
            assert (
                hashlib.file_digest(f, "sha256").hexdigest()
                == manifest["clocks_sha256"]
            )
        with np.load(cached / "clocks.npz") as z:
            tclock = z["clocks"][:, :-1].astype(np.int64)
        tt_all = (tclock >= 1) & (tclock <= 16)  # mover has at most 15 s left
        sclock = tclock if getattr(model, "use_clock", False) else None
        selo = elo_buckets(srows) if getattr(model, "use_elo", False) else None
        sfeat = None
        if getattr(model, "use_feats", False):  # continuous clock features
            with (cached / "feats.npz").open("rb") as f:
                assert (
                    hashlib.file_digest(f, "sha256").hexdigest()
                    == manifest["feats_sha256"]
                )
            with np.load(cached / "feats.npz") as z:
                sfeat = z["feats"][:, :-1].astype(np.int64)
        cells = manifest["cells"]
        k = len(cells)
        sums, tsums = np.zeros((k, 2)), np.zeros((k, 2))
        for lo in range(0, len(srows), batch):
            rows = srows[lo : lo + batch]
            with torch.inference_mode():
                x = torch.as_tensor(rows[:, :-1], device="cuda")
                y = torch.as_tensor(rows[:, 1:], device="cuda")
            clk = (
                None
                if sclock is None
                else torch.as_tensor(sclock[lo : lo + batch], device="cuda")
            )
            el = (
                None
                if selo is None
                else torch.as_tensor(selo[lo : lo + batch], device="cuda")
            )
            ft = (
                None
                if sfeat is None
                else torch.as_tensor(sfeat[lo : lo + batch], device="cuda")
            )
            scores = forward(x, clock=clk, elo=el, feat=ft)[..., 378:2346].float()
            nll = scores.logsumexp(-1) - scores.gather(
                -1, (y - 378).clamp(0, 1967)[..., None]
            ).squeeze(-1)
            lab = torch.as_tensor(slabels[lo : lo + batch], device="cuda")
            keep = lab >= 0
            sums[:, 0] += (
                torch.zeros(k, device="cuda", dtype=torch.float64)
                .index_add_(0, lab[keep], nll[keep].double())
                .cpu()
                .numpy()
            )
            sums[:, 1] += torch.bincount(lab[keep], minlength=k).cpu().numpy()
            tt = keep & torch.as_tensor(tt_all[lo : lo + batch], device="cuda")
            tsums[:, 0] += (
                torch.zeros(k, device="cuda", dtype=torch.float64)
                .index_add_(0, lab[tt], nll[tt].double())
                .cpu()
                .numpy()
            )
            tsums[:, 1] += torch.bincount(lab[tt], minlength=k).cpu().numpy()
        assert (
            np.isfinite(sums).all() and (sums[:, 1] == manifest["scored_moves"]).all()
        )
        ce = dict(zip(cells, (sums[:, 0] / sums[:, 1]).tolist()))
        group = lambda pred: float(np.mean([v for c, v in ce.items() if pred(c)]))
        report = common | dict(
            split="strat",
            strat_sha256=manifest["sha256"],
            cells=ce,
            counts=dict(zip(cells, sums[:, 1].astype(int).tolist())),
            macro=group(lambda c: True),
            expert_macro=group(lambda c: c.endswith(">=2400")),
            by_format={
                f: group(lambda c, f=f: c.startswith(f + "/"))
                for f in ("bullet", "blitz", "rapid", "classical")
            },
            by_band={
                b: group(lambda c, b=b: c.endswith("/" + b))
                for b in ("<1400", "1400-2000", "2000-2400", ">=2400")
            },
            time_trouble=dict(
                definition="scored moves whose mover has at most 15 s left",
                ce=float(tsums[:, 0].sum() / tsums[:, 1].sum()),
                moves=int(tsums[:, 1].sum()),
                by_format={
                    f: float(tsums[m, 0].sum() / tsums[m, 1].sum())
                    for f in ("bullet", "blitz", "rapid", "classical")
                    if tsums[(m := [c.startswith(f + "/") for c in cells]), 1].sum()
                },
            ),
        )
        report["sidecars"] = {
            k: manifest[k]
            for k in ("clocks_sha256", "feats_sha256")
            if k in manifest and (k == "clocks_sha256" or sfeat is not None)
        }
        tmp = out / "strat-v1.json.tmp"
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(
            out / "strat-v1.json"
        )  # atomic: a preempted write never looks complete
        print(
            json.dumps(
                dict(
                    phase="strat",
                    macro=report["macro"],
                    expert_macro=report["expert_macro"],
                )
            ),
            flush=True,
        )
        dist.destroy_process_group()
        return

    if a.split == "original_val":
        durable = Path("/data/group_data/dei-group/yimingz3/allie")
        cached = durable / "validation_cache"
        data_path = durable / "lichess_tokens_v2"
        if (cached / "manifest.json").exists():
            manifest = json.loads((cached / "manifest.json").read_text())
            assert manifest["revision"] == "20a899ddf344ccaea74e273509a60e5a511125f8"
            assert len(manifest["original_files"]) == 100 and manifest["rows"] == 5371
            with (cached / manifest["derived_file"]).open("rb") as f:
                assert (
                    hashlib.file_digest(f, "sha256").hexdigest() == manifest["sha256"]
                )
            data_path = cached
            common["validation_cache_sha256"] = manifest["sha256"]
        val = Packed(data_path, "val")
        indices = np.arange(int(val.ends[-1]))
        basename = "original-val"
        if a.match_training_validation:
            indices = np.array(
                json.loads((run / "config.json").read_text())["val_indices"]
            )
            basename = "original-val-selection-check"
        n = len(indices)
        print(
            json.dumps(dict(phase="evaluate_original_validation", rows=n, batch=batch)),
            flush=True,
        )
        totals = np.zeros((n, 6), np.float64)
        for lo in range(0, n, batch):
            rows = val.rows(indices[lo : min(n, lo + batch)])
            # Match the tensor kind used inside training_evaluate's decorator.
            # Entering inference mode only inside forward leaves ordinary input
            # tensors and can select a different compiled specialization.
            with torch.inference_mode():
                x = torch.as_tensor(rows[:, :-1], device="cuda")
                y = torch.as_tensor(rows[:, 1:], device="cuda")
            el = (
                torch.as_tensor(elo_buckets(rows), device="cuda")
                if getattr(model, "use_elo", False)
                else None
            )
            scores = forward(x, elo=el)[..., 378:2346].float()
            nll = scores.logsumexp(-1) - scores.gather(
                -1, (y - 378).clamp(0, 1967)[..., None]
            ).squeeze(-1)
            valid = (y >= 378) & (y < 2346)
            elo = torch.as_tensor(ratings(rows), device="cuda")
            for j, mask in enumerate(
                (valid, valid & (elo >= 2400), valid & (elo >= 2600))
            ):
                totals[lo : lo + len(rows), 2 * j] = (
                    (nll.double() * mask).sum(-1).cpu().numpy()
                )
                totals[lo : lo + len(rows), 2 * j + 1] = mask.sum(-1).cpu().numpy()
        assert np.isfinite(totals).all()
        np.savez(
            out / (basename + "-rows.npz"),
            row=indices,
            nll_sums=totals[:, ::2],
            counts=totals[:, 1::2],
        )
        sums = totals.sum(0)
        report = common | dict(
            rows=n, dataset_revision="20a899ddf344ccaea74e273509a60e5a511125f8"
        )
        for i, name in enumerate(("move", "expert2400", "expert2600")):
            report[name + "_ce"] = sums[2 * i] / max(1, sums[2 * i + 1])
            report[name + "_count"] = int(sums[2 * i + 1])
        if a.match_training_validation:
            from types import SimpleNamespace
            from modded_train import evaluate as training_evaluate

            manager = SimpleNamespace(
                ws_short=schedule.ws_short,
                ws_long=schedule.ws_long,
                get_forward_args=lambda: schedule,
            )
            direct = training_evaluate(net, manager, val.rows(indices), batch)
            diagnostic = dict(
                checkpoint=common,
                saved=state["metrics"],
                standalone={
                    k: report[k] for k in ("move_ce", "expert2400_ce", "expert2600_ce")
                },
                frozen_training_path=direct,
            )
            if a.diagnose_restore:
                print(json.dumps(dict(phase="restore_full_training_state")), flush=True)
                assert pointer["world_size"] == 1
                from modded_medium import TrainingManager
                from lm_checkpoint import restore_rng

                manager = TrainingManager(model, cfg)
                local = torch.load(
                    run / pointer["directory"] / "rank0.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                manager.load_rank_state_dict(local["manager"])
                restore_rng(local["rng"])
                core.grad_accum_steps = manager.batch_size // cfg.max_tokens
                assert all(
                    torch.equal(value.cpu(), state["model"][name])
                    for name, value in model.state_dict().items()
                )
                diagnostic["full_training_state_path"] = training_evaluate(
                    net, manager, val.rows(indices), batch
                )
                diagnostic["model_tensors_equal_checkpoint"] = True
                diagnostic["diagnostic_only"] = True
            (out / "validation-equivalence-diagnostics.json").write_text(
                json.dumps(diagnostic, indent=2) + "\n"
            )
            identifier = os.environ.get("SLURM_JOB_ID", "local")
            (out / f"validation-equivalence-{identifier}.json").write_text(
                json.dumps(diagnostic, indent=2) + "\n"
            )
            print(json.dumps(diagnostic), flush=True)
            if a.diagnose_evaluator_paths or a.audit_evaluator_paths:
                from diagnose_eval_paths import compare_paths

                path_diagnostic = compare_paths(
                    net,
                    forward,
                    make_context,
                    schedule,
                    ratings,
                    val.rows(indices),
                    batch,
                )
                path_diagnostic["checkpoint"] = common
                (out / f"evaluator-path-diagnostics-{identifier}.json").write_text(
                    json.dumps(path_diagnostic, indent=2) + "\n"
                )
                print(
                    json.dumps(
                        dict(
                            phase="evaluator_path_diagnostics",
                            ce=path_diagnostic["ce"],
                            counts=path_diagnostic["counts"],
                        )
                    ),
                    flush=True,
                )
                if a.diagnose_evaluator_paths:
                    dist.destroy_process_group()
                    return
            if a.diagnose_restore:
                dist.destroy_process_group()
                return
            keys = ("move_ce", "expert2400_ce", "expert2600_ce")
            # Both paths on the restored model must agree tightly. Separately,
            # a fresh compiled BF16 process can differ from training's recorded
            # metric: full-state checks found identical tensors and consistent
            # differences up to1.42e-4 nats. Keep that numerical allowance fixed
            # and visible, not mislabeled as bitwise or statistical uncertainty.
            tolerance = 1e-7 if state["args"].get("deterministic", False) else 5e-4
            deltas = {key: report[key] - state["metrics"][key] for key in keys}
            if any(abs(report[key] - direct[key]) >= 1e-7 for key in keys):
                # Retain the failing allocation long enough to diagnose it on
                # the same GPU/process; an external retry may land elsewhere.
                from diagnose_eval_paths import compare_paths

                failed_paths = compare_paths(
                    net,
                    forward,
                    make_context,
                    schedule,
                    ratings,
                    val.rows(indices),
                    batch,
                )
                failed_paths["checkpoint"] = common
                failed_paths["initial_standalone"] = diagnostic["standalone"]
                failed_paths["initial_frozen_training_path"] = direct
                (out / f"evaluator-path-failure-{identifier}.json").write_text(
                    json.dumps(failed_paths, indent=2) + "\n"
                )
                print(
                    json.dumps(
                        dict(phase="failed_path_diagnostics", ce=failed_paths["ce"])
                    ),
                    flush=True,
                )
            for key in keys:
                assert abs(report[key] - direct[key]) < 1e-7, ("evaluator_paths", key)
                assert abs(deltas[key]) < tolerance, (
                    "saved_validation",
                    key,
                    deltas[key],
                    tolerance,
                )
            for key in ("move_count", "expert2400_count", "expert2600_count"):
                assert report[key] == direct[key] == state["metrics"][key]
            report.update(
                matches_frozen_training_evaluator=True,
                matches_training_validation=all(abs(x) < 1e-7 for x in deltas.values()),
                matches_saved_validation_within_tolerance=True,
                saved_validation_tolerance=tolerance,
                saved_validation_deltas=deltas,
            )
        tmp = out / (basename + ".json.tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(out / (basename + ".json"))
    else:
        assert not a.match_training_validation
        out = out / a.split
        out.mkdir(exist_ok=True)
        all_rows = []
        for shard in range(shards):
            reference_dir = ROOT / "results" / a.split
            cache = ROOT / "data" / f"prepared-{a.split}-{shards}-{shard}"
            if final:
                reference_dir = (
                    ROOT
                    / "results/final-evaluation"
                    / a.selection_sha256
                    / "reference"
                    / a.split
                )
                cache = reference_dir / f"prepared-{shard}"
                from final_selection import digest

                receipt = json.loads((reference_dir / f"done-{shard}.json").read_text())
                assert receipt["final_selection_sha256"] == a.selection_sha256
                assert (
                    receipt["split"] == a.split
                    and receipt["shard"] == shard
                    and receipt["shards"] == shards
                )
                for name, sha in receipt["output_sha256"].items():
                    assert (
                        Path(name).name == name and digest(reference_dir / name) == sha
                    )
            with np.load(cache.with_suffix(".npz")) as prepared:
                data = {key: prepared[key] for key in ALIGNMENT_KEYS}
            with np.load(reference_dir / f"meta-{shard}.npz") as reference:
                alignment = verify_alignment(data, reference)
            prompts = json.loads(cache.with_suffix(".json").read_text())
            prompts.sort(key=lambda x: len(x[0]))
            n = len(data["raw_target"])
            raw = np.full(n, np.nan, np.float32)
            legal = np.full(n, np.nan, np.float32)
            for lo in range(0, len(prompts), batch):
                group = prompts[lo : lo + batch]
                # A few fixed shapes reduce compilation across varying games.
                length = ((max(len(x[0]) for x in group) + 127) // 128) * 128
                assert length <= 1024
                with torch.inference_mode():
                    ids = torch.full(
                        (len(group), length), 2348, dtype=torch.long, device="cuda"
                    )
                    for i, (tokens, _, _) in enumerate(group):
                        ids[i, : len(tokens)] = torch.as_tensor(tokens, device="cuda")
                logits = forward(
                    ids, scored_positions=sum(end - start for _, start, end in group)
                )
                for i, (_, start, end) in enumerate(group):
                    if start == end:
                        continue
                    positions = torch.as_tensor(
                        data["ply"][start:end].astype(np.int64) + 10, device="cuda"
                    )
                    scores = logits[i, positions].float()
                    targets = torch.as_tensor(
                        data["raw_target"][start:end].astype(np.int64), device="cuda"
                    )
                    truth = scores.gather(1, targets[:, None])[:, 0]
                    raw[start:end] = (
                        (scores[:, 378:2346].logsumexp(-1) - truth).cpu().numpy()
                    )
                    qids = torch.as_tensor(
                        data["qids"][start:end].astype(np.int64), device="cuda"
                    )
                    selected = scores.gather(1, qids.clamp_min(0)).masked_fill(
                        qids < 0, -torch.inf
                    )
                    legal[start:end] = (selected.logsumexp(-1) - truth).cpu().numpy()
            assert np.isfinite(raw).all() and np.isfinite(legal).all()
            np.savez(
                out / f"ce-{shard}.npz",
                raw_nll=raw,
                legal_nll=legal,
                game=data["game"],
                ply=data["ply"],
                elo=data["elos"][:, 0],
                alignment_sha256=alignment,
                model_sha256=model_sha,
                checkpoint_sha256=pointer_sha,
                final_selection_sha256=a.selection_sha256 if final else "",
            )
            all_rows.append((raw, legal, data["elos"][:, 0]))
        raw, legal, elo = (
            np.concatenate([row[i] for row in all_rows]) for i in range(3)
        )
        report = common | dict(
            split=a.split,
            positions=len(raw),
            move_ce=float(raw.mean()),
            legal_ce=float(legal.mean()),
            forward_timing=forward_timer.report(),
            inference_gpu=torch.cuda.get_device_name(),
        )
        assert report["forward_timing"]["all_forwards"]["scored_positions"] == len(raw)
        for threshold in (2400, 2600):
            mask = elo >= threshold
            report[f"expert{threshold}_positions"] = int(mask.sum())
            report[f"expert{threshold}_move_ce"] = (
                float(raw[mask].mean()) if mask.any() else None
            )
            report[f"expert{threshold}_legal_ce"] = (
                float(legal[mask].mean()) if mask.any() else None
            )
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
