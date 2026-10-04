"""Main evaluation (strat-eval-v1) or original-validation CE of a checkpoint, on one GPU.

strat         CE per format x mover-Elo cell (16), their macro, the >= 2400 cells' macro and
              time-trouble CE -> results/lm-eval/RUN/strat-v1.json
original_val  move / >= 2400 / >= 2600 CE over every original validation row
              -> results/lm-eval/RUN/original-val.json (+ per-row sums)

The model is rebuilt from this package, whose hashed files (train.provenance) must be the checkpoint's:
score a study's checkpoints with its frozen package. Checkpoints from before the package layout load from
their run's frozen flat source (--source), or from this package where it is known to score them as that
source does (EQUIVALENT). The final test splits are not exposed.
"""

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from allie import paths
from allie.train.provenance import source_hashes

REVISION = "20a899ddf344ccaea74e273509a60e5a511125f8"
# pre-package checkpoints this package scores as their frozen source does (main evaluation and MoE search
# bitwise on the same GPU; the Maia-3 benchmark scorer is not repeat-deterministic): digest of the recorded
# source_sha256 -> (name, digest of the tested package's source_hashes(), which any edit to a hashed file moves)
EQUIVALENT = {
    "3efce06d87e5e5fb22e2d6e79bbc52ab7386b5a3af9923a4eafcbbc42995b944": (
        "Allie 2.0",
        "e00649cbfd056c3b76b29289295907ef650d0c620e1c3e344e8e2065b6d986ca",
    ),
}


def digest(hashes):
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def read_hashed(path):
    with path.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
        f.seek(0)
        return torch.load(f, map_location="cpu", weights_only=False), sha


def tokens(model, x):
    """forward()'s token arguments: frozen sources from before this package also take an unused target."""
    return (x, x) if "target_seq" in inspect.signature(model.forward).parameters else (x,)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def training_code(state, source=None):
    """(packed, network, trainer): the modules that trained a checkpoint. Without source, this package's,
    whose hashed files must be the checkpoint's (or an EQUIVALENT pre-package checkpoint's); with source, a
    checkpoint from before the package layout, loaded from its run's frozen flat source, checked file by file."""
    recorded = state["source_sha256"]
    if source is None:
        if all("/" not in name for name in recorded):
            name, tested = EQUIVALENT.get(digest(recorded), (None, None))
            assert name, "a pre-package checkpoint: pass its run's frozen source (--source)"
            assert digest(source_hashes()) == tested, (
                f"{name} was checked against another version of this package: pass its frozen source (--source)"
            )
        else:
            assert recorded == source_hashes(), (
                "this package is not the code the checkpoint was trained with: run its study's frozen package"
            )
        from allie.data import packed
        from allie.model import network
        from allie.train import trainer

        return packed, network, trainer
    source = Path(source).resolve()
    for name, expected in recorded.items():
        assert (source / name).exists() and sha(source / name) == expected, (
            f"{source / name} is not the code this checkpoint was trained with: "
            "pass its run's frozen source"
        )
    sys.path.insert(0, str(source))
    import lm_data
    import modded_medium
    import modded_train

    for name, module in sys.modules.items():
        if name.startswith("modded_"):
            assert Path(module.__file__).resolve().parent == source, (name, module.__file__)
    return lm_data, modded_medium, modded_train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["original_val", "strat"], required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--source", help="a pre-package checkpoint's frozen flat source directory")
    a = p.parse_args()
    print(json.dumps(dict(phase="load_checkpoint", split=a.split)), flush=True)
    checkpoint = Path(a.checkpoint).resolve()
    pointer, pointer_sha = read_hashed(checkpoint)
    assert pointer["format"] == "allie-modded-medium-1"
    run = checkpoint.parent
    state, model_sha = read_hashed(run / pointer["directory"] / "model.pt")
    assert "inference" in state, "Checkpoint needs saved inference schedule/buffers"
    packed, network, trainer = training_code(state, a.source)
    Packed, ratings = packed.Packed, trainer.ratings
    Config, core, create_model = network.Config, network.core, network.create_model
    make_context = network.make_context

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.use_deterministic_algorithms(state["args"].get("deterministic", False))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 1, "one GPU, no distributed optimizer"
    cfg = Config(**state["config"])
    model = create_model(cfg)
    dropped = [k for k in ("use_clock", "use_elo") if getattr(model, k, False)]
    assert not dropped, (
        f"checkpoint uses the dropped {dropped} inputs: score it with its study's "
        "evaluator-ours copy"
    )
    # an N-GPU checkpoint pads the scalars; keep that shape for strict loading
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
    feats = getattr(model, "use_feats", False)
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
    out = paths.ROOT / "results/lm-eval" / run.name
    out.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def nll(rows, feat=None):
        """Next-move NLL over the 1968 move ids. rows become tensors inside inference mode, as in
        the training evaluator, feat outside it (the tensor kind picks the compiled graph); a short
        last batch is padded with BOS rows (isolated one-token games) to keep the compiled shape."""
        count = len(rows)
        x = torch.as_tensor(rows[:, :-1], device="cuda")
        y = torch.as_tensor(rows[:, 1:], device="cuda")
        if count < batch:
            x = torch.cat((x, x.new_full((batch - count, x.shape[1]), 2348)))
        kw = {}
        if feat is not None:
            if count < batch:
                pad = feat.new_full((batch - count, *feat.shape[1:]), -1)
                feat = torch.cat((feat, pad))
            kw["feat_seq"] = feat.flatten(0, 1)
        context = make_context(x, schedule.ws_short * 128, schedule.ws_long * 128)
        logits = net(*tokens(model, x.flatten()), context, schedule, **kw)
        scores = logits.reshape(*x.shape, -1)[:count, :, 378:2346].float()
        truth = scores.gather(-1, (y - 378).clamp(0, 1967)[..., None]).squeeze(-1)
        return scores.logsumexp(-1) - truth, y

    if a.split == "strat":
        cached = paths.DATA / "strat-eval-v1"
        manifest = json.loads((cached / "manifest.json").read_text())
        assert sha(cached / "strat.npz") == manifest["sha256"]
        with np.load(cached / "strat.npz") as z:
            srows = z["rows"].astype(np.int64)
            slabels = z["labels"][:, 1:].astype(np.int64)
        assert sha(cached / "clocks.npz") == manifest["clocks_sha256"]
        with np.load(cached / "clocks.npz") as z:
            tclock = z["clocks"][:, :-1].astype(np.int64)
        tt_all = (tclock >= 1) & (tclock <= 16)  # mover has at most 15 s left
        sfeat = None
        if feats:
            assert sha(cached / "feats.npz") == manifest["feats_sha256"]
            with np.load(cached / "feats.npz") as z:
                sfeat = z["feats"][:, :-1].astype(np.int64)
        cells = manifest["cells"]
        k = len(cells)
        sums, tsums = np.zeros((k, 2)), np.zeros((k, 2))

        def add(acc, lab, loss):
            acc[:, 0] += (
                torch.zeros(k, device="cuda", dtype=torch.float64)
                .index_add_(0, lab, loss.double())
                .cpu()
                .numpy()
            )
            acc[:, 1] += torch.bincount(lab, minlength=k).cpu().numpy()

        for lo in range(0, len(srows), batch):
            ft = torch.as_tensor(sfeat[lo : lo + batch], device="cuda") if feats else None
            loss, _ = nll(srows[lo : lo + batch], ft)
            lab = torch.as_tensor(slabels[lo : lo + batch], device="cuda")
            keep = lab >= 0
            add(sums, lab[keep], loss[keep])
            tt = keep & torch.as_tensor(tt_all[lo : lo + batch], device="cuda")
            add(tsums, lab[tt], loss[tt])
        assert (
            np.isfinite(sums).all() and (sums[:, 1] == manifest["scored_moves"]).all()
        )
        ce = dict(zip(cells, (sums[:, 0] / sums[:, 1]).tolist()))
        group = lambda pred: float(np.mean([v for c, v in ce.items() if pred(c)]))
        formats = ("bullet", "blitz", "rapid", "classical")
        report = common | dict(
            split="strat",
            strat_sha256=manifest["sha256"],
            cells=ce,
            counts=dict(zip(cells, sums[:, 1].astype(int).tolist())),
            macro=group(lambda c: True),
            expert_macro=group(lambda c: c.endswith(">=2400")),
            by_format={f: group(lambda c, f=f: c.startswith(f + "/")) for f in formats},
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
                    for f in formats
                    if tsums[(m := [c.startswith(f + "/") for c in cells]), 1].sum()
                },
            ),
        )
        report["sidecars"] = {"clocks_sha256": manifest["clocks_sha256"]}
        if feats:
            report["sidecars"]["feats_sha256"] = manifest["feats_sha256"]
        tmp = out / "strat-v1.json.tmp"
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(out / "strat-v1.json")  # a preempted write never looks complete
        summary = {k: report[k] for k in ("macro", "expert_macro")}
        print(json.dumps(dict(phase="strat") | summary), flush=True)
        dist.destroy_process_group()
        return

    cached, data = paths.DATA / "validation_cache", paths.DATA / "lichess_tokens_v2"
    if (cached / "manifest.json").exists():
        manifest = json.loads((cached / "manifest.json").read_text())
        assert manifest["revision"] == REVISION
        assert len(manifest["original_files"]) == 100 and manifest["rows"] == 5371
        assert sha(cached / manifest["derived_file"]) == manifest["sha256"]
        data = cached
        common["validation_cache_sha256"] = manifest["sha256"]
    val = Packed(data, "val")
    indices = np.arange(int(val.ends[-1]))
    n = len(indices)
    print(
        json.dumps(dict(phase="evaluate_original_validation", rows=n, batch=batch)),
        flush=True,
    )
    totals = np.zeros((n, 6), np.float64)
    for lo in range(0, n, batch):
        rows = val.rows(indices[lo : min(n, lo + batch)])
        loss, y = nll(rows)
        valid = (y >= 378) & (y < 2346)
        elo = torch.as_tensor(ratings(rows), device="cuda")
        for j, mask in enumerate((valid, valid & (elo >= 2400), valid & (elo >= 2600))):
            totals[lo : lo + len(rows), 2 * j] = (
                (loss.double() * mask).sum(-1).cpu().numpy()
            )
            totals[lo : lo + len(rows), 2 * j + 1] = mask.sum(-1).cpu().numpy()
    assert np.isfinite(totals).all()
    np.savez(
        out / "original-val-rows.npz",
        row=indices,
        nll_sums=totals[:, ::2],
        counts=totals[:, 1::2],
    )
    sums = totals.sum(0)
    report = common | dict(rows=n, dataset_revision=REVISION)
    for i, name in enumerate(("move", "expert2400", "expert2600")):
        report[name + "_ce"] = sums[2 * i] / max(1, sums[2 * i + 1])
        report[name + "_count"] = int(sums[2 * i + 1])
    tmp = out / "original-val.json.tmp"
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(out / "original-val.json")
    print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
