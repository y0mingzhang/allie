"""Legal-policy CE and top-1 accuracy of an MoE checkpoint on the sampled blitz positions.

Same model loading, source-hash check and full-sequence batched forward as the golden
evaluator (scripts/eval_strat.py, imported for its hashing helpers and copied for the
restore sequence); this adds the legal-move renormalization and argmax that the CE-only
evaluator does not produce. It also re-accumulates the canonical per-cell CE over every
scored blitz move so the run can be checked against the evaluator's strat-v1.json.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, "/home/yimingz3/src/allie/scripts")
from eval_strat import read_hashed, sha  # noqa: E402

G = Path("/data/group_data/dei-group/yimingz3/allie/strat-eval-v1")
DATA = Path("/data/group_data/dei-group/yimingz3/allie/maia3-bench")
CELLS = (4, 5, 6, 7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=str(DATA), help="directory of legal.npz")
    a = p.parse_args()

    checkpoint = Path(a.checkpoint).resolve()
    if checkpoint.is_dir():  # a published checkpoint directory, no pointer
        model_path, pointer_sha = checkpoint / "model.pt", None
    else:
        pointer, pointer_sha = read_hashed(checkpoint)
        assert pointer["format"] == "allie-modded-medium-1"
        model_path = checkpoint.parent / pointer["directory"] / "model.pt"
    state, model_sha = read_hashed(model_path)
    source = Path(a.source).resolve()
    for name, expected in state["source_sha256"].items():
        assert (source / name).exists() and sha(source / name) == expected, name
    sys.path.insert(0, str(source))
    from modded_medium import Config, core, create_model, make_context

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    cfg = Config(**state["config"])
    model = create_model(cfg)
    assert not any(getattr(model, k, False) for k in ("use_clock", "use_elo"))
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
    net = torch.compile(model, dynamic=False, fullgraph=True)
    schedule = core.ForwardScheduleConfig(
        None, inference["ws_short"], inference["ws_long"]
    )
    batch = min(a.batch, cfg.max_tokens // 1024)
    feats_on = getattr(model, "use_feats", False)

    manifest = json.loads((G / "manifest.json").read_text())
    with np.load(G / "strat.npz") as z:
        srows, slabels = z["rows"].astype(np.int64), z["labels"][:, 1:].astype(np.int64)
    sfeat = None
    if feats_on:
        with np.load(G / "feats.npz") as z:
            sfeat = z["feats"][:, :-1].astype(np.int64)
    with np.load(Path(a.data) / "legal.npz") as z:
        pos, target, legal, offsets = z["pos"], z["target"], z["legal"], z["offsets"]
    width = int(np.diff(offsets).max())
    pad = np.zeros((len(pos), width), np.int64)
    ok = np.zeros((len(pos), width), bool)
    for i, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        pad[i, : hi - lo] = legal[lo:hi]
        ok[i, : hi - lo] = True
    pad_t = torch.as_tensor(pad, device="cuda")
    ok_t = torch.as_tensor(ok, device="cuda")
    tgt_t = torch.as_tensor(target, device="cuda")

    n = len(pos)
    ce = torch.zeros(n, dtype=torch.float64, device="cuda")
    ce_full = torch.zeros(n, dtype=torch.float64, device="cuda")
    top1 = torch.zeros(n, dtype=torch.bool, device="cuda")
    sums = np.zeros((16, 2))
    t0 = time.perf_counter()

    @torch.inference_mode()
    def run_batch(lo):
        hi = min(lo + batch, len(srows))
        rows = srows[lo:hi]
        x = torch.as_tensor(rows[:, :-1], device="cuda")
        y = torch.as_tensor(rows[:, 1:], device="cuda")
        count = hi - lo
        if count < batch:
            x = torch.cat((x, x.new_full((batch - count, x.shape[1]), 2348)))
        kw = {}
        if feats_on:
            f = torch.as_tensor(sfeat[lo:hi], device="cuda")
            if count < batch:
                f = torch.cat((f, f.new_full((batch - count, *f.shape[1:]), -1)))
            kw["feat_seq"] = f.flatten(0, 1)
        context = make_context(x, schedule.ws_short * 128, schedule.ws_long * 128)
        logits = net(x.flatten(), x.flatten(), context, schedule, **kw)
        scores = logits.reshape(*x.shape, -1)[:count, :, 378:2346].float()
        truth = scores.gather(-1, (y - 378).clamp(0, 1967)[..., None]).squeeze(-1)
        loss = scores.logsumexp(-1) - truth
        lab = torch.as_tensor(slabels[lo:hi], device="cuda")
        keep = lab >= 0
        acc = torch.zeros(16, device="cuda", dtype=torch.float64)
        acc.index_add_(0, lab[keep], loss[keep].double())
        sums[:, 0] += acc.cpu().numpy()
        sums[:, 1] += torch.bincount(lab[keep], minlength=16).cpu().numpy()
        sl = np.flatnonzero((pos[:, 0] >= lo) & (pos[:, 0] < hi))
        if len(sl):
            i = torch.as_tensor(sl, device="cuda")
            z = scores[
                torch.as_tensor(pos[sl, 0] - lo, device="cuda"),
                torch.as_tensor(pos[sl, 1], device="cuda"),
            ]
            lg = z.gather(1, pad_t[i]).masked_fill(~ok_t[i], float("-inf"))
            t = z.gather(1, tgt_t[i][:, None]).squeeze(1)
            ce[i] = (lg.logsumexp(-1) - t).double()
            ce_full[i] = (z.logsumexp(-1) - t).double()
            top1[i] = (
                pad_t[i].gather(1, lg.argmax(-1, keepdim=True)).squeeze(1) == tgt_t[i]
            )

    for lo in range(0, len(srows), batch):
        run_batch(lo)
    dt = time.perf_counter() - t0
    assert (sums[:, 1] == manifest["scored_moves"]).all()
    cells = dict(zip(manifest["cells"], (sums[:, 0] / sums[:, 1]).tolist()))
    out = Path(a.out)
    np.savez(
        out,
        ce_legal=ce.cpu().numpy(),
        ce=ce_full.cpu().numpy(),
        top1=top1.cpu().numpy(),
    )
    info = dict(
        checkpoint=str(checkpoint),
        checkpoint_sha256=pointer_sha,
        model_sha256=model_sha,
        parameters=sum(p.numel() for p in model.parameters()),
        step=state["step"],
        tokens=state.get("tokens"),
        useful_training_flops=state.get("useful_training_flops"),
        seconds=dt,
        batch=batch,
        canonical_cells=cells,
        positions=n,
        device=torch.cuda.get_device_name(0),
    )
    out.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    print(
        json.dumps({k: info[k] for k in ("parameters", "seconds", "canonical_cells")}),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
