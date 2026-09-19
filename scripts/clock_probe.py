"""Golden-eval move CE of a clock model with its real clocks vs the clock channel zeroed.

The gap is what the model gets from the clock, reported by the mover's time left, by cell and by
format x time left. Writes results/lm-eval/<run>/clock-probe.json.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path("/home/yimingz3/src/allie")
CACHED = Path("/data/group_data/dei-group/yimingz3/allie/strat-eval-v1")
GROUPS = {
    "0-5s": (1, 6),
    "6-10s": (7, 11),
    "11-15s": (12, 16),
    "16s-1m": (17, 26),
    "1-3m": (27, 34),
    "3-10m": (35, 43),
    ">10m": (44, 63),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source", required=True, help="the run's frozen source directory")
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()
    ckpt = Path(a.checkpoint).resolve()
    pointer = torch.load(ckpt, weights_only=False, map_location="cpu")
    state = torch.load(
        ckpt.parent / pointer["directory"] / "model.pt",
        weights_only=False,
        map_location="cpu",
    )
    source = Path(a.source).resolve()
    for name, h in state["source_sha256"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == h, name
    sys.path.insert(0, str(source))
    from modded_medium import Config, core, create_model, make_context

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    cfg = Config(**state["config"])
    model = create_model(cfg)
    model.scalars.data = state["model"]["scalars"].to(
        device="cuda", dtype=model.scalars.dtype
    )
    model.load_state_dict(state["model"])
    inf = state["inference"]
    model.split_embed = inf["split_embed"]
    for k in ("angular_freq", "cos", "sin"):
        getattr(model.yarn, k).copy_(inf["yarn"][k].to("cuda"))
    model.yarn.attn_scale = inf["yarn"]["attn_scale"]
    model.eval()
    assert model.use_clock, "not a clock model"
    assert not (model.use_elo or model.use_feats), (
        "the probe feeds only the clock channel"
    )
    net = torch.compile(model, dynamic=False, fullgraph=True)
    sched = core.ForwardScheduleConfig(None, inf["ws_short"], inf["ws_long"])
    batch = min(a.batch, cfg.max_tokens // 1024)

    manifest = json.loads((CACHED / "manifest.json").read_text())
    with (CACHED / "clocks.npz").open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == manifest["clocks_sha256"]
    with np.load(CACHED / "strat.npz") as z:
        rows, labels = z["rows"].astype(np.int64), z["labels"][:, 1:].astype(np.int64)
    with np.load(CACHED / "clocks.npz") as z:
        clocks = z["clocks"][:, :-1].astype(np.int64)
    cells = manifest["cells"]

    def nll(x, y, clk):
        n = len(x)
        if n < batch:  # same compiled shape; BOS rows are isolated one-token documents
            x = torch.cat(
                (
                    x,
                    torch.full(
                        (batch - n, x.shape[1]), 2348, dtype=x.dtype, device=x.device
                    ),
                )
            )
            clk = torch.cat((clk, clk.new_zeros(batch - n, clk.shape[1])))
        ctx = make_context(x, sched.ws_short * 128, sched.ws_long * 128)
        s = net(x.flatten(), x.flatten(), ctx, sched, clk.flatten()).reshape(
            *x.shape, -1
        )
        s = s[:n, :, 378:2346].float()
        return s.logsumexp(-1) - s.gather(
            -1, (y - 378).clamp(0, 1967)[..., None]
        ).squeeze(-1)

    parts = []
    with torch.inference_mode():
        for lo in range(0, len(rows), batch):
            r, lab, c = (
                rows[lo : lo + batch],
                labels[lo : lo + batch],
                clocks[lo : lo + batch],
            )
            x, y = (
                torch.as_tensor(r[:, :-1], device="cuda"),
                torch.as_tensor(r[:, 1:], device="cuda"),
            )
            clk = torch.as_tensor(c, device="cuda")
            keep = lab >= 0
            with_clock, zeroed = (
                nll(x, y, clk).cpu().numpy(),
                nll(x, y, torch.zeros_like(clk)).cpu().numpy(),
            )
            parts.append((lab[keep], c[keep], with_clock[keep], zeroed[keep]))
    L, B, W, Z = (np.concatenate(x) for x in zip(*parts))
    assert (np.bincount(L, minlength=len(cells)) == manifest["scored_moves"]).all()

    def ce(m):
        return dict(
            moves=int(m.sum()),
            clock=float(W[m].mean()),
            zeroed=float(Z[m].mean()),
            gain=float(Z[m].mean() - W[m].mean()),
        )

    time = lambda lo, hi: (B >= lo) & (B <= hi)
    by_cell = {c: ce(L == i) for i, c in enumerate(cells)}
    fmt = lambda f: np.isin(
        L, [i for i, c in enumerate(cells) if c.startswith(f + "/")]
    )
    report = dict(
        checkpoint=str(ckpt),
        macro=dict(
            clock=float(np.mean([v["clock"] for v in by_cell.values()])),
            zeroed=float(np.mean([v["zeroed"] for v in by_cell.values()])),
        ),
        overall=ce(np.ones_like(L, bool)),
        no_clock_positions=ce(B == 0),
        by_time={g: ce(time(*r)) for g, r in GROUPS.items()},
        by_cell=by_cell,
        by_format_time={
            f: {g: ce(fmt(f) & time(*r)) for g, r in GROUPS.items()}
            for f in ("bullet", "blitz", "rapid", "classical")
        },
    )
    out = ROOT / "results/lm-eval" / ckpt.parent.name / "clock-probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(
        f"PROBE {ckpt.parent.name}: macro clock {report['macro']['clock']:.4f} zeroed {report['macro']['zeroed']:.4f}"
    )
    for g, v in report["by_time"].items():
        print(
            f"PROBE time {g:7} moves {v['moves']:8d} gain {v['gain']:+.4f} (zeroed {v['zeroed']:.4f})"
        )
    for f, d in report["by_format_time"].items():
        print(
            f"PROBE {f:9} "
            + " ".join(f"{g}:{v['gain']:+.3f}" for g, v in d.items() if v["moves"])
        )


if __name__ == "__main__":
    main()
