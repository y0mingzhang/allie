"""Stratified July CE (strat-eval-v1) for scaled native Qwen exports.

Mirrors isoflop-v1 source-qwen/eval_scaled_native.py: same identity and hash checks, BF16 SDPA causal
forward over whole rows, move CE over the 1968 move ids. Writes <out>/strat-v1.json with the same
fields as eval_strat.py.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

STRAT = Path("/data/group_data/dei-group/yimingz3/allie/strat-eval-v1")


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--identity", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    started = time.monotonic()
    identity = json.loads(Path(a.identity).read_text())
    assert identity["format"] == "scaled-native-export-v1"
    assert identity["all_tensor_payloads_exact"]
    model_path = Path(identity["path"])
    for name, expected in identity["files_sha256"].items():
        assert digest(model_path / name) == expected, name
    manifest = json.loads((STRAT / "manifest.json").read_text())
    assert digest(STRAT / "strat.npz") == manifest["sha256"]
    with np.load(STRAT / "strat.npz") as z:
        rows, labels = z["rows"].astype(np.int64), z["labels"][:, 1:].astype(np.int64)
    cells = manifest["cells"]
    k = len(cells)
    torch.set_num_threads(4)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    assert sum(x.numel() for x in model.parameters()) == identity["parameters"]
    sums = np.zeros((k, 2))
    with torch.inference_mode():
        for lo in range(0, len(rows), a.batch):
            data = torch.as_tensor(rows[lo : lo + a.batch], device="cuda")
            x, y = data[:, :-1], data[:, 1:]
            scores = model(x, use_cache=False).logits[..., 378:2346].float()
            nll = scores.logsumexp(-1) - scores.gather(
                -1, (y - 378).clamp(0, 1967)[..., None]
            ).squeeze(-1)
            lab = torch.as_tensor(labels[lo : lo + a.batch], device="cuda")
            keep = lab >= 0
            sums[:, 0] += (
                torch.zeros(k, device="cuda", dtype=torch.float64)
                .index_add_(0, lab[keep], nll[keep].double())
                .cpu()
                .numpy()
            )
            sums[:, 1] += torch.bincount(lab[keep], minlength=k).cpu().numpy()
    assert np.isfinite(sums).all() and (sums[:, 1] == manifest["scored_moves"]).all()
    ce = dict(zip(cells, (sums[:, 0] / sums[:, 1]).tolist(), strict=True))
    group = lambda pred: float(np.mean([v for c, v in ce.items() if pred(c)]))
    report = dict(
        split="strat",
        identity=a.identity,
        model_sha256=identity["files_sha256"],
        parameters=identity["parameters"],
        strat_sha256=manifest["sha256"],
        forward_protocol="BF16 Qwen SDPA causal full strat rows, use_cache=False",
        evaluator_sha256=digest(__file__),
        seconds=time.monotonic() - started,
        cells=ce,
        counts=dict(zip(cells, sums[:, 1].astype(int).tolist(), strict=True)),
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
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "strat-v1.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                phase="strat",
                macro=report["macro"],
                expert_macro=report["expert_macro"],
            )
        )
    )


if __name__ == "__main__":
    main()
