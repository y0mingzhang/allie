"""Dry-run manifest of heavy files to drop from old runs and studies. Deletes nothing.

Kept: bigrun-v1, dmix*, iso-* runs; runs touched in the last 48 h; runs named by the original-Qwen
and reference-model identities; the isoflop-v1, bigrun-v1, data-v1-*, strat-eval-v1 studies; the
qwen-reproduction runtime dependencies. Other runs drop checkpoints/ and *.pt; other studies drop
files over 100 MB. Small logs, metrics and reports stay. Writes results/prune-plan.json.
"""

import json
import re
import time
from pathlib import Path

ROOT = Path("/home/yimingz3/src/allie/results")
KEEP_RUNS = re.compile(r"^(bigrun-v1.*|dmix.*|iso-.*)$")
KEEP_STUDIES = re.compile(r"^(isoflop-v1|bigrun-v1|data-v1-.*|strat-eval-v1)$")
PROTECTED = (
    "recipe10x/qwen-reproduction/source-v1",
    "recipe10x/qwen-reproduction/qwen-norm-provenance/upstream-qk-norms.safetensors",
    "recipe10x/qwen-reproduction/qwen-wandb-audit",
)
IDENTITIES = (
    "../analysis/original_qwen.json",
    "final-training/reference-model-identity.json",
)
BIG = 100 << 20


def referenced():
    names = set()
    for f in IDENTITIES:
        p = ROOT / f
        if p.exists():
            names |= set(re.findall(r"pretrain/([A-Za-z0-9_.\-]+)", p.read_text()))
    return names


def size(p):
    return (
        sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        if p.is_dir()
        else p.stat().st_size
    )


def main():
    refs, recent, plan = referenced(), time.time() - 48 * 3600, []
    for run in sorted((ROOT / "pretrain").iterdir()):
        if not run.is_dir() or KEEP_RUNS.match(run.name) or run.name in refs:
            continue
        if run.stat().st_mtime > recent or any(
            f.stat().st_mtime > recent for f in run.glob("*.json*")
        ):
            continue
        heavy = [run / "checkpoints", *run.glob("*.pt")]
        items = [(str(p.relative_to(ROOT)), size(p)) for p in heavy if p.exists()]
        if items:
            plan.append(dict(owner=f"pretrain/{run.name}", items=items))
    for study in sorted((ROOT / "recipe10x").iterdir()):
        if not study.is_dir() or KEEP_STUDIES.match(study.name):
            continue
        items = [
            (str(f.relative_to(ROOT)), f.stat().st_size)
            for f in study.rglob("*")
            if f.is_file()
            and f.stat().st_size > BIG
            and not any(str(f.relative_to(ROOT)).startswith(p) for p in PROTECTED)
        ]
        if items:
            plan.append(dict(owner=f"recipe10x/{study.name}", items=items))
    total = sum(b for e in plan for _, b in e["items"])
    (ROOT / "prune-plan.json").write_text(
        json.dumps(
            dict(
                created=time.time(),
                kept_references=sorted(refs),
                total_bytes=total,
                plan=plan,
            ),
            indent=1,
        )
        + "\n"
    )
    for e in sorted(plan, key=lambda e: -sum(b for _, b in e["items"]))[:25]:
        print(f"{sum(b for _, b in e['items']) / 2**30:8.1f} GB  {e['owner']}")
    print(
        f"{len(plan)} owners, {total / 2**40:.2f} TB reclaimable; kept by identity: {sorted(refs)}"
    )


if __name__ == "__main__":
    main()
