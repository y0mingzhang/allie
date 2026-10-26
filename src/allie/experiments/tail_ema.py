"""Golden CE of a run's final weights blended toward its tail EMAs (trainer --tail-ema), and the release blend.

  python -m allie.experiments.tail_ema STUDY RUN [GAMMA ...]   (default gammas 0.5 1)
  python -m allie.experiments.tail_ema blend CHECKPOINT TAIL GAMMA OUT

For each uninterrupted tail-ema-H.pt of results/pretrain/RUN and gamma: (1 - gamma) * final + gamma * EMA on the
parameters (buffers stay final), written as a run directory in node-local scratch and scored with the
study's frozen evaluator and source; scores land in results/lm-eval/RUN-temaHgG/. Needs one GPU.

blend writes one such blend to OUT (last.pt -> checkpoints/blend/model.pt, readable by the scorers and allie-bot
export) from CHECKPOINT (a last.pt pointer) and TAIL (its tail-ema-H.pt), with a readout record (both files'
sha256, horizon, gamma) in model.pt that export carries into config.json. It refuses an EMA of another step or one
restarted inside its window.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from allie.paths import ROOT


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def mix(state, tail, gamma):
    model = dict(state["model"])
    for k, e in tail["ema"].items():
        w = model[k]
        assert w.shape == e.shape, k
        model[k] = torch.lerp(w.float(), e.float(), gamma).to(w.dtype)
    return model


def save(pointer, state, model, out):
    (out / "checkpoints/blend").mkdir(parents=True)
    torch.save(state | dict(model=model), out / "checkpoints/blend/model.pt")
    torch.save(pointer | dict(directory="checkpoints/blend"), out / "last.pt")


def blend(checkpoint, tail_path, gamma, out):
    checkpoint, tail_path, out = Path(checkpoint), Path(tail_path), Path(out)
    pointer = torch.load(checkpoint, weights_only=False)
    final = checkpoint.parent / pointer["directory"] / "model.pt"
    state = torch.load(final, map_location="cpu", mmap=True, weights_only=False)
    tail = torch.load(tail_path, map_location="cpu", mmap=True, weights_only=False)
    assert tail["step"] == state["step"] == pointer["step"], "an EMA of another step"
    assert tail["ema_from"] == tail["window_from"], f"EMA restarted at {tail['ema_from']}, not {tail['window_from']}"
    readout = dict(
        final_step=state["step"],
        final_sha256=sha256(final),
        tail=tail_path.name,
        tail_sha256=sha256(tail_path),
        horizon=tail["horizon"],
        every=tail["every"],
        ema_from=tail["ema_from"],
        gamma=gamma,
    )
    save(pointer, state | dict(readout=readout), mix(state, tail, gamma), out)
    return readout


def main():
    if sys.argv[1] == "blend":
        checkpoint, tail, gamma, out = sys.argv[2:]
        print(json.dumps(blend(checkpoint, tail, float(gamma), out), indent=1))
        return
    study, run, *gammas = sys.argv[1:]
    gammas = [float(g) for g in gammas] or [0.5, 1.0]
    study = ROOT / "results/recipe10x" / study  # or an absolute path
    run = ROOT / "results/pretrain" / run
    pointer = torch.load(run / "last.pt", weights_only=False)
    state = torch.load(run / pointer["directory"] / "model.pt", map_location="cpu", weights_only=False)
    scratch = Path(tempfile.mkdtemp())
    out = {}
    try:
        for path in sorted(run.glob("tail-ema-*.pt")):
            tail = torch.load(path, map_location="cpu", weights_only=False)
            assert tail["step"] == pointer["step"], "an EMA of another step"
            if tail["ema_from"] != tail["window_from"]:  # a leg resumed inside the window restarted it
                print(json.dumps(dict(skipped=path.name, ema_from=tail["ema_from"])), flush=True)
                continue
            for g in gammas:
                name = f"{run.name}-tema{tail['horizon']:g}g{g:g}"
                d = scratch / name
                save(pointer, state, mix(state, tail, g), d)
                cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1"]
                env = os.environ | {"ALLIE_PROJECT_ROOT": str(ROOT)}  # scores land in ROOT/results, read below
                if (study / "source/allie").exists():  # a package study: its frozen package
                    cmd += ["-m", "allie.eval.score", "--checkpoint", d / "last.pt"]
                    env["PYTHONPATH"] = str(study / "source")
                else:  # a flat study: its frozen evaluator on its frozen source
                    cmd += [study / "evaluator-ours/eval_strat.py", "--checkpoint", d / "last.pt"]
                    cmd += ["--source", study / "source-ours"]
                cmd += ["--split", "strat", "--batch", 16]
                subprocess.run([str(c) for c in cmd], check=True, env=env)
                shutil.rmtree(d)
                s = json.loads((ROOT / "results/lm-eval" / name / "strat-v1.json").read_text())
                out[name] = dict(macro=s["macro"], expert_macro=s["expert_macro"])
                print(json.dumps({name: out[name]}), flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
