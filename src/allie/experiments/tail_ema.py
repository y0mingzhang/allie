"""Golden CE of a run's final weights blended toward its tail EMAs (trainer --tail-ema).

  python -m allie.experiments.tail_ema STUDY RUN [GAMMA ...]   (default gammas 0.5 1)

For each uninterrupted tail-ema-H.pt of results/pretrain/RUN and gamma: (1 - gamma) * final + gamma * EMA on the
parameters (buffers stay final), written as a run directory in node-local scratch and scored with the
study's frozen evaluator and source; scores land in results/lm-eval/RUN-temaHgG/. Needs one GPU.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(os.environ.get("ALLIE_PROJECT_ROOT", "/home/yimingz3/src/allie"))


def main():
    study, run, *gammas = sys.argv[1:]
    gammas = [float(g) for g in gammas] or [0.5, 1.0]
    study = ROOT / "results/recipe10x" / study  # or an absolute path
    run = ROOT / "results/pretrain" / run
    pointer = torch.load(run / "last.pt", weights_only=False)
    state = torch.load(run / pointer["directory"] / "model.pt", map_location="cpu", weights_only=False)
    scratch = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/scratch/yimingz3")))
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
                model = dict(state["model"])
                for k, e in tail["ema"].items():
                    w = model[k]
                    model[k] = torch.lerp(w.float(), e.float(), g).to(w.dtype)
                d = scratch / name
                (d / "checkpoints/blend").mkdir(parents=True)
                torch.save(state | dict(model=model), d / "checkpoints/blend/model.pt")
                torch.save(pointer | dict(directory="checkpoints/blend"), d / "last.pt")
                cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1"]
                env = None
                if (study / "source/allie").exists():  # a package study: its frozen package
                    cmd += ["-m", "allie.eval.score", "--checkpoint", d / "last.pt"]
                    env = os.environ | {"PYTHONPATH": str(study / "source")}
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
