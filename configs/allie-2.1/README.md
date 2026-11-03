# Allie 2.1: the run's record

Copies of what ran, before the public package layout (paths are our cluster's, file names the flat `scripts/` ones). The only edits are names: this run was called Allie 2.2 while it trained, and the name Allie 2.1 then meant a v3 run that was paused.
[`../allie-2.1.json`](../allie-2.1.json) is the same run as a config for `allie-train`.

| File | Contents |
|---|---|
| `plan.json` | The frozen study: shape, schedule, data policy and every trainer flag, with the sha256 of each frozen source file |
| `resume-config.json` | The trainer's record of its arguments and model config at its last resume |
| `round.py` | The round file that declared the study |
| `run.sbatch` | The Slurm job that ran it in 2-day chunks on 8 × L40S |
| `data-pin.json`, `history-counts.json` | The frozen data selection and the per-bucket game counts the sampler reads |
| `blend.sbatch` | The launcher of the release blend: half the final weights, half the tail EMA |

The sampling table is [`../recipes/c8s200f0v4d112-b8276fd781ac.json`](../recipes/c8s200f0v4d112-b8276fd781ac.json).
