# Allie 2.0: the run's record

Verbatim copies of what ran, before the package layout (paths are our cluster's, file names the flat `scripts/` ones).
[`../allie-2.0.json`](../allie-2.0.json) is the same run as a config for `allie-train`.

| File | Contents |
|---|---|
| `plan.json` | The frozen study: shape, schedule, data policy and every trainer flag, with the sha256 of each frozen source file |
| `resume-config.json` | The trainer's record of its arguments and model config at its last resume, including the gate floor added at step 59,392 |
| `round.py` | The round file that declared the study |
| `run.sbatch` | The Slurm job that ran it in 2-day chunks on 8 × L40S |
| `data-pin.json`, `history-counts.json` | The frozen data selection (111 Lichess months plus the over-the-board and engine stores, each month's bucket file hashed) and the per-bucket game counts the sampler uses |
| `anneal.sbatch` | The launcher of the second anneal (Allie 2.0 annealed): `--init-from` the final checkpoint, 1B tokens of recent months, peak learning rate 0.05 |

The sampling table is [`../recipes/c8s200f0v4-fcfbf8858a28.json`](../recipes/c8s200f0v4-fcfbf8858a28.json).
