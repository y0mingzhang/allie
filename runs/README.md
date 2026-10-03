# Run configurations

**bigrun/** is the frozen study of the released model (MoE 0.69B active / 5.6B total, 75B tokens), as `scripts/modelexp.py plan` wrote it:

| File | Contents |
|---|---|
| `plan.json` | The run's full specification: shape, schedule, data policy and every trainer flag, with the sha256 of each frozen source file |
| `resume-config.json` | The trainer's own record of its arguments and model config at the last resume, including the gate floor added at step 59,392 |
| `round.py` | The round file that declared the study (`bigfix()` waves at the end) |
| `run.sbatch` | The Slurm job that ran it in 2-day chunks on 8 × L40S |
| `recipes/c8s200f0v4-fcfbf8858a28.json` | The Elo-ramp sampling table: expected passes per (format, rating) bucket |
| `data-pin.json`, `history-counts.json` | The frozen data selection (111 Lichess months plus the over-the-board and engine stores, each month's bucket file hashed) and the per-bucket game counts the sampler uses |

**distill-v2/** holds the launchers for everything trained from the big run's weights: `ann.sbatch` runs the second anneal on recent months (January 2024 to June 2026 plus August 2026) and the fewer-experts fine-tunes (arch `moe_keep`, distilled toward the 16-expert model with `--kd-teacher`); `kd.sbatch` trains the distilled students and their hard-label controls.
