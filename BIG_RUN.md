# Training and compute state

No independent training or evaluation job is active. The controller/tunnel remains running (read `controller/job.id` for its current job ID). No final model is selected.

## Cumulative compute

GPU-hours remain separate by phase and hardware. These are cumulative charges, not fresh allowances after restart. Refresh with `python scripts/budget.py ledger` before planning any new run.

| Phase | Used GPU-hours | Cap |
|---|---:|---:|
| preliminary | 23.870556 | 24.0 |
| selection | 72.241667 | 96.0 |
| big | 47.822222 | 384.0 |
| science_a6000 | 113.242778 | 135.0 |
| scaling_wsd | 438.942778 | 455.0 |
| hardware_h200 | 0.000000 | 2.0 |

The final pool is 384 L40S GPU-hours total; science is accounted separately. `results/jobs.json` is the job registry and `results/compute.json` is the ledger. Do not reset either.

## Completed studies

- `results/recipe10x/tiny-scaling-v1`: 24 initial runs; depth/width ladder and four independent token horizons.
- `results/recipe10x/tiny-size-validation-v1`: six prospective intermediate-size checks.
- `results/recipe10x/tiny-seed-check-v1`: six additional seed runs; canonical review passed. Cost 1.961944 A6000 GPU-hours including evaluation.
- Final scaling fit: 42 endpoints including 12 historical large-model anchors. Seed replicas and the original big Qwen checkpoint are outside this fit.

Current coefficients, observations and calculations are checked into analysis/. Raw validation reports and full optimizer/RNG/data checkpoints remain in durable results storage. Exact historical jobs must use their original frozen source packages and runtime; moving code into Git does not authorize a checkpoint migration.

The most recent discussion proposed expert data-mixture experiments. No new mixture or final training run has been launched.
