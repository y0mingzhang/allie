# Research goal

Finish Qwen versus our architecture scaling laws: obtain a high-fidelity L(N,D) estimate for both recipes.

The thread goal is currently blocked, not complete. All scaling and seed-check runs finished. The simple additive law has systematic residuals; its forecasts and limitations are in README.md and analysis/scaling_fit.json. The user has since discussed expert data mixing, then requested repository cleanup. No new training has been requested in this cleanup task.

## Constraints

- Reuse original lichess_tokens_v2 corpus, revision 20a899ddf344ccaea74e273509a60e5a511125f8, splits and both Elo headers.
- Fit the requested additive law E+A/N^alpha+B/D^beta. Do not silently replace it or treat its conditional forecasts as measured gains.
- The current comparison is move-only, with no board inputs, distillation or search. Expert mixture changes are a proposed experiment, not a completed result.
- Keep final tests unopened. Preserve raw reports, checkpoints, source identities and all compute charges. Exception (user-approved 2026-09-18): weights of old runs/studies not planned for re-evaluation were pruned per results/prune-plan.json; each owner keeps a PRUNED.json tombstone plus its logs, metrics and reports.
- Golden eval (user, 2026-09-18): strat-eval-v1 macro CE — unweighted mean over 16 cells {bullet, blitz, rapid, classical} × mover Elo {<1400, 1400–2000, 2000–2400, ≥2400}, ~100K held-out July 2026 human moves each (dev/test excluded); expert macro = mean of the four ≥2400 cells. Original-val CE is kept for continuity only. Tools: scripts/eval_strat.py, scripts/eval_qwen_strat.py, scripts/strat_eval_all.py.
- General QoS: 8 GPUs, best chips available (usually L40S). DEI QoS: 8 A6000 always; up to 16 when no other dei-group jobs compete.
- Durable storage: /data/group_data/dei-group/yimingz3/allie. Scratch is a cache.
- Read controller/STOP before autonomous work. It stops controller activity, not independent training. Never stop the current tunnel/controller job.
- No duplicate submissions. All previous tiny-grid, size-validation and seed observers have completed; do not restart them.
- Generate images only when requested.

IsoFLOP v1 (owned by Claude) completed 2026-09-17: results/recipe10x/isoflop-v1/RESULTS.md. Data-v1 store complete (2025-01..2026-08, 1.83B games). Active (Claude, user-approved 2026-09-18): bigrun-v1, a two-day 8xL40S run of our recipe at 28x1792 (~1.08B non-embedding params), results/recipe10x/bigrun-v1 (PAUSED by user 2026-09-18 ~22:55 at step 9680/43000, checkpoint intact for exact resume; its 8 general L40S reallocated to data/model work); and data-v1 mixing experiments (waves 1–3 on A6000/preempt, ledger results/recipe10x/data-v1-ledger.md) selected on the golden eval; the data-v1-hist store (2023–2024) is building. Targets (user, 2026-09-18): ≥1.5× compute multiplier on golden macro and ≥4× on golden expert macro vs the control mix; after the 3e17 wave, shift the ladder down ~3× (1e16/3e16/1e17) if results transfer. Do not duplicate or cancel these jobs. Model track (Claude, user 2026-09-18 ~22:40): hill-climb model architecture / training recipe on the pinned round-3 data baseline (B_2) with the same round procedure (3e16 screens at matched trainer FLOPs, 1σ gate, 1e17 confirm, stack); stop at ≥4× golden CM over the control model; dei-group ≤16 A6000 for the model track, preempt ≤24 for data (round 3 on).

Current code and model documentation: README.md. Budget state: results/compute.json. Controller operation: CONTROLLER.md.
