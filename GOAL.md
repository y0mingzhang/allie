# Research goal

Finish Qwen versus our architecture scaling laws: obtain a high-fidelity L(N,D) estimate for both recipes.

The thread goal is currently blocked, not complete. All scaling and seed-check runs finished. The simple additive law has systematic residuals; its forecasts and limitations are in README.md and analysis/scaling_fit.json. The user has since discussed expert data mixing, then requested repository cleanup. No new training has been requested in this cleanup task.

## Constraints

- Reuse original lichess_tokens_v2 corpus, revision 20a899ddf344ccaea74e273509a60e5a511125f8, splits and both Elo headers.
- Fit the requested additive law E+A/N^alpha+B/D^beta. Do not silently replace it or treat its conditional forecasts as measured gains.
- The current comparison is move-only, with no board inputs, distillation or search. Expert mixture changes are a proposed experiment, not a completed result.
- Keep final tests unopened. Preserve raw reports, checkpoints, source identities and all compute charges.
- At most 8 L40S plus 16 A6000, with A6000 admission conditional on no competing DEI GPU demand; normal QoS permits 8 GPUs.
- Durable storage: /data/group_data/dei-group/yimingz3/allie. Scratch is a cache.
- Read controller/STOP before autonomous work. It stops controller activity, not independent training. Never stop the current tunnel/controller job.
- No duplicate submissions. All previous tiny-grid, size-validation and seed observers have completed; do not restart them.
- Generate images only when requested.

Current code and model documentation: README.md. Budget state: BIG_RUN.md and results/compute.json. Controller operation: CONTROLLER.md.
