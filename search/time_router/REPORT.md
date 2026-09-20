# Time-aware search and matched-cost repaired Allie

**Parked by the user, 2026-09-19. Keep the previously validated coverage search with its frozen Elo budget. Move development effort to MoE.**

The live time-aware follow-up did not establish an improvement. No new inference method is promoted. All development grids and both July fixed-budget grids finished. Three coverage routers finished live evaluation; Allie live evaluation and the within-cell diagnostics were stopped at the user’s request.

## Validated recommendation

The preceding completed transfer study’s frozen Elo-adaptive coverage search: **1.3375 macro CE / 1.1798 expert CE at 461.4 actual neural nodes**. Conditional training-equivalent CM is **1.52× / 3.86× relative to the legal policy** (1.57× / 4.00× including port/legality effects relative to canonical raw).

This method covers root moves, uses a count-dependent soft value backup, then calibrates the output policy. Its frozen router assigns budgets by mover Elo. This is the same recommendation as the completed transfer report; the follow-up below refits allocation at a common 460-node target and uses 256-root batches. Differences from the preceding study are not isolated time-feature effects.

## Completed live follow-up

| Router | Mean nodes | Macro CE | Expert CE | Macro CM | Expert CM |
|---|---:|---:|---:|---:|---:|
| coverage-elo | 459.3 | 1.3381 | 1.1817 | 1.50× | 3.68× |
| coverage-format | 459.6 | 1.3384 | 1.1803 | 1.49× | 3.83× |
| coverage-time | 459.5 | 1.3391 | 1.1812 | 1.46× | 3.73× |

CM columns exclude legality normalization. These are actual mixed-budget runs, not cached-stop estimates.

| Difference versus new Elo router | Macro Δ [95% CI] | Expert Δ [95% CI] |
|---|---:|---:|
| coverage-format | +0.0003 [-0.0013, +0.0018] | -0.0015 [-0.0061, +0.0029] |
| coverage-time | +0.0010 [-0.0006, +0.0028] | -0.0005 [-0.0050, +0.0043] |

Neither added-format nor added-clock comparison excludes zero. The time-aware router’s small expert point gain trades against worse macro CE. That does not justify another broad inference sweep now.

## Repaired Allie: provisional cached results

Allie received cpuct 0.5/1.25/2.5 sweeps, per-Elo reverse-KL calibration separately at every budget, and the same cost target. Fit-game CV selected cpuct 2.5, held fixed across routers. This is a much stronger evaluation than the earlier 50-node arm, but it is bounded, not an exhaustive search over Allie designs.

| Cached method | Mean nodes | Macro CE | Expert CE | Macro CM | Expert CM |
|---|---:|---:|---:|---:|---:|
| coverage-elo | 459.3 | 1.3380 | 1.1814 | 1.51× | 3.71× |
| allie-elo | 461.3 | 1.3440 | 1.2066 | 1.29× | 2.00× |
| allie-format | 461.4 | 1.3435 | 1.2018 | 1.31× | 2.23× |
| allie-time | 461.3 | 1.3437 | 1.2024 | 1.30× | 2.20× |
| allie-predicted-time | 463.3 | 1.3429 | 1.2027 | 1.33× | 2.18× |
| allie-fixed1000 | 961.9 | 1.3420 | 1.1996 | 1.36× | 2.35× |

These Allie numbers remain provisional: its live mixed-budget run was not completed. Allie is beneficial relative to the legal policy, but trails coverage in this cached comparison. Do not label every adaptive-MCTS design strictly dominated. The predicted-time arm discretizes proportional thinking-time budgets to 128/256/512/1000; it does not reproduce the original paper’s entire budget/exploration schedule.

## Limits and retained evidence

- Same finished 129M ship checkpoint throughout, no neural training, retrieval or external engine. Future clocks in tree nodes come from the model’s predicted thinking time. Router features use only pre-move information.
- Development: 2,048 August fit positions with whole-game CV plus 2,048 disjoint confirmation positions. August may be training-seen. Exploration/ridge settings were frozen before July scoring.
- July: full canonical cell CE anchored with paired differences on 512 positions per cell. This golden sample has been reused; no fresh final-confirmation claim. Final test games remain unopened.
- CM is conditional on the prior scaling-law shape at rung 3e17. Bootstrap intervals include whole-game sampling, not scaling-law uncertainty or the full multiple-testing family.
- Live versus cached coverage macro changes were 0.00004–0.00010; expert changes 0.00017–0.00032. Individual policy changes reached 0.1565, so cached routing is not exactly batch invariant. All reported live/cached delta intervals included zero.
- Coverage and Allie use different output-calibration families. Their package comparison does not isolate the backup rule.
- Allocation cost is actual non-root neural evaluations. Root prefills are additional. All completed fixed-budget curves, per-cell costs, and clock diagnostics remain in JSON; unfinished runs are not filled in or presented as completed.
- Resource usage: general10506788, 1,288s; preempt10506918, 771s; sequential single L40S allocations, **0.572 GPU-hours** total. Both exited; all previous compute charges remain.

Durable evidence: `results/search-v1/time-router-v1/{plan.json,frozen.json,gold-allocations.json,cached-results.json,live-partial-results.json}`. The earlier validated recommendation is in `search/TRANSFER_REPORT.md` and `results/search-v1/transfer-v1/results-large.json`. Source and recovery: `search/time_router/`.
