# Time-aware routing and matched-cost Allie

PARKED by user 2026-09-19 ~21:15 EDT: commit/merge inference and move to MoE. No further inference runs authorized by this goal.

Originally user-authorized 2026-09-19. The completed transfer study remains closed.

Test whether adding time control and pre-move clocks to Elo improves search
allocation at equal mean actual NN evaluations. Give repaired Allie a separately
tuned, matched-cost comparison. Primary checkpoint: finished 129M ship recipe,
read-only. No neural training, external memory, engine or test-split use.

Reuse the exact August fit/confirmation games and July golden inventory from
transfer-v1. Fit only on August fold0 with whole-game CV; report fold1 without
selecting on it. Freeze before July scoring. Golden is a reused evaluation,
not fresh final confirmation. Report all preregistered arms, paired game CIs,
actual NN counts, per-cell/clock allocation and conditional CM.

One general L40S at most, peer coordinated. Initial collection: same-hardware
coverage control and repaired-Allie cpuct0.5/1.25/2.5, cumulative budgets
64/128/256/512/1000. Repairs retain PUCT and visit-average values; calibrate
reverse-KL alpha/beta per Elo on development data. No claim of equal cost from
nominal simulations alone. Three router families: Elo; Elo+format; Elo+format+
base time/increment/remaining clocks/previous own think time. Never use the
target move's thinking time as a feature.

Keep all previous charges (15.727778 GPU-hours). Outputs: time-router-v1 under
the durable search results. Own STOP is study-local; the earlier study STOP
stays intact. Global controller STOP always applies. No duplicate jobs.

Finish after the frozen matched-cost evaluation and concise results report;
do not reopen the old10×/10× search goal.

At park: all development grids and both July grids complete; live coverage Elo/format/time complete; live Allie and per-cell checks unfinished. Allie cached results are provisional. The previous completed transfer report remains the validated recommendation. See REPORT.md and RECOVERY.md.
