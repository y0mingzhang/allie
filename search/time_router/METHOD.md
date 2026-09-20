# Time-aware search and repaired Allie

This follow-up tests search allocation on the finished 129M ship checkpoint. It
reuses the original transfer study's August development games and July evaluation
inventory. The old inference stage and its results are unchanged.

The comparison uses one L40S, the same 256-root batches, and actual new neural
node evaluations. A forced move uses zero search nodes. Root prefills, rule-only
terminal work and startup are reported separately rather than counted as nodes.

**Coverage** evaluates all root moves under a policy-dependent coverage allocation,
then uses a count-dependent soft value backup and the previously frozen calibrated
output policy. **Repaired Allie** uses PUCT, visit-average value backup, the
first-prior/depth repairs, and a reverse-KL output policy. Its per-Elo output
calibration is fitted separately at every budget. Its exploration constant is
chosen from 0.5, 1.25 and 2.5 using game-wise development CV, then held fixed across
routers. The alpha/beta readout fit is repeated inside each CV fold.

Each method gets budgets 128/256/512/1000. An initial 64-step prefix is also saved.
The utility routers predict the CE change relative to 128 simulations and select
one budget per position under a mean-cost constraint:

- Elo: predicted gain depends on the four mover-Elo groups.
- Format: gain depends on the 16 format × Elo cells.
- Time: adds base time, increment, both remaining clocks, the mover's previous
  own think time, and low-clock indicators, interacted with Elo.

All use the same development-fitted cost table by cell. Thus the Elo utility
model can respond to format differences in *cost*, but not in predicted gain.
A deterministic prefix hash resolves allocation ties without looking at the
played move. Ridge strength is selected on the fit games; confirmation games
never choose settings. The time-aware input never contains the thinking time of
the move being predicted.

A further Allie arm assigns budgets proportional to the root model's predicted
thinking seconds. The scale is calibrated without move labels, and the budget is
clipped/discretized to the same four choices. This is an Allie-inspired budget
ablation, not an exact reproduction of the paper's full continuous allocation:
the exploration constant and calibrated readout remain those selected above.

All arms target 460 actual NN evaluations per position. Query-pool threshold
calibration uses root features and development cost estimates. If any arm's
realized cost differs by more than 2%, every arm receives the same additional
label-free aggregate cost calibration. A separate diagnostic fixes each cell's
mean cost to its Elo-router reference, separating gains from moving compute
between time controls from gains within a cell. Both use cached trees only for
calibration; actual mixed-budget reruns are authoritative because batching can
change BF16 numerics and hence tree paths.

July reports use the full canonical per-cell CE plus a paired difference on
8,192 sampled positions (512/cell). Intervals resample whole games. This is a
reused golden sample, not a fresh final confirmation. The final test split stays
closed. CM is an estimate from the prior scaling-law shape anchored at rung
3e17 for this checkpoint; it is not measured extra training. Search-only CM
excludes legal-move renormalization. Neither these intervals nor a CM point
estimate include uncertainty in the scaling-law shape.

The Allie comparison changes the tree backup and readout as a package: coverage
has its earlier frozen adaptive output calibration, while Allie has newly fitted
per-Elo reverse-KL calibration. A loss difference cannot by itself isolate which
component caused it.
