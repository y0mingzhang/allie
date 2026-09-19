# Current inference strategy

Predict the distribution of human moves, using the fixed model and this game's
information. A stronger playing engine need not be a better predictor. No
external memory, other-game retrieval, external evaluator, or weight updates.

## What the current algorithm does

1. The model supplies a move prior and a win/draw/loss critic.
2. Root actions receive coverage proportional to `sqrt(p(1-p))`, scheduled as
   weighted quotas. Internal action subtrees use PUCT with exploration 2.5.
   This preserves plausible human alternatives instead of spending all work
   resolving the strongest root move.
3. A prior-weighted soft Bellman backup converts the partial tree into action
   values. Unexpanded prior mass retains a critic fallback; terminal results
   are exact. Backup temperature depends on explored descendants, with a
   budget-normalized variant tested when increasing the total budget.
4. The output combines actual-human prior and searched values, roughly
   `p^alpha * exp(beta Q)`. Small calibrated mixtures capture variation in
   search strength; rating, format and legitimate pre-move information can
   condition it. Parameters are fitted on development games, not golden.
5. Adaptive depth first buys 128 simulations and then chooses a stopping
   budget from already available information. It cannot inspect the later
   values before deciding to compute them.

This remains an MCTS-family algorithm. Relative to the Allie-style reference,
the principal changes are root coverage, soft value backup/output calibration,
and allocation by estimated prediction benefit rather than only predicted
thinking time. These are separate ablation axes, not one indivisible method.

## Evidence and next decisions

- Search has established substantial golden gains; exploratory results remain
  around 3x macro / 5x expert training-equivalent CM, short of 10x/10x.
- More depth improves point estimates but has diminishing returns. Selective
  depth is a candidate cost improvement, not yet established dominance.
- Identical-order repeats on the current A100 agree exactly across 4,096
  development positions. Permuting fixed-search batches shifts macro/expert
  CE by about -0.00013/-0.00032. Adaptive permutation deltas were -0.000052/-0.000041; individual policies can vary more.
- Entire-continuation rating interventions actual/+200/+400/equal2400 are complete. No resolved checking gain; no promotion.
- FP32 WDL-only readout reduces critic numerical tails but gave no resolved CE gain on the full paired development study. Default unchanged.
- Complementary two500-search portfolios lose to one1000 control; no promotion.
- Tree-wide critic consistency is the current candidate. Project model values toward human-policy Bellman consistency, then apply the existing soft backup. Expanded August selects lambda3 and supports small paired gains at identical NN cost. Frozen golden comparison109 is underway; no10x claim.

Prioritize a clear intervention over a large parameter sweep. A null result
rules out promoting that tested variant, not its entire algorithm family.
Any claimed winner must beat controls at measured cost, survive game-level
uncertainty and a fresh disjoint golden confirmation, and report all 16 cells.
See `RECOVERY.md` for live job/request IDs and `results/search-v1/REPORT.md`
for the complete results ledger.
