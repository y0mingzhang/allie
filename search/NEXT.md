# Next inference experiments

All choices below are development-only. Golden method choices are already frozen in
results/search-v1/golden-balanced-v1. Test/test_expert remain closed. No new training.

1. Finish the first balanced confirmation (queue012) and report every preregistered
   method, with paired intervals, cell regressions and conditional training CM.
2. Benchmark integer-only KV-map inheritance and packed request metadata (queue013).
   Exact outputs and valid slot tables must match before timing; no automatic rollout
   merely because an implementation looks faster. Report startup separately.
3. MCTS at1000 simulations (queue014), with released and repaired trees and calibration
   learned only on original dev fold0. This is much closer to four-ply node cost than
   the earlier50-simulation comparison. Report actual nodes/tokens/wall time.
4. Six-ply expectation (queue015), widths4,2,2,2,2. One traversal returns every horizon.
   Fit the original alpha/beta correction for each horizon and one declared extra arm:
   alpha*logprior + beta*Qdeep + gamma*(Qdeep-Q2). Report all arms on dev fold1.
   The extra term tests whether shallow/deep disagreement predicts critic bias.

Next independent idea: policy-sensitive adaptive expectation, rather than using a
win-maximizing MCTS allocation to estimate a human distribution.

Keep the current expectation estimator: unexpanded continuation probability retains
its parent critic value. Expanding a node changes the root action value by path_mass
* sum_j p_j*(child_value_j-parent_value). It therefore has full root legal support,
can stop at any budget, and reuses the exact same model WDL and move predictions.
Bootstrap every legal root move once; never give unvisited root moves a fictitious
neutral Q=0 merely because the budget is small. Preserve terminal/draw/context rules.

A possible allocation objective follows from a LOCAL surrogate, not a claim about
true human optimality. If pi*=softmax(alpha*logprior+beta*Q*) and critic errors e are
small, then KL(pi* || pi(Q*+e)) is approximately beta^2/2 times
sum_a pi_a*e_a^2 - (sum_a pi_a*e_a)^2. Independent zero-mean errors give expected
cost proportional to sum_a pi_a*(1-pi_a)*Var(e_a). A frontier critic error contributes
path_mass^2 times its own error variance. This suggests expanding frontier nodes by
estimated root-policy error reduction per model call, instead of largest win Q.

Development ablations should isolate path mass alone, policy sensitivity, and a
model-only uncertainty proxy. WDL outcome variance is NOT epistemic critic variance;
it is only a proxy to test. Start with fixed per-position budgets, batch several
independent frontier choices across roots, and compare against four-ply and calibrated
MCTS at measured node count and time. No claim of a theoretical guarantee on human CE.

All adaptive policies retain a nonzero legal prior and frozen dev calibration. Any
method that skips a position falls back and remains in the denominator. All reports
include failed ideas; golden results never tune these choices.

Update after the first dev extensions (2026-09-19):
- 1000-simulation calibrated MCTS beats four-ply on reused dev; six-ply improves
  continuation but spends about4x the nodes for expertCE comparable to MCTS1000.
- The mass/policy/variance adaptive-continuation priorities are nearly tied.
- Cached MCTS postprocessing (search/mcts_postprocess.py) selects Elo-dependent
  value weighting by fit-game CV for both the rating macro and expert metrics.
  Extra GPU cost is zero. Shrinking root Q toward the root critic loses quality.
- Queue019 tests4000 fixed simulations. Queue020 retries Allie allocation at
  exactly1000 simulations/root on average, with predicted-time-only allocation,
  the paper's inverse-sqrt exploration coupling, and shuffled-time allocation.
  Fixed controls are reused byte for byte from014 with provenance checks.
  Output calibration stays shared across arms first. Allocation alone and time-
  dependent output regularization are distinct effects; analyze them separately.
- All higher-budget work remains development-only; do not convert its losses with
  golden CM laws or pick parameters against the first golden comparison.
