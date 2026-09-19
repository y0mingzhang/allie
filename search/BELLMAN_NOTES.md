# Critic consistency after search

The model predicts expected outcome `y = P(win) - P(loss)` from the side to move.
If its outcome head and move policy described the same human continuation process
perfectly, they would satisfy

```
v_i = -sum_j p_ij v_j
```

The minus sign changes player perspective. The probabilities are the original
legal-renormalized move priors, at the actual player ratings. This identity concerns
human-policy outcomes; it is separate from the soft optimization used for the final
searched policy.

The heads are imperfect. Instead of trusting every critic independently, the new
method adjusts all visited critics toward consistency, while penalizing changes
from their predictions. Search expansion still uses the original critic. Only the
final backup changes, so the method consumes exactly the original tree's NN calls.

## The surrogate we actually solve

Let `r_i` be the prior mass of unexpanded moves. Its aggregate outcome has prior
mean `y_i` and variance1. Nonterminal node measurements also have variance1.
These are relative regularization choices, not calibrated uncertainty estimates.
With a consistency strength `lambda`, minimize

```
sum_nonterminal_i (v_i - y_i)^2
  + sum_expanded_i
      (v_i + sum_observed_j p_ij v_j - r_i y_i)^2
      / (1/lambda + r_i^2).
```

Rule-terminal values are fixed exactly. Unobserved actions do not become data.
With lambda0 the projection is bypassed and the original action values are
recovered exactly. After projection, clip the adjusted critics to[-1,1], then run
the existing prior-weighted soft backup and development-fitted output calibration.
Clipping means the served values are not precisely the unconstrained Gaussian
solution; its frequency is reported (about0.0037% of nodes at lambda3 in the
expanded development study).

## Why this is cheap

The node/factor graph is a tree. An upward pass summarizes each subtree by a
Gaussian mean `m_j` and variance `s_j^2`. For a nonterminal parent define

```
T_i = 1/lambda + r_i^2 + sum_j p_ij^2 s_j^2
R_i = sum_j p_ij m_j - r_i y_i
s_i^2 = T_i / (1 + T_i)
m_i = s_i^2 (y_i - R_i/T_i).
```

The root posterior mean is its upward mean. A downward pass propagates the
ancestor information:

```
v_j = m_j - s_j^2 p_ij (v_i + R_i) / T_i.
```

Terminal variances are zero. Nodes without observed children retain their neural
measurement. Both passes are linear in the number of visited nodes. The native
implementation agrees with an independent dense normal-equation solve across
different sizes, strengths and partial-tree budgets. Tests also cover exact
terminals, zero-strength identity and exclusion of nodes beyond the paid budget.

## What the results do and do not establish

On the expanded reused August development cohort, both game-CV metrics selected
lambda3 over0/1/3/10. Checking CE improved by0.00134 macro and0.00464 expert, with
nominal paired intervals favoring both. This is small compared with the remaining
10x/10x target gap. Projection plus soft backup took1.87CPU seconds across16,384
positions, excluding cache loading and neural search. The frozen golden comparison
is a separate study with a paired fresh-search control.

Neural errors are correlated, expansion is selective, and unseen-mass fallback is
self-referential. None of the Gaussian variances should be read as statistical
confidence in a chess evaluation. In this checkpoint there are no clock inputs;
the later clock-conditioned B2 model would require care about hypothetical future
clock states before asserting the same consistency relation.

There is also an attribution question: a root-only consistency correction shifts
child values in proportion to their prior probabilities. Some gain could therefore
come from nonlinear prior calibration. Planned development controls are root-only,
full upward/downward, upward-only, and exclusion of the root factor, with lambda3
fixed. Those are not yet evidence for or against the mechanism. No variant will be
selected by its golden score.
