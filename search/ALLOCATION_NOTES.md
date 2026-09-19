# Distribution prediction and search allocation

The target is human move likelihood, not the value of the move played by an
engine. Those objectives can prefer different search schedules. No external
memory is allowed. All quantities below come from the frozen model, current
game, rules, and this query's own search.

## Two costs that must remain separate

1. **Simulated human deliberation.** How much search and optimization should the
   behavioral model attribute to the player? This can depend on rating, clocks,
   and the position.
2. **Our numerical approximation cost.** How many model evaluations are needed
   to estimate that behavioral distribution accurately?

Increasing our node count should not silently make the simulated player more
rational. The deep-scale study therefore compares the current count-dependent
backup to a budget-normalized version and a constant-temperature version. The
search path remains fixed across these backup ablations. A selected backup would
need a later check with branch selection using the same values.

This distinction is consistent with latent inference-budget models, which
explicitly marginalize over an agent's search runtime. Their chess results
support trying runtime mixtures; they do not establish that our particular
policy/value model benefits from one. [Jacob et al., 2023](https://arxiv.org/pdf/2312.04030).

## Local allocation derivation

For an idealized policy `pi = softmax(log prior + beta Q)`, small value errors
`e` induce

```
KL(pi(Q) || pi(Q+e))
  ~= beta^2 / 2 * e.T [diag(pi) - pi pi.T] e.
```

If errors across actions were independent with variance `s_i^2 / n_i`, the
expected discrepancy would be proportional to

```
sum_i pi_i (1-pi_i) s_i^2 / n_i.
```

Minimizing it at a fixed total number of samples gives

```
n_i proportional to s_i sqrt(pi_i (1-pi_i)).
```

This is a local approximation, not a theorem about MCTS or human CE. Our search
estimates are correlated and biased, branch evaluations have different costs,
and the deployed output is a calibrated mixture rather than one softmax.
Replacing pi by the root prior and s_i by a constant recovers the current root
coverage rule. The quota-replay experiment tests whether an action-specific
instability proxy improves that approximation.

## What the first tests establish

- Across positions, early value changes predict later value changes well, but
  explain very little variation in human CE improvement. Value convergence is
  not the objective itself.
- Marginalizing uncertainty along `Q1000-Q512`, a diagonal approximation, and
  signed extrapolation did not yield a resolved development win.
- These results do not rule out using instability to allocate computations.
  That is a different intervention, tested separately at 1000 simulations.

The quota screen uses only Q128 and Q256, then distributes the remaining744
simulations. It tests a fixed floor on the observed change and a visit-scaled
version. Both preserve exploration with bounded multipliers. It rejects trace
exhaustion rather than silently substituting a different action.

Research on thinking time relates human time allocation to expected computation
benefit. Its uncertainty estimate uses the eventual shallow/deep difference as
an explanatory approximation, explicitly not a causal estimation algorithm.
We cannot use our eventual deep result to choose whether to compute that result;
only already-paid prefix information is eligible. [Russek et al., 2025,
section 6.1](https://cocosci.princeton.edu/papers/RussekThinking2025.pdf).

## Evidence required before promotion

The independent root-action traces allow exact replay conditional on stored NN
outputs. The unchanged quota must recover Q and node counts bit for bit; all
recorded birth times must agree with the reconstructed schedule. Future value
changes must leave the early allocation inputs unchanged. Those checks passed
on the first64-position block; every block enforces the baseline checks again.

This is not an end-to-end invariance proof. A live reallocation changes GPU
batching and may change BF16 predictions. Any candidate must be run live before
golden evaluation, compared at actual evaluated nodes and wall time, with all
parameter selection confined to development games. No golden CM is assigned to
August losses.
