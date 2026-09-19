# Released Allie adaptive MCTS: audit

Reference: ippolito-cmu/allie, commit a50f2d86618798cec2195e37e3484da579631328,
src/evaluation/decode.py. Compared with ICLR 2025 Appendix E.2 and Grill et al.
(2020), equations 4, 7 and 8. This establishes discrepancies in the released code;
it does not establish which executable produced each historical paper result.

Sources:
- https://github.com/ippolito-cmu/allie/blob/a50f2d86618798cec2195e37e3484da579631328/src/evaluation/decode.py
- https://proceedings.iclr.cc/paper_files/paper/2025/file/0ef1afa0daa888d695dcd5e9513bafa3-Paper-Conference.pdf
- https://proceedings.mlr.press/v119/grill20a/grill20a.pdf

## Confirmed paper/code discrepancies

1. KL direction: Appendix E.2 equation 1 uses KL(pi || p). Its exact maximizer is
   pi(a) proportional to p(a)*exp(Q(a)/lambda). Released lines 416-418 and 593-594
   compute lambda*p(a)/(alpha-Q(a)), which solves KL(p || pi), as in Grill's paper.
   Both objectives are defensible; they are different objectives.
2. Final-policy coefficient: released lines 405 and 582 use lambda=c*N/(K+N).
   Grill equation 4 has lambda=c*sqrt(N)/(K+N), where K is legal action count.
   For fixed c, the released coefficient tends to c rather than zero. At fixed
   N=50, it is sqrt(50) times the Grill coefficient using the same c (ignoring
   the small additional logarithmic PUCT term).
3. Adaptive c: Appendix E.2 says c grows with sqrt(N) to preserve approximately
   constant regularization. Released lines 552-555 set c=c0*sqrt(Nmean/N).
   Combined with the released final coefficient, lambda=c0*sqrt(Nmean*N)/(K+N),
   which is not constant across budgets or branching factors. Thus the adaptive
   treatment changes both simulation allocation and the resulting policy's
   regularization, confounding a pure test of where additional computation helps.

The parent-mover value convention and alternating backup signs are internally
consistent. Our adaptation matches the released tree/policy on 24 deterministic
synthetic comparisons. That is equivalence evidence, not proof that the research
objective or historical implementation is correct. Preserve this released-code
baseline; any changed formula is a separately named method.

## Modeling interpretation

Limited search around a learned human prior is a reasonable model of bounded
deliberation. Predicted human time is a plausible budget signal, but the benefit of
an extra simulation is not generally proportional to thinking time. Uncertainty,
remaining clock and tactical structure can matter independently.

An exact human conditional policy already minimizes expected human CE, including
the effects of human deliberation. Extra value optimization would not improve
that ideal predictor. On a finite model, outcome supervision and search can repair
policy approximation errors, especially on expert/tactical positions, but can also
remove authentic human mistakes. A value trained on human continuations is not an
optimal-play value, and applying it on unusual search branches adds another error.

## Next separate ablation

Keep tree allocation and output regularization independent. Compare fixed-budget
and time-adaptive search at matched evaluated leaves and measured time, holding the
same output lambda fixed. Include a shuffled-time allocation control. Fit lambda
on development games, compare reverse-KL and exponential-tilt outputs, and retain
the released baseline unchanged. Existing pilot tree Q/visit caches can test the
output-policy change without spending GPU inference; they cannot establish the
effect of a changed tree-selection rule without new trees.

## Follow-up audit of the external critique (2026-09-19)

The external critique correctly observes the algebraic cancellation in the adaptive
**output** coefficient: c_tree = c0*sqrt(Nmean/N), hence
lambda_out = c0*sqrt(Nmean)*sqrt(N)/(K+N). This has Grill's functional form with
c_output = c0*sqrt(Nmean). However c_tree is not c_output. Ignoring the small log
term, the output coefficient is still sqrt(N) times the coefficient associated
with that tree's own PUCT traversal. The cancellation therefore does not restore
the traversal/output correspondence. Grill's Proposition1 is a local statement
about selection and the regularized objective; the convergence-rate result also
assumes a fixed target policy. It is not a human-prediction improvement guarantee.

Two practical issues deserve independent ablations:
- At zero root visits every selection score is zero; the first rollout follows
  move-generation order. Fixing the exploration numerator to sqrt(max(N,1)) makes
  it follow the learned prior while retaining exact simulation/visit accounting.
  Do not infer the loss effect from the fraction 1/N of simulations affected:
  this visit also changes subsequent selection and value estimates.
- Re-expanding an already expanded node at the depth boundary destroys its child
  statistics. The repaired variant retains its bootstrap value and subtree,
  backs up that value, and spends no redundant network call at the boundary.
  The original depth100/small budgets rarely reach it, but 'unreachable' is too
  strong (and the remaining context can impose a tighter bound).

Our reference adapter already reuses root predictions, avoiding the released
implementation's duplicate root forward pass without changing its tree policy.

Do not claim that an audit of the present public commit proves the exact executable
behind the published results. Nor does finding a mismatch itself invalidate a
measured result. Reproduction and matched-coefficient ablations are needed to
separate empirical performance from its proposed mechanism. Both explicit KL
objectives are valid; their human CE and calibration must be measured.

The development pilot retains released fixed/adaptive distributions, then changes
only the practical repairs, and separately holds tree exploration/output
calibration fixed while comparing fixed, predicted-time, shuffled-time and
entropy allocations. Exact total simulation counts are enforced for the allocation
ablation; actual evaluated leaves and wall time are additionally reported. No
observed thinking time or future outcome enters allocation.
