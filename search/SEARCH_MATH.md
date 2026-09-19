# Search for prediction rather than playing strength

The target is 10× training-equivalent CM on both golden macros, with an improved
average model-node/CE frontier. Laws and checkpoint stay fixed. Macro targets are
CE 1.383204 and expert CE 1.299293. The law is a conditional training equivalence,
not a measurement of a tenfold training saving.

An exact autoregressive model does not improve its next-move distribution simply
by summing its own future rollouts: their conditional probabilities sum to one.
Search must correct approximation error, reconcile auxiliary predictions, or
introduce a better behavioral model. Stronger chess play alone is insufficient.

## Preserve human-policy tails

For legal prior p and score Q, forward-KL regularization gives
π(a) ∝ p(a) exp(βQ(a)). Reverse KL gives π(a) = λp(a)/(ν−Q(a)), λ=1/β,
where ν normalizes the distribution. Since ν ≤ max Q + λ,

π_reverse(a) ≥ p(a)/(1 + β range(Q)).

The corresponding forward lower bound is p(a) exp(−β range(Q)). Reverse KL thus
limits how aggressively value errors suppress moves humans might still play.
This is a mathematical property, not proof that reverse KL always predicts better.
Our golden1000 result favors reverse over forward with the frozen calibrations.
A mixture (1−η)p + ηπ_search gives a still more explicit floor (1−η)p and caps
per-position CE degradation relative to p at −log(1−η). Fit η on representative
development games; do not use golden to choose its value.

## What a deeper critic estimates

If V exactly predicts outcomes under policy p, the tower property gives
V(s)=Σ_a p(a|s)V(s,a). Policy-expectation rollout then agrees with the shallow
critic in expectation. Disagreement measures inconsistency. An optimizing search
additionally changes the assumed continuation behavior; its Q need not match V.

We tested separate shallow/search coefficients on cached blitz development trees.
Pure Q_search−Q_shallow was worse. Combining six-ply policy expectation with MCTS
improved that dev loss but pays for both trees; it does not establish a frontier win.

The next implementation computes several backups on ONE fixed tree. Unexpanded
action mass keeps its parent's critic. For current-mover action values q:

V_τ(s)=τ log Σ_a p(a|s) exp(q(a)/τ).

τ→∞ is policy expectation; τ→0 is maximization. Opponent signs flip on every
edge; checkmate/draw values come from rules. These backups use identical model
nodes. CPU traversal cost is measured separately. Independent recursive tests,
terminal-sign checks, original-MCTS identity and per-root node counts passed.
The first shared-calibration blitz comparison did not beat native MCTS averaging.
Representative development calibration is required before a broader conclusion.

## Reconcile outcome predictions instead of always rewarding wins

The model predicts root outcome probabilities r(o), and searched continuations
can estimate q(o|a), o∈{win,draw,loss}. Coherence would require
r(o)=Σ_a p(a)q(o|a). Mean win-minus-loss discards draw/risk information.

A proposed experiment (NOT implemented or validated yet) preserves the prior as
far as possible while reducing this discrepancy:

min_π KL(π||p) + κ · divergence(r, Σ_a π(a)q(·|a)).

An exact moment constraint, when feasible, yields an exponential tilt whose dual
coefficients are chosen from the auxiliary consistency condition, rather than
from an assumption that every human maximizes wins. Use soft constraints or bounded
mixtures when r lies outside the convex hull of the child distributions. A necessary
negative control is the already-coherent case: π must remain p. Another is κ=0.
No actual game result or future human move may enter this inference computation.

Alternative behavioral hypothesis: marginalize a latent search budget using the
predicted thinking-time distribution, rather than plug its mean into one search.
These operations are generally unequal. A shared-prefix simulation ladder makes
mixtures cheap, but useful time information must beat a shuffled-time control.
The current equal-total-budget routing experiment did NOT pass that test.

## How to decide

Use representative 16-cell development games, game-disjoint fit/confirmation folds,
and regularize calibration complexity. The old entire dev inventory is blitz-only;
its Elo-calibration win failed golden transfer. Report failures rather than retune
against those golden results. Freeze methods before each golden confirmation.

Report both CMs, per-cell CE, paired game uncertainty, average actual new model
nodes (with expert cost separately), nominal simulations and wall time. Compare
budget curves, not an expensive method against a much cheaper baseline. A point
estimate of Pareto dominance is distinct from a statistically supported one.
# August research additions

The user chose August games for tuning, including games potentially seen in model
training. Only unchanged July golden establishes held-out gains. Game splits and
fit-only cross-validation prevent fitting output coefficients to the same game
used for the immediate parameter check, but do not remove model-training overlap.

The full categorical outcome head contains draw information discarded by scalar
win-minus-loss backups. For each tree node, `local_visits = visits - sum(child
visits)`; its WDL sum is local_visits times its own predicted WDL plus each child's
WDL sum with win/loss exchanged. This reconstructs the original scalar visit mean
exactly, including terminal and depth-limited visits, with no extra neural nodes.

For root outcome distribution r and action-conditioned outcomes q(o|a), let
qbar(o)=sum_a p(a)q(o|a). One coherent Bayesian mixture correction is
p_new(a)=p(a) sum_o r(o) q(o|a)/qbar(o). It is normalized, and equals p exactly when
r=qbar. This is an error-correction hypothesis, not a guarantee of lower human CE.
The August experiment tests its log-ratio as a calibrated feature alongside Q and
draw probability. Missing/unvisited action outcomes inherit the root distribution;
they are never interpreted as certain draws. Scalar-MCTS controls retain Q=0 for
unvisited actions so their original behavior is preserved.

Next candidate, not yet implemented: remove root-action value blindness from MCTS.
The current tree assigns Q=0 to unvisited actions, regardless of whether a position
is winning or losing. This can inflate unvisited tails in losing positions and
suppress them in winning positions. Fixed-depth search evaluates every legal root
action and avoids this mismatch. Compare an all-root critic prepass plus MCTS with
the original tree at equal total evaluated nodes; then distinguish (a) changing
allocation, (b) changing the unvisited output fallback, and (c) shrinking noisy
visit means toward their first critic. Do not attribute a more expensive prepass
to an algorithmic gain without charging its nodes.

Another next candidate (not implemented): learn a budget rule from root features
using the cached 16/64/256/1000 ladder. Predict per-position loss reductions from
policy entropy, rating/format, root WDL, predicted time and history length; choose
the budget maximizing predicted gain minus a fixed node-cost penalty. The human
target is used only to fit this small calibration rule on August, never at search
time. Use nested game cross-validation: calibrate node-budget output policies
inside each router-training fold, so held-out labels do not affect its targets.
Compare to fixed budgets, time-only allocation and a shuffled-time control at the
same realized average node cost. Actual adaptive batching needs a numerical/cost
check against the cached stopping-policy simulation before claiming a deployment
speedup. The present budget snapshots use a constant selection cpuct, so their
prefixes are valid fixed-budget trees; a budget-dependent cpuct would invalidate
that reuse.


## Soft-value-guided allocation (2026-09-19)

The successful output currently uses a regularized backup
V(s)=tau log sum_a pi(a|s) exp(-V(s a)/tau), tau=0.1, with the current
model value for unexpanded edges. Selection still used averaged rollout
returns. These estimates answer different questions. The soft-selection pilot
substitutes -V(child) for the visited edge's rollout mean in PUCT, retaining
bootstrap first-play urgency, priors and the same simulation cap. Incremental
updates recompute only the changed leaf-to-root path; independent recomputation
of every node agrees, including terminal signs. This is a consistency hypothesis,
not a theorem of superior finite-budget human prediction: stronger best-response
planning can diverge from human choices, and optimism about unexpanded actions
can redirect work poorly. Match nodes and measure the policy on separate games.

The value-scale calibration probe treats the root correction strength as a
function of root-predicted time, policy entropy, absolute predicted value and
searched-action value spread. Its contextual weights are shared across Elo
groups and regularized; all feature normalization and fits stay inside game CV.
No observed human thinking time or future outcome is an input. The unrestricted
linear beta may become negative in rare states, so this variant is a predictive
logit calibration, not a claim of a universally positive rationality coefficient.


## Coverage rather than best-move identification

For an idealized unbiased independent action-value estimate with variance
sigma_a^2/n_a, a local softmax-policy KL expansion gives error proportional to
sum_a p_a(1-p_a) sigma_a^2/n_a, times a common beta^2/2 factor.
Minimizing that surrogate under sum n_a=B yields n_a proportional to
sqrt(p_a(1-p_a) sigma_a^2). Search values are biased and correlated, so this is a
heuristic motivation, not a guarantee. The test uses sqrt(prior*(1-prior)),
prior^0.5 (dropping the1-p factor) or prior (proportional coverage) as root quotas,
implemented by choosing max weight/(1+visits). It keeps ordinary PUCT below
root. This separates estimating the plausible-action distribution from the
best-arm focus of playing-strength search. All likelihood calibration stays
on August fit games, and node cost includes terminal visits separately.

## Allocate internal work by its influence on the returned value

The root-coverage intervention improved the reused golden sample: at about976
nodes, macro/expert CE1.45297/1.35634, conditional CM2.65x/4.00x. Its actual
Elo-budget router needs615 nodes and gives1.45405/1.35634. This does not complete
the10x goal or replace fresh confirmation. Calibration normalization and commuting
history averaging did not establish additional gains.

For a soft Bellman value V=tau*log(sum_a prior_a*exp(q_a/tau)), the derivative
with respect to q_a is the tilted distribution w_a proportional to
prior_a*exp(q_a/tau). Under independent local estimation noise sigma_a^2/n_a,
the delta-method value variance is sum_a w_a^2*sigma_a^2/n_a. Holding the local
weights fixed, minimizing it at fixed total samples gives n_a proportional to
w_a*sigma_a. This differs from the root likelihood surrogate's sqrt(p*(1-p)).

The next test retains the successful root quota, and allocates internal visits
by max(weight/(1+visits)): either the model prior, current soft-value derivative,
or75% derivative +25% prior. The prior mixture guards against an erroneous early
critic monopolizing later work. Soft values are recomputed along changed paths;
the output backup and model calls are otherwise unchanged. Independent recursive
values and selected-path tests must pass, including exact identity with ordinary
root-coverage/PUCT when the internal change is disabled. This is a finite-budget
heuristic: actual neural value errors are biased, correlated and not independent
Monte Carlo samples. The derivative calculation alone predicts no quality win.

A separate recursive-critic regularization test used
V=(1-lambda)*critic+lambda*soft_backup, with exact terminal values. Its smaller
lambdas worsen August confirmation, so lambda1 stays the baseline. Unlike a
root-only shallow/deep mixture, this attenuates corrections at every tree depth;
that mathematical distinction did not make it empirically helpful.
