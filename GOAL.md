# Model-only inference research

Owner: Codex. Worktree: `codex/search-v1`, independent of Claude's main worktree.
Status: active, explicitly resumed by Yiming on 2026-09-19 after the usage reset.

Latest constraint (Yiming,2026-09-19): NO EXTERNAL MEMORY. Human-game retrieval,
nearest-neighbor datastores, opening books of other games, cross-game player
profiles and episodic memories are excluded from further inference research.
Allowed inference information: fixed model, current game prefix/header/pre-move
context, chess rules, and the current query's search/KV state. Cached model
outputs may support controlled research comparisons; they must not introduce
information from other games into a prediction. Preserve historical retrieval
reports as rejected/out-of-scope evidence; do not restart retrieval work.
Ablate components before dismissing an entire idea; prioritize the strongest
available research direction over exhaustive weak variants.

Current work: inference infrastructure is operational. Resume the existing10x/10x
search goal with batched experiment execution and reporting. One preempt GPU (workbench 10503933) while general is reserved for Claude;
the current 84-minute backfill ends at 19:28:51 UTC. Allocation boundaries are not research stop conditions. The previous pause is superseded by the explicit user instruction.
All prior compute charges and fixed-checkpoint targets remain unchanged.

Infrastructure milestone completed: fast hackable SGLang inference port.
Current priority: algorithm research and fair cost/quality comparisons. Latest refinement: make MCTS throughput per
evaluated node comparable to four-ply by batching independent roots, including CPU
tree overhead. After infrastructure, compare corrected adaptive MCTS with the released
algorithm and continuation expectation at matched compute; isolate predicted-time
allocation from other uncertainty signals. Validate full-prefix versus cached branch
outputs, then measure startup and end-to-end search latency. Cheap x86 CPU hosting
on Lichess should remain possible through portable model math; implementing that
hosting path is explicitly deferred. Keep the approved research target below.

Evaluation update: a smaller balanced sample of the existing 16 time-control ×
skill cells is now allowed. Preserve the macro/expert definitions and use paired
game-level uncertainty to size it; do not require every experiment to score all
1.55M golden moves. Keep tuning and confirmation games separate and report cell
regressions, not only aggregate improvements.

Improve human move prediction through inference-time methods using the model itself,
starting with `r2-3e16-control-t20-w20-pf052h-s42`. No Stockfish or external value model.
The small model is deliberate: optimize experiment speed and build reusable infrastructure.

Success (Yiming's latest instruction, 2026-09-18): one frozen inference method must
achieve BOTH at least 10x golden macro training-equivalent CM AND at least 10x golden
expert-macro training-equivalent CM over the same checkpoint's raw direct policy.
Do not stop at the earlier 0.01-nat threshold. Select methods on separate development
games before golden confirmation. Keep the base checkpoint and reference laws fixed
when assessing these targets; changing either requires an explicit new comparison.
Report direct-policy calibration and legal-mask baselines separately so their gains
are not misattributed to search. Also report expert move accuracy, all 16 cell losses,
calibration, and measured inference cost.

For the initial checkpoint at ND=3e16, the frozen, vertically anchored independent
isoflop laws imply these targets (conditional estimates, not empirical learning curves):
- Macro: raw CE 1.5240077557 -> CE <= 1.3832044317 (gain >= 0.1408033240).
- Expert macro: raw CE 1.4764288958 -> CE <= 1.2992929938 (gain >= 0.1771359021).
Reference snapshots and calculations: results/search-v1/training-cm-laws.json and
results/search-v1/goal-targets-10x-both.json (the earlier goal-targets.json remains
historical). Report law sensitivity and paired uncertainty;
a point estimate alone is insufficient evidence of a robust improvement.

Yiming requests training-equivalent CM whenever reporting results. For golden macro
and expert macro, invert the corresponding frozen compute-optimal training law,
anchored to this checkpoint's raw CE at its own training budget. Report total CM
versus raw and incremental CM versus cheap calibration on that same curve. Label
this a fitted training equivalence, distinct from measured inference cost. Do not
convert development losses with a golden-evaluation law.

After every completed experiment, show Yiming an updated table of ideas tried,
loss and CM. Maintain results/search-v1/REPORT.md as the cumulative table. Separate
small and expanded development checks from golden macro results; label CM pending
when the available scaling law does not apply to the evaluated split.

Evaluation uses Claude's 16 format × mover-rating cells and four-cell expert macro.
Do not tune against strat-eval-v1. Latest user direction (2026-09-19): use August
games for parameter tuning, even though some were available to model training.
Split tuning/confirmation by game. Label both as potentially training-seen;
held-out quality and CM are established only on unchanged July golden. Earlier
dev/dev_expert inventories are blitz-only and cannot establish macro transfer.
The private July-plus-2024 dev build was validated but is not the selected tuning
dataset. No replacement or relabeling of the golden benchmark.
Freeze method before golden evaluation. Never use the human next move, future moves,
actual thinking time or actual game outcome when selecting candidates or allocating search.

The current golden sample has been reused across research rounds. Maintain a registry
of every method scored on it. Before declaring the 10x goal achieved, freeze the
winner and obtain fresh confirmation with disjoint evaluation games and the same
16-cell definitions; keep test/test_expert unopened. Check the new sampling frame
and its baseline anchor explicitly instead of silently treating a changed frame as
the identical full-population control-variate estimate. Preserve current goal targets.

Resources: ONE reasonably fast persistent preempt GPU while general is reserved
for Claude's training/benchmark. Current workbench 10503933 replaces timed-out 10503413. Keep all prior charges. User permits infrastructure work and a faster
available accelerator; compare end-to-end cost before migrating. Never exceed
one GPU or displace peers. Coordinate the shared preempt cap with Claude.
Latest user instruction: maintain one long-running GPU session for fast iteration.
Start with an eight-hour allocation; record all reserved GPU time, including idle time.
Eight hours is an allocation/recovery boundary, not the new research stop condition.
The earlier two-hour pilot limit is superseded, not charged to training budgets.
Coordinate preempt capacity with Claude. Do not displace/cancel its jobs, alter its
sources, mutate its datasets, or use the final-training compute pool. No model training.

Own code: `search/`. Own artifacts and ledger: `results/search-v1/` in this durable
worktree. Checkpoints and frozen training sources in the original durable store are
read-only inputs. Main-worktree GOAL.md and compute.json remain untouched.

First sequence: establish baseline equivalence; cache root policy/time/WDL and child
WDL predictions; test legal masking and temperature; test prior-preserving shallow
value reranking; test uncertainty/time-adaptive expansion if value adds signal;
then compare deeper model-guided search at measured cost. Record failures as well as wins.
A successful playing-strength search is not automatically a better human predictor.

Required comparison (Yiming, 2026-09-18): Allie-style adaptive MCTS from the ICLR
paper, using the released implementation as reference, plus fixed-budget MCTS.
Pin the source revision and document the adapter from categorical time/WDL heads.
The released default uses a full-support regularized policy, not raw visit counts.
Preserve that output for its CE comparison and measure actual inference cost.

Recovery: read this file, search/PLAN.md and results/search-v1/status.json. Inspect own
job receipts and live Slurm state before submission; never duplicate a job. Respect
/data/group_data/dei-group/yimingz3/allie/controller/STOP. A controller restart does not
resume paused/completed work. Claude owns the main worktree's data/model goals.

Efficiency requirement (Yiming, 2026-09-19): measure the frontier of average nodes
searched over the golden population versus macro and expert CM. Count actual new
model-evaluated search nodes per position, average per cell then macro-average, and
also report nominal simulations, prefix/prefill tokens, and wall time. Report expert
subset cost separately. Show budget sweeps against the existing fixed-ply and MCTS
baselines. A winning algorithm must improve the quality/cost frontier: no greater
average node cost with no worse macro or expert CE, with a strict supported gain
in at least one quality measure. Do not present a more expensive point alone as
Pareto dominance. Cheap direct-policy points remain part of the frontier.

Stop condition: continue useful research until BOTH revised 10x CM targets and the
efficiency requirement are confirmed,
the user explicitly pauses/stops the work, or progress is blocked on required input
or unavailable resources. Do not claim completion for exhausting an allocation or
for a smaller gain. Before replacing an allocation, inspect jobs and cumulative
usage and coordinate the single reserved GPU with Claude. A search claim additionally
requires an incremental win over the best cheap control with a paired game-level
uncertainty estimate supporting improvement. If only masking/calibration helps,
report it as such. Preserve complete experiment and cost reports, including failures;
preliminary development results do not establish completion.

Follow-up (Yiming, 2026-09-19): after achieving this fixed-checkpoint goal, obtain
Claude's B_2 model and reproduce the winning method on it. Do not swap the current
checkpoint or move the current targets in the meantime. First evaluate the frozen
method without retuning; distinguish any later B_2-specific tuning. Establish
B_2's own raw/legal baselines, held-out CE gains and node/quality frontier, and
label any training-equivalent CM with its own reference and law assumptions.
