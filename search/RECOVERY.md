# Search worktree recovery

Codex owns this worktree (`codex/search-v1`), its GOAL.md, search/ and private
results/search-v1/. Claude owns main and training/data/model research. Do not
submit training or change main. Read GOAL.md and the controller STOP file first.

## Active work, 2026-09-19 UTC

User priority: make MCTS and four-ply achieve comparable throughput per evaluated
node with enough independent positions to batch. Then resume the >=2x macro /
>=10x expert training-equivalent CM research goal. Do not declare success merely
for finishing infrastructure. All search golden-method scores remain unopened.

One allocated GPU: Slurm 10497511, preempt, RTX6000Ada on babel-x9-24, through about
10:03 UTC. Inspect squeue and results/search-v1/status.json before any submission.
Never duplicate it. Never stop tunnel/controller or Claude jobs. All reserved time
including idle is charged by search/advance.py --watch. This is our separate ledger.

Tmux socket search-v1-10497511, session oracle. Original reference evaluator stays
in window 0; server-ready.json describes it. Never print rpc-token. It uses the
original staged torch2.10 runtime. The new engine uses a private SGLang0.5.9 /
torch2.9.1 runtime and is in window fast-engine. Queue/ready state lives in
results/search-v1/engine-queue; check process liveness because ready.json may be
stale during startup. Use the existing allocation only.

- 001-benchmark.result.json: 2048-root port/caching numerical drift and four-ply
  timing. Serving BF16 is not bit-identical to the reference; CE drift is small.
- 002-mcts.result.json: Python/native-board MCTS benchmark, before C++ tree logic.
- 003-deeper.result.json: complete fast 2048-position four-ply pilot, 58.6s total.
  fast-deeper-pilot/results.json uses coefficients frozen before the port.
- 004-equal-nodes.request.json: native C++ MCTS equal-node benchmark, pending/running.
  It compares ~64k leaves, 128/512/1024 independent MCTS trees vs four-ply. Do not
  duplicate a producer; inspect .result.json / .error.json and engine-service.log.

The native tree is in search/engine/mcts_native.hpp. test_mcts passes identical
paths/budgets/visits and policies within1e-7 against the original Python reference
on deterministic predictions. Other tests cover 18,613 rules positions, FP64 cache
branching/reuse, and four-ply equivalence. The board extension is built atomically
so a resident process never maps a truncated library. Restart the engine after a
native library change. Existing completed experiments remain immutable.

One restart encountered minutes of NFS metadata waits while importing unrelated
Transformers models. A runtime tar archive is being prepared under our private
runtime directory for staging to /scratch; the group copy stays authoritative.
workbench.sh accepts SEARCH_ENGINE_RUNTIME for the scratch copy. direct.py now
sets SGLANG_DISABLED_MODEL_ARCHS to skip unused built-in registry models. These
startup changes have not yet been benchmarked. No additional GPU was requested.

The resident engine accepts coarse experiment requests; all tree edges stay in
process. See search/engine/README.md. Completed blocks are atomic and immutable
under their recorded input/checkpoint/source hashes. Both STOP files prevent new
tasks. Filesystem request kinds include benchmark, mcts_benchmark, tree, and
experiment (a module under search.engine exposing run(oracle,spec)).

## Science state and next evaluation

The original 64,366-position dev confirmation is complete for released MCTS and
calibrated two-ply. Four-ply is promising only on the reused 1050-position dev
confirmation; port verification agrees. Do not label dev losses as macro metrics
or use golden scaling laws to convert them to CM.

Exact golden raw/legal baselines and root caches are complete in golden-baseline/.
The old golden.py expert-only full-eval plan has NEVER been frozen/launched and is
superseded by the user's allowance for a smaller balanced existing-eval sample.
Proposed first comparison: 512 uniformly selected existing scored moves per cell,
fixed seed independent of scores, all16cells, methods frozen from dev. Report all
2ply/4ply/released adaptive methods, no golden-based selection. Keep test shut.

Claude agreed the difference estimator is sound: for each cell use its exact full
canonical raw baseline + paired sample (method - canonical raw), then macro-average.
Bootstrap whole games and transform paired uncertainty through the frozen law.
Also report ordinary sample CE, all16cells, raw->port->legal->search changes and
inference cost. Never drop skipped positions; declared fallback remains scored.
Clock inputs are absent in this checkpoint. No golden sample has yet been created.

After each experiment update search/report.py's output results/search-v1/REPORT.md
and show the user the important loss/CM table (CM pending for unmatched dev metrics).
No images unless requested. Keep model and original data read-only.

Claude also requested a focused pre-sweep review of model-screen common paths.
Already phoned findings: board optimizer group breaks copy_lm_to_embed's last-group
assumption; global BOARD leaks across constructions; B2 stored SHA needs verification;
within-track hashes need driver/evaluator coverage. Wait for the frozen snapshot
before explicit done-review; cross-track controls must compare executed configs,
source/evaluator hashes and B2 SHA rather than requiring identical drivers. No main
files were edited. User's MCTS optimization retains priority.
