# Model-only inference research

Owner: Codex. Worktree: `codex/search-v1`, independent of Claude's main worktree.
Status: active, authorized by Yiming on 2026-09-18.

Improve human move prediction through inference-time methods using the model itself,
starting with `r2-3e16-control-t20-w20-pf052h-s42`. No Stockfish or external value model.
The small model is deliberate: optimize experiment speed and build reusable infrastructure.

Provisional success: at least 0.01 nat lower expert macro CE than the same checkpoint's
raw direct policy, with no clear overall macro CE regression, confirmed after method
selection on separate development games. Report direct-policy calibration and legal-mask
baselines separately, so their gains are not misattributed to search. Also report expert
move accuracy, all 16 cell losses, calibration, and measured inference cost.

Evaluation uses Claude's 16 format × mover-rating cells and four-cell expert macro.
Do not tune against strat-eval-v1. Use the existing prepared dev/dev_expert splits for selection; split development
confirmation by game, not move. No replacement stratified evaluation dataset.
Freeze method before golden evaluation. Never use the human next move, future moves,
actual thinking time or actual game outcome when selecting candidates or allocating search.

Resources: user explicitly permits one reasonably fast GPU and infrastructure work.
Latest user instruction: maintain one long-running GPU session for fast iteration.
Start with an eight-hour allocation; record all reserved GPU time, including idle time.
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

Recovery: read this file, search/PLAN.md and results/search-v1/status.json. Inspect own
job receipts and live Slurm state before submission; never duplicate a job. Respect
/data/group_data/dei-group/yimingz3/allie/controller/STOP. A controller restart does not
resume paused/completed work. Claude owns the main worktree's data/model goals.
