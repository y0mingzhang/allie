# Search worktree recovery

Codex owns this worktree (`codex/search-v1`), its GOAL.md, search/ and private
results/search-v1/. Claude owns main and training/data/model research. Read
GOAL.md and /data/group_data/dei-group/yimingz3/allie/controller/STOP first.
Never submit training, stop the tunnel/controller, or change Claude's jobs.
The UI goal is stale/paused; do not reset it. Our authorized goal is in GOAL.md.

## Current state: 2026-09-19 ~05:04 UTC

Goal remains active: BOTH >=2x golden macro and >=10x expert training-equivalent
CM, with incremental search improvement beyond cheap controls. Not achieved.
User wants a fast resident inference workbench, then sustained creative research.
One fast GPU is allowed; it need not be L40S. Keep one allocation at a time.

Current GPU job10498670: general/normal, 1 L40S, babel-t5-32, 8h from04:52UTC.
Socket search-v1-10498670, session oracle, direct engine PID1046638. Inspect live
Slurm/processes before any submission. Previous10497511 was PREEMPTED at04:48:30
(9899seconds=2.74972GPUh); pending replacement10498656 was cancelled at0 GPUh.
All old engines/reference oracle died with10497511. Do not restart them.
Current engine imports are slow from cold shared storage (~10min at this update).
Its log is logs/engine-10498670.log. ready.json is stale until its job is10498670.
The GPU is not failing: Python is blocked on cold library/import I/O.
A library-prefetch helper completed4.36GB in92.6s; it did not eliminate all imports.

Private paths below are relative to results/search-v1/ unless stated otherwise.
- engine-queue/011-l40s-recovery.request.json: existing2048dev positions, validates
  new hardware port/caching drift and four-ply timing. No method tuning.
- engine-queue/012-balanced-golden.request.json: first frozen golden comparison.
  It checks011 parity before scoring. Atomic immutable blocks resume automatically.
- tmux window final-report: search/engine/finish_balanced.sh waits for012, runs
  analyzer and report on this independent GPU allocation; survives controller death.
- Accounting: search/advance.py --watch (PID in accounting.pid), singleton flock;
  restart after controller reboot only if dead. sacct-D per-incarnation charges
  and job-receipts/ preserve all usage. Main compute.json remains untouched.
- Runtime archive builder: bash search/engine/pack_runtime.sh, controller process;
  writes runtime/sglang-0.5.9-torch2.9.1-cu128.tar.zst plus.sha256 atomically.
  Restart if controller reboots before completion. This packs the pinned environment
  for node-local staging without thousands of network metadata accesses. A completed
  archive is used automatically by search/engine/workbench.sh on later cold boots.
  Do not interrupt an active scoring task to switch runtimes.

Claude plans to requeue controller10478640 with5min warning. Our Slurm workbench
and its tmux/tasks survive. Controller-local archive/accounting processes do not.

## Frozen golden confirmation

8192 existing scored moves (512/cell,6752 games), seed1926731; same strat-eval-v1
mask/labels/headers, no replacement dataset. sample.json is immutable; never reroll.
plan SHA68c73ab4a7bcb132c7b749d49653b7b82aa97edcc526a1e6393a76d3b6620496
sample SHA6e3bfafbdceaf473eb070dd1a426aefadf0e11dc314d2bac2aeb07728b6f59ff
Directory golden-balanced-v1 contains plan.json/sample.json/direct-control.json/
execution.json, with checkpoint, source/binary, export and dataset provenance.
All method choices were fixed before new golden scoring: legal, dev-fit temperature,
calibrated2ply, calibrated4ply, releasedadaptive50, repairedfixed50+reverseKL.
No choosing among them or tuning parameters on golden. Report every method.

Estimator: exact full canonical-raw cell means + paired sample(method-raw), then
unweighted16/4cell macros. Whole-game bootstrap is shared across methods/cells;
report port drift, all cell deltas, accuracy, calibration, timing and law sensitivity.
CM = frozen vertically anchored training-law equivalence, not inference speedup.
Cheap/legal gains are separate. Test/test_expert remain unopened. This checkpoint
has no clock/Elo/continuous side inputs; strength/game headers remain unchanged.

CPU analyzer: OPENBLAS_NUM_THREADS=2 <runtime>/bin/python -B -m search.engine.analyze_balanced
Report: <python> -B -m search.report
Analyzer's equal-cell/paired/game-bootstrap invariants pass test_balanced.
Queue errors must be diagnosed explicitly; never delete completed scores to retune.

## Completed work

009-nvme-final: real-model MCTS replay exact paths/visits/output; ~62k leaves takes
1.69s four-ply vs1.81s MCTS512roots,1.92s MCTS1024roots. These are throughput checks,
not quality comparisons. Warm model-runner wall time is ~.5-.7s; remainder includes
CPU tree/plumbing/cleanup. Local cached-runtime cold process startup30.7s on old node.
Port BF16 is not bit-identical; existing2048dev mean CE drift near zero, KL~.001.

010-adaptive-repairs completed8 dev variants in69.7s. Practical first-visit/depth
repairs help slightly; predicted-time allocation did not beat fixed allocation.
Calibrated repaired fixed expert devCE1.4775 vslegal1.4947. Four-ply's earlier small
confirmation expertCE1.4157 is promising, not a golden metric. Dev CM stays pending.
Full cumulative tables are REPORT.md, generated by search/report.py. After EACH
experiment show user important loss/CM results. No new images unless requested.

Our ALLIE_REVIEW.md addresses the uploaded critique: distinguish lambda algebra,
KL direction, practical bugs and empirical historical evidence; no theory alone
establishes human-move CE gains or proves reported paper runs were unaffected.

Claude focused tier1 review is DONE. Inspected fixes for actual pooled-control B2 SHA,
continuation provenance rejection and board meta matmul MACs. Exact same-run resumes
are allowed. Launch still requires its real smoke5 identity/resume pass. MoE/diffattn
remain a separate unreviewed delta. No main files were edited by us.

After012:013-packed-plumbing is an experimental CPU/batching fast path, with exact
logit/slot-map parity followed by repeated timings; it always restores the old runner.
It does not alter the frozen golden computation.014-mcts1000 and015-depth6 are
subsequent DEV-ONLY research, preregistered before opening first golden results:
fixed/repaired MCTS1000 (128 roots/batch), six-ply continuation widths4,2,2,2,2
(16 roots/batch). Analyze on original fit/confirmation game folds; never apply
these dev losses to golden CM laws. These research tasks use only our existing GPU.
Runtime archive builder now uses bounded parallel small-file reads (pack_runtime.py),
excluding packaged tests and pycache; source code/libraries/metadata stay unchanged.

05:13UTC startup amendment: oldPID1046638 was still in cold NFS imports after20min,
not scoring. Replaced ONLY our engine pane; currentPID1097060. direct.py now executes
the exact hash-checked SGLang setup function through startup_env.py, avoiding the
full scheduler/tokenizer/multimodal import dependency. No model/search math changed.
Golden execution was explicitly re-frozen before ANY method scores; original preserved
as execution.before-startup-opt.json with startup-amendment.json. Plan/sample hashes
unchanged. Queue011 parity is still mandatory. All requests restored; no newSlurmjob.
Dev analyzers run in independent tmux window dev-report after014/015, then updateREPORT.
Next experiment rationale is search/NEXT.md. Own source commits must survive restart.

Controller requeue is imminent (~05:25UTC). Commit everything; all GPU tasks survive.
Runtime archive packs ordinary packages but represents flashinfer_cubin/cubins as a
symlink to its unchanged durable source (tens of thousands of unused architecture-
specific binaries caused slow packing). All binaries remain available; local archive
contents and linkage are checksummed. This made packing much faster.
GPU tmux window runtime-cache runs recover_cache.sh: waits for the archive checksum,
stages locally, then replaces ONLY a still-not-ready engine. If the engine is already
ready, it leaves it alone. Never interrupt scoring for a runtime switch. The archive
builder remains controller-local and needs restart after reboot if incomplete.

GPU tmux runtime-builder now runs ensure_archive.sh. It takes over the archive build
if the controller-local builder dies, using the same durable flock (no double writer).
Thus archive construction, local staging, cold-engine recovery, all experiments and
analysis can progress even during the controller restart. Check these tmux windows
before restarting ANY helper manually. PID accounting is the sole remaining
controller-local task that requires restart after controller migration.
