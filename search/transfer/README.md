# Scaled-checkpoint transfer

Finite follow-up to the closed inference research stage: reproduce frozen search
on the shipped 34M and 129M checkpoints, then compare quality versus search cost.
This package does not train a model or modify the main worktree.

Results live in `results/search-v1/transfer-v1/`. `plan.json` fixes methods and
sampling; `cm-convention-addendum.json` corrects the reporting coordinate without
changing any predictions. `REPORT.md`, generated after every comparison completes,
is the readable artifact. Raw reports and per-position scores remain alongside it.

## Implementation

- `model.py`, `board.py`, `context.py`: portable frozen math and causal side inputs.
- `oracle.py`, `handles.py`, `sglang_models/`: resident SGLang, query-owned KV and
  board/clock state. No cross-position value or tree cache.
- `canonical.py`, `check_math.py`, `parity.py`: original packed evaluation,
  independent source-math equality, and cached/full-prefix comparisons.
- `collect.py`: shared fixed-budget trees, frozen adaptive allocation and critic
  correction. Completed blocks are immutable and resumable.
- `baselines.py`, `allie_handles.cpp`: fixed-ply and original/repaired Allie
  transports, verified against their original algorithms by `test_baselines.py`.
- `analyze.py`: August-only calibration and whole-game paired golden estimates.
- `cost.py`, `report.py`: node frontiers, analytical inference FLOPs, uncertainty,
  scale comparisons and exportable PNG/PDF/SVG plots.

## Runtime and recovery

Read `RECOVERY.md`, the worktree's `GOAL.md`, both STOP files and live Slurm state.
Never submit a second worker while the current one is running or requeued.
Training/checkpoints in Claude's main worktree are read-only.

Frozen-source tests require the staged PyTorch2.10 runtime; the controller's old
PyTorch cannot import that source. Serving uses the checksummed SGLang runtime
archive staged locally by `workbench.sbatch`. CPU analysis needs NumPy/SciPy and
Matplotlib and can run on the controller:

```sh
OPENBLAS_NUM_THREADS=2 python -m search.transfer.analyze golden small
OPENBLAS_NUM_THREADS=2 python -m search.transfer.analyze golden large
OPENBLAS_NUM_THREADS=2 python -m search.transfer.report
```

GPU requests are JSON files under `transfer-v1/queue/`; `service.py` runs one
resident model and switches processes between models. Every request gets an
immutable result/error receipt. Retry failed work using a new request rather than
erasing its receipt. Source/config hashes prevent silently resuming changed work.
The independent `search.advance --watch` observer charges every Slurm incarnation,
including startup and idle time, and never submits a job.

CM is conditional on transferring an old learning-law shape. Primary CE estimates
retain all sixteen golden cells and all sampled positions. Current-game clocks,
headers and chess rules are allowed; external memory, game retrieval, engines,
future observed clocks/outcomes and neural-weight updates are not used.
