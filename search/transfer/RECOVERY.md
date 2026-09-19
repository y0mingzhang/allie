# Transfer recovery

Status: COMPLETE, 2026-09-19. Do not submit or restart experiments. The old
10×/10× research goal is closed unmet; its paused API goal is not this task.

Read `search/TRANSFER_REPORT.md` for the summary and
`results/search-v1/transfer-v1/REPORT.md` for tables, uncertainty, methods and
PNG/PDF/SVG plots. Machine reports retain all methods and per-position scores.
The original search stage remains in `results/search-v1/STAGE_REPORT.md`.

Only job 10505672 was used for this transfer. It had three allocations:
1,397 + 323 seconds on preempt RTX PRO 6000, then 1,530 seconds on general L40S
after peer coordination. Every queue request 002–015, including 010a, completed.
Request 001 retains its expected failure before the CPU-math gate; 002 passed it.
All GPU work finished before the final automatic requeue, which was cancelled.
Own `results/search-v1/STOP` is present. The accounting observer exited.
`accounting.json` records 0.902778 GPU-hours for this task and 15.727778 cumulative.
No peer checkpoint, job, corpus or main-worktree file was changed.

Primary small-model predictions use RTX PRO 6000; large golden predictions use
L40S only. Incomplete large RTX blocks are preserved in `interrupted-rtx/` and
excluded. Cross-scale wall times are not common-hardware comparisons. The
large clock sensitivity crosses hardware types and is explicitly confounded.

Validation: independent CPU FP32 source equality for both models; full canonical
golden evaluations; 32,669 board and 30,653 clock transitions checked; baseline
algorithm parity; cached/full-prefix BF16 checks; 1,024-position order audits.
The final L40S audit gives macro/expert order deltas +0.000128/+0.000365, both
intervals including zero. Numerical branch sensitivity is disclosed.

Primary CM uses the source-law rung labels 3e16/3e17. `/6` coordinates remain
reported sensitivity, not the primary result. No fitted law for the improved
recipe is claimed. Golden estimates use the existing 8,192 positions and full
canonical cell anchors, with whole-game bootstrap; this is not a fresh
independent confirmation. Test/test_expert remain unopened.

CPU-only reproduction from this worktree:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m search.transfer.analyze golden small
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m search.transfer.analyze golden large
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m search.transfer.report
```

Do not run `workbench.sbatch` unless the user authorizes new GPU work. GPU/source
tests require the node-local pinned runtimes; the controller's old torch cannot
import the frozen trainer. Source and checkpoint hashes are in the frozen plan,
exports and canonical reports. Current-game clocks/boards are retained; no
external memory, Stockfish or neural weight updates are used.
