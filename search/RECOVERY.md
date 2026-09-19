# Recovery and ownership

This branch owns GOAL.md and search/*. Main-worktree training/data research belongs to Claude.
All pilot state is in results/search-v1 under this worktree's durable path. No symlink to the
main results directory is used. The only external checkpoint inputs are read-only.

1. Read GOAL.md and results/search-v1/status.json. Check controller/STOP.
2. Inspect recorded job IDs with squeue/sacct before any submission. Do not resubmit a
   running/pending job. The first pilot is 10497506, one preempt L40S, at most two hours.
3. If the pilot finishes successfully, run:
   /home/yimingz3/src/allie/.venv/bin/python -B search/analyze.py
4. If preempted or failed, read logs and account its GPU time before deciding on a retry.
   Atomic cache-*.npz batches are reusable only with the same cache-identity.json.
   Identity binds checkpoint, input data, oracle code and top-k/packing settings.
5. Never modify the dev confirmation selection after viewing its result. selected.json
   records the fit-only choice. More research requires a clearly identified new round
   and independent confirmation; do not silently reuse the final golden set for tuning.
6. Final evaluation uses the existing strat-eval-v1 rows and labels, not a rebuilt sample.
   Preserve its per-cell mean and equal-cell macro weighting. Development is mostly blitz
   and cannot alone establish gains on classical or rapid experts.
7. The first cache does not implement MCTS. It measures a one-ply model-value correction.
   Deeper search is conditional on evidence that values improve over cheap controls.

The model oracle currently recomputes packed prefixes. It batches candidates and caches
all raw predictions to make CPU policy sweeps cheap. KV caching or a specialized leaf-only
head can be added if measured throughput says they are worth the implementation cost.
