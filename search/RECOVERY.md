# Recovery and ownership

This branch owns GOAL.md and search/*. Main-worktree training/data research belongs to Claude.
All pilot state is in results/search-v1 under this worktree's durable path. No symlink to the
main results directory is used. The only external checkpoint inputs are read-only.

1. Read GOAL.md and results/search-v1/status.json. Check controller/STOP.
2. Inspect recorded job IDs with squeue/sacct before any submission. Do not resubmit a
   running/pending job. Persistent workbench: 10497511, one modern preempt GPU, eight hours.
   The earlier pending pilot 10497506 was cancelled with zero GPU time.
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

Persistent session: results/search-v1/allocation.txt contains job, host, tmux socket,
and the staged Python interpreter. The socket is unique to this job and unrelated
 to the user's controller/tunnel. SSH to the allocated host and attach with:
  tmux -L search-v1-10497511 attach -t oracle
The worker leaves an interactive shell available if inference fails. Restart its
oracle with SEARCH_PYTHON from allocation.txt and the command in worker.sh.

When server-ready.json exists, send requests without reloading the model:
  /home/yimingz3/src/allie/.venv/bin/python -B search/request.py ping
  /home/yimingz3/src/allie/.venv/bin/python -B search/request.py score --input <json> --output <npz>
Inputs have {"prefixes": [[tokens...], ...]}; paths must be inside results/search-v1.
Queue requests and outputs are durable. Completed requests are not retried; requests
interrupted by server restart are recovered. Check .done.json/.error.txt before retrying.
The server will retain its GPU while idle, and all allocation time is charged.
Writing results/search-v1/STOP releases only this workbench, not other Slurm jobs.

Fast transport: server-ready.json includes the allocated host's HTTP URL. The client
in confirm.py sends authenticated JSON prefixes and receives NumPy logits directly,
optionally only requested columns. The token stays in results/search-v1/rpc-token
(mode0600). Do not print or commit it. The filesystem request queue remains a recovery
fallback. Use `bash search/worker.sh --serve-only` to restart the service without
regenerating the completed pilot: it checks semantic identity and exact cached-root
parity before exposing the server.

Expanded confirmation command:
  /home/yimingz3/src/allie/.venv/bin/python -B search/confirm.py
It resumes completed batches and holds the selected parameters fixed. A running
confirmation producer must not be duplicated. Check the process table first.
