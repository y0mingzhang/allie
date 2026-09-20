# Inference research

The search stage and its transfer study are complete. No GPU job or automatic
research loop remains active. The track's goal is [GOAL.md](GOAL.md); the root
repository goal belongs to the training work.

- [Results and Pareto plots](TRANSFER_REPORT.md): frozen search on the final
  34M and 129M recipes, with CE, conditional training-equivalent CM and cost.
- [Algorithm-family explainer](explainer/index.html).
- [Transfer implementation and reproduction](transfer/README.md).
- [GPU inference engine](engine/README.md): resident SGLang and native tree code.
- [Allie implementation review](ALLIE_REVIEW.md) and [search math](SEARCH_MATH.md).

`transfer/` contains the final two-checkpoint comparison. The other modules retain
the experiment implementations and checks behind the research history. Retrieval
experiments are historical rejected arms; the final methods use no external
memory, Stockfish or neural weight updates. Old experiment scripts are not a
pending work queue.

## Artifacts and runtime

Large results, checkpoint exports and runtime archives stay outside Git. On the
cluster, main's `results/search-v1` links to the original durable search results:

```
/data/group_data/dei-group/yimingz3/allie/worktrees/search-v1/results/search-v1
```

The ignored `vendor/` link similarly reuses the original worktree's pinned Allie
reference and native chess-library sources. Those dependencies and their licenses
are retained there; see [Allie attribution](ALLIE_LICENSE).

Keep these artifact and vendor directories when removing or archiving worktrees. Both checkouts
see the same STOP, queue receipts and compute ledger; importing this package does
not resume work. Historical Slurm scripts still name the original worktree and
must not be submitted without a new user instruction and peer coordination.

CPU analysis requires NumPy, SciPy and Matplotlib. Model validation requires the
frozen trainer's PyTorch runtime; serving requires the pinned SGLang runtime,
staged to node-local storage. See [transfer recovery](transfer/RECOVERY.md) and
[engine setup](engine/README.md) for exact paths and reproduction commands.

The merged code preserves the evaluated algorithms. The original 10×/10× target
was not achieved; completing this finite transfer report does not change that.

Time-aware budget follow-up (parked by user): [results and limits](time_router/REPORT.md). The earlier validated Elo-adaptive coverage method remains the recommendation.
