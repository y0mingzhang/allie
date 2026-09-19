# Cached chess inference

This is an inference-only port of the frozen `r2-3e16-control-t20-w20-pf052h-s42`
checkpoint. The source checkpoint and training runtime are read-only.

`model.py` contains portable PyTorch model math. `sglang_models/allie.py` connects
it to SGLang paged attention and compiled graphs. `direct.py` calls the model
runner in process: tree edges do not travel through HTTP, a tokenizer, a sampling
API, or the serving scheduler. `tree.py` implements human-policy continuation
expectation. `native_mcts.py` / `mcts_native.hpp` preserve the released Allie MCTS
algorithm with C++ selection, expansion and backup; `policy.py` batches its exact
FP32 output solver. `mcts.py` retains the intermediate Python-tree adapter for
differential tests. `board.cpp` wraps chess rules and our tree mechanics; it
contains no external engine evaluation or external search policy.

The architecture needs two extra cached states besides K/V: the embedding before
the previous-token smear, and the unshifted stationary key components. Both use
the same physical slots as K/V, so branches share their ancestors' state. A task
owns its cache and resets it between groups of roots. The direct runner is serial,
single-GPU, BF16, with a maximum context of 1,025 tokens. Clock/Elo side-channel
transport and quantization are not implemented.

## Runtime and recovery

Private environment: `results/search-v1/runtime/sglang-0.5.9` (Python 3.12,
SGLang 0.5.9, torch 2.9.1+cu128, pybind11 3.0.1, python-chess 1.11.2).
The original evaluator retains its separate torch 2.10 runtime.

Rules source: <https://github.com/Disservin/chess-library>, MIT,
commit `53e6a841dcda7059a2af363d85f785ef1817304a`, checked out in ignored
`vendor/chess-library`. Its license is retained there. Compile the wrapper using
the private interpreter with `python -m search.engine.build_board`.
`-march=native` is deliberately absent; eventual cheap x86 hosting remains
possible through the portable model backend. CPU deployment is deferred.

On the already allocated GPU, run `bash search/engine/workbench.sh`. It loads once
and watches `results/search-v1/engine-queue/*.request.json`. Requests execute
serially and produce `.result.json` or `.error.json`. The process lock prevents a
second runner. This command does not allocate a GPU or submit a Slurm job.

Example tree request (use absolute paths inside our private results directory):

```json
{
  "kind": "tree",
  "input": "/path/to/results/search-v1/dev.json",
  "output": "/path/to/results/search-v1/fast-deeper-pilot",
  "widths": [4, 2, 2],
  "batch_size": 1024,
  "roots_per_batch": 32
}
```

Input is `{"positions": [...]}` with the existing prefix, legal moves, game and
ply identifiers. Each output block is atomic; completed blocks can be resumed
only under the same input, implementation, checkpoint and search plan. Both the
private and controller STOP files stop further task execution. An error is never
silently retried. Inspect its traceback before reissuing a request.

## Correctness and numerical limits

- `test_board`: 18,613 positions, including castling, en passant, promotion,
  insufficient material, 75-move and fivefold draws. Legal move **order** and
  automatic outcomes match python-chess. Claimable draws are not automatic.
- `test_cache`: FP64 differential checks for chunked prefill, interior-prefix
  branches, continuation and reused cache slots; tolerance 1e-12.
- `test_tree`: matches the earlier tree at every horizon with deterministic
  predictions, two batch sizes and a forced mate.
- `test_mcts`: identical paths, visits, values, budgets and policies to the
  audited Allie adapter under deterministic predictions.
- `validate_mcts_gpu`: real-logit replay across 128 trees; fixed and adaptive
  paths/visits match exactly, with zero output-policy difference in the test.
- `test_policy`: 4,096 random/tied/zero-budget cases match the released per-tree
  FP32 policy solver bit-for-bit.
- `check_math`: eager portable model matches the frozen source using the same
  dense kernel exactly on the tested inputs. The actual compiled serving kernels
  have different BF16 rounding. They are **not bit-identical** to the evaluator.

On 2,048 development positions, serving versus original legal-policy CE changes
by -0.00003 nat (paired game-bootstrap 95% interval [-0.00197, +0.00193]); top-move
agreement is 98.29%. Cached versus fresh prefill changes CE by +0.00058
([-0.00090, +0.00222]); agreement is 99.12%. Mean absolute WDL expectation changes
are 0.00935 versus the original and 0.00152 cached versus fresh. Use the matching
engine baseline for search gains and disclose conversion drift separately.

Full details: `results/search-v1/engine-queue/001-benchmark.result.json`.
On the reserved RTX6000Ada, warmed four-ply search measures roughly 35–44 root
positions/s on the tested 16/32/64-root batches, including rules, tree assembly,
GPU forward and score transfer. Model-runner initialization takes 18–25 s with
cached compilation, excluding Python imports. The resident service amortizes
startup. One cold shared-filesystem restart took several minutes of dependency
and compilation-cache I/O. `stage_runtime.py` stages the same environment on local
NVMe; its durable source stays under results/. The command can resume an interrupted
copy and publishes STAGED.json only after all workers complete. `workbench.sh`
accepts SEARCH_ENGINE_RUNTIME to use that cache. Cold-start timing for this staged
path still needs verification. The full 2,048-root four-ply task took 58.6 s warm,
including atomic result writing; this is a measured task, not a throughput extrapolation.

Equal-node cost is in engine-queue/004-equal-nodes.result.json: about 62k evaluated
leaves take 1.56–2.02 s for four-ply, and 2.01–2.04 s for C++ MCTS batched over
1,024 roots, plus 0.066 s MCTS root prefill. The former uses 64 roots with wider
trees. This isolates throughput, not prediction quality or equal effort per root.
`forward_seconds` measures wall time inside the model runner, including metadata,
host dispatch and synchronization; it is not a CUDA-kernel-only measurement.

The HTTP prototype (`serve.sh`, `client.py`, `transport.py`) is retained for
comparison and integration work; the in-process runner is the research fast path.
The binary HTTP hook is specific to pinned SGLang 0.5.9 and is not used by it.
