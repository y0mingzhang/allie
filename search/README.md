# Human-move search

The production method is **root-coverage search with a frozen Elo budget and a
calibrated policy/value mixture**. It improves human move prediction using the
model's own move, outcome and thinking-time heads. There is no Stockfish, game
retrieval, external memory or neural-weight update.

`Search.predict()` returns a probability for every legal move, its searched
value, the original legal prior, root W/D/L probabilities and actual neural-node
cost. These distributions can be served directly or used as distillation
targets. Importing the package starts no worker or job.

## What works

At each root, spend simulations across legal actions using the diminishing-return
score `sqrt(p * (1-p)) / (1 + visits)`. This covers plausible human choices instead
of putting nearly all work into finding a single best move. Below each root
action, use PUCT with `cpuct=2.5` and the model's W/D/L critic. Independent root
branches are evaluated in batches using native chess rules and a resident KV
cache.

Read the completed tree with a policy-weighted soft value backup. Its temperature
is `0.2 / sqrt(1 + (subtree_count - 1)/16)`: deeper evidence makes the backup more
selective. Unvisited actions retain the node critic; terminal outcomes are exact.
Values alternate player perspective. This backup is distinct from the averages
used to select nodes during search.

The output is a calibrated mixture of `softmax(alpha * policy_logits + beta * Q)`
policies, followed by a fitted temperature correction. Elo, known time remaining,
predicted thinking time and distribution statistics condition that calibration.
The **budget router itself uses Elo only**, choosing 128/256/512/1000 simulations.
All coefficients in [calibration.json](calibration.json) were frozen before the
transfer study; serving does not fit them. Forced moves need no child evaluation.

Imagined clock transitions use predicted thinking time, rounded and clamped to
the remaining clock, with the increment and first-move conventions used in
training. Only previously observed clocks enter the root. The third clock feature
is the mover's previous **own** thinking time, not the immediately preceding ply.

## Measured quality and cost

Final-recipe **129M checkpoint**, golden evaluation. Lower CE is better. CM below
is training-equivalent compute **relative to this checkpoint's legal policy**,
so search gets no credit for legality renormalization.

| Method | Mean new NN nodes | Macro CE | Expert macro CE | Macro CM | Expert CM |
|---|---:|---:|---:|---:|---:|
| Legal policy | 0 | 1.35437 | 1.24028 | 1.00× | 1.00× |
| Calibrated 2-ply reference | 147.0 | 1.35769 | 1.21697 | 0.93× | 1.59× |
| Calibrated 4-ply reference | 843.2 | 1.36510 | 1.20544 | 0.79× | 2.04× |
| Coverage, 128 simulations | 124.9 | 1.33986 | 1.19041 | 1.43× | 2.93× |
| Coverage, 1000 simulations | 966.5 | 1.33858 | 1.18006 | 1.48× | 3.83× |
| **Coverage, Elo-adaptive** | **461.4** | **1.33748** | **1.17985** | **1.52×** | **3.86×** |

Adaptive coverage uses **52.3% fewer neural nodes** than fixed 1000, with no
resolved CE difference: paired macro difference −0.00110
[−0.00272, +0.00053], expert −0.00021 [−0.00085, +0.00038]. That supports retaining
quality at lower cost; it does not establish strictly better quality. At matched
roughly 460-node cost, the later format/time routers did not establish gains over
Elo. These are the relevant cost comparisons, rather than comparing different
simulation limits as though they were equal work.

The same-recipe scale comparison shows diminishing **CE** gains: fixed-1000
search reduces macro/expert CE by 0.05172/0.10196 on 34M and 0.01579/0.06022 on
129M. Corresponding legal-relative CM is 2.02×/3.22× and 1.48×/3.83×. The expert
law is flatter at the larger anchor, which can increase CM despite a smaller CE
gain. Model size and training tokens both changed. The larger legal model also
beats smaller-model heavy search at about 12 versus 58 analytical GFLOPs/query,
including fresh root prefill. Nodes alone are suitable within a checkpoint;
cross-model comparisons need FLOPs or common-hardware latency.

Evidence: [full transfer report and plots](../results/search-v1/transfer-v1/REPORT.md),
[exact large-model results](../results/search-v1/transfer-v1/results-large.json),
[small-model results](../results/search-v1/transfer-v1/results-small.json), and
[time-router follow-up](TIME_ROUTER_REPORT.md).

Macro averages the 16 format × mover-Elo cells equally; expert macro averages
their four ≥2400 cells. Search CE uses paired differences on 512 positions/cell,
anchored to the same checkpoint's full canonical cell evaluation. This July
sample was reused during research; it is not a fresh confirmation. August
calibration may be training-seen, and the large-model gain there was null/slightly
worse despite the July gain. Whole-game confidence intervals exclude training
seed and scaling-law uncertainty. CM assumes the old learning-law shape transfers
after vertical anchoring at the exact training rung; it is not a measured training
saving or inference speedup. The original 10×/10× goal was not reached.

Nodes count new non-root neural evaluations, excluding rule-only terminals and
root prefill. The API reports prefill tokens separately. Historical warm timing
on the smaller exploration model was about 7 s for 512 roots ×1000 simulations;
that is not a latency promise for the 129M model or this reorganized package.

## What was not promoted

| Idea | Outcome |
|---|---|
| Repaired Allie adaptive MCTS | Beneficial in the stronger matched-cost cached test, but behind coverage: at ≈461 nodes, 1.3440/1.2066 versus coverage 1.3380/1.1814. Allie's live mixed-budget run was stopped; those cached results remain provisional. This does not rule out every adaptive-MCTS design. |
| Time-control / remaining-clock budget routing | Live ≈460-node comparisons did not resolve improvements. New Elo: 1.3381/1.1817; format: 1.3384/1.1803; clock: 1.3391/1.1812. |
| Bellman critic-consistency projection | Same-node 129M result 1.33833/1.17854; small incremental differences unresolved. Retained as explicit `project=True` at fixed 1000, off by default. |
| Simple output recalibration | Better macro but worse expert CE (1.33396/1.18470); no joint replacement. |
| Two 500-simulation search portfolios; extra depth | Portfolios lost to one 1000-simulation search; deeper expansions did not justify their cost. |
| Elo shifts, history/transposition ensembles, tactical/convergence routing, asymmetric backups, confidence-weighted consistency | No compelling confirmed improvement over the retained method. |
| Human-game retrieval | Removed by user constraint: no external memory. |

Repaired Allie is retained as an optional comparator: first expansion follows the
prior, depth-limited nodes preserve cached values/statistics, and output uses
reverse-KL regularization. `method="allie", budget=50` defaults to the original
transfer reference's `cpuct=1.25, alpha=.9, beta=2`. Changing the budget or these
parameters creates a different comparison; it does not reproduce the later
per-Elo/per-budget calibrated sweep. See the [Allie review](ALLIE_REVIEW.md).

## Run

The retained model port supports the two evaluated **dense ship-recipe** models:
board CNN, SwiGLU, no key offset, three continuous clock features. It rejects
unsupported checkpoint configurations. A future MoE checkpoint needs its own
model port and parity check. The tree/calibration layer is backend-independent.

From the repository root, install the CPU dependencies and run:

```sh
python -m pip install -r search/requirements-cpu.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s search/tests -v
```

The native loader needs a C++17 compiler with OpenMP and the existing
`vendor/chess-library/include/chess.hpp` (MIT, pinned commit
`53e6a841dcda7059a2af363d85f785ef1817304a`). Set `ALLIE_CHESS_INCLUDE` when it is
elsewhere. Builds are locked, content-addressed and cached under
`~/.cache/allie/search`; `ALLIE_SEARCH_CACHE` and `ALLIE_BOARD_CACHE` override the
caches. Nothing is compiled into results or vendor.

GPU execution retains the pinned SGLang 0.5.9 / PyTorch 2.9.1+cu128 runtime, a
single GPU and full-precision weights with BF16 inference. The original environment
and its checksummed archive are under `results/search-v1/runtime/`. Use the staged
node-local copy where available to avoid cold NFS imports. This is a direct
ModelRunner adapter, not an HTTP generation server; no tokenizer is involved.

```sh
# With the pinned runtime active; export only if a new private export is needed.
python -m search.export /path/to/last.pt /path/to/new-export

# One resident model; read one {"positions": [...]} object per input line.
python -m search results/search-v1/transfer-v1/ship-large-export \
  --budget adaptive < requests.jsonl > predictions.jsonl

# Portable full-prefix PyTorch fallback. It has no CPU latency optimization yet.
python -m search /path/to/last.pt --backend cpu --budget 128 < requests.jsonl
```

No command submits a job. An already allocated GPU must be provided for SGLang.
Keep one serial caller per `Search` instance: it owns/reset its entire cache.

The Python interface is:

```python
import numpy as np
from search import Search
from search.runtime import ShipOracle

engine = Search(ShipOracle("results/search-v1/transfer-v1/ship-large-export"))
# Starting board: 120+1 seconds; White 1800, Black 1900.
prefix = [2348, 198, 11, 1, 8, 0, 0, 1, 9, 0, 0]
query = {"prefix": prefix, "cell": 1,
         "features": np.full((len(prefix), 3), -1, dtype=np.float32)}
prediction = engine.predict([query])[0]
# prediction["tokens"] are legal move IDs; probabilities align with them.
```

Input follows `lichess_tokens_v2`: START=2348, base-time token, increment token,
four decimal White-Elo digits, four Black-Elo digits, then past move tokens
378..2345. `search.native.MOVE_ID` maps UCI moves to these tokens. No target move,
termination token or future clock is appended. Context includes 11..1024 tokens
and starts from the normal initial chess position. Terminal roots are rejected.
`cell = 4*format + Elo_band`, where format is bullet/blitz/rapid/classical (0..3)
and band is <1400/1400–2000/2000–2400/≥2400 (0..3); use the mover's rating.
Each feature row describes the player to move **after** its prefix token. Missing
features default to −1. Known clocks should be supplied to match evaluation.

For distillation, retain the entire returned distribution and its token ordering,
checkpoint/export provenance, calibration file hash, method, budget and batching.
The `values` use W−L in the root mover's perspective; `root_wdl` is W/D/L in the
same perspective. `nodes` counts actual new child NN calls; `simulations` is the
allocated tree work. `batch_prefill_tokens` is repeated metadata for the batch,
not an additive per-position charge; sum `prefill_tokens` instead. Distillation
training itself is outside this package.

## Layout, parity and retained artifacts

| Code | Responsibility |
|---|---|
| `algorithm.py`, `policy.py`, `calibration.json` | One batch API, budget routing and frozen output policy |
| `native.py`, `native/tree.cpp`, `native/value.cpp` | Rules, coverage/Allie trees, soft backup and optional critic projection |
| `model.py`, `board.py`, `native/board_encode.cpp`, `native/move-table.json` | Shared model math, board encoding and causal clocks |
| `runtime.py`, `sglang_models/allie.py` | Resident SGLang KV/state transport |
| `cpu.py` | Portable full-prefix reference backend |
| `export.py`, `configuration_allie.py`, `__main__.py` | Export and resident JSONL entry point |
| `tests/` | CPU rules, clocks, backup recursion and pre-cleanup regression fixtures |

Cleanup CPU tests compare all frozen fixed budgets, the adaptive router and
repaired Allie against pre-cleanup native/policy outputs using a deterministic
neural oracle. They also check causal board/clock transitions and the portable
model backend. The model expressions are preserved. **No new GPU parity or
latency run was performed for this cleanup.** BF16 output and search branches can
depend on batch shape/order; CPU algorithm parity does not promise bit-identical
historical GPU predictions. Keep batch size/order fixed for comparisons. The
historical large-model order audit's macro/expert changes were +0.000128/+0.000365,
with intervals including zero.

Experiment drivers, queue daemons, exploratory kernels and fit scripts were
deleted from the production tree. Their source remains at Git commit `48445c7`
and in the preserved search worktree for historical reproduction. The original
frozen trainer/evaluator copies remain untouched; production imports no live
`scripts/` module. Research reports and raw arrays were not deleted or regenerated.

Keep both existing links intact:

- `results/search-v1` → `/data/group_data/dei-group/yimingz3/allie/worktrees/search-v1/results/search-v1`
- `vendor` → `/data/group_data/dei-group/yimingz3/allie/worktrees/search-v1/vendor`

The destination worktree must remain while these links are used. Checkpoints,
exports, runtime archives, receipts and cumulative compute charges stay there.
Allie attribution is in [ALLIE_LICENSE](ALLIE_LICENSE); the vendored rules library
retains its own license. The [algorithm-family explainer](explainer/index.html)
and [search math notes](SEARCH_MATH.md) provide research context.
