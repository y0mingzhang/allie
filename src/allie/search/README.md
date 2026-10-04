# Search

`allie.search` runs a tree search over the model's own predictions of the move, the game outcome and the thinking
time, and returns a distribution over the legal moves. It predicts the human move better than the raw policy, most
of all for strong players ([results](../../../docs/DETAILS.md#search)). It uses no engine and no game database.

At the root, simulations are spread over the legal moves by `sqrt(p(1-p)) / (1 + visits)`, which covers the
plausible human choices instead of hunting for the single best move. Below each root move it runs PUCT
(`cpuct = 2.5`) with the model's win/draw/loss head as the value. Values come back up by a soft backup whose
temperature, `0.2 / sqrt(1 + (subtree size - 1) / 16)`, sharpens as a subtree grows. The output is a calibrated
mixture of `softmax(alpha * logits + beta * Q)` with a fitted temperature. Search moves advance the clocks by the
model's predicted thinking time. The default budget is picked from the mover's rating: 128, 256, 512 or 1000
simulations.

## Install

```sh
uv sync --extra search
```

The native rules and trees (`native/*.cpp`) compile on first use. They need a C++17 compiler with OpenMP. The
header-only [chess-library](https://github.com/Disservin/chess-library) (MIT, commit
`53e6a841dcda7059a2af363d85f785ef1817304a`) ships in `native/chess-library/`; `ALLIE_CHESS_INCLUDE` points the
build at another directory holding `chess.hpp`. Builds are cached by content under
`~/.cache/allie/search` and `~/.cache/allie/board-encoder`; `ALLIE_SEARCH_CACHE` and `ALLIE_BOARD_CACHE` move them.

## Run

`MoEOracle` serves a mixture-of-experts training checkpoint such as Allie 2.0's on one CUDA GPU, built by the
checkpoint's own training code (a checkpoint from before the package layout takes `source=`, its run's frozen
source). The released Hugging Face weights run search through the Lichess bot instead (`play.search`, see
[allie.lichess](../lichess/README.md)).

```python
import json
from importlib.resources import files

import numpy as np
from allie.search import Search
from allie.search.moe_oracle import MoEOracle

calibration = json.loads(files("allie.lichess").joinpath("calibration-allie-2.0.json").read_text())
engine = Search(MoEOracle("path/to/model.pt"), threads=8, calibration=calibration)
# start position, 120+1 seconds, White 1800, Black 1900
prefix = [2348, 198, 11, 1, 8, 0, 0, 1, 9, 0, 0]
query = {"prefix": prefix, "cell": 1, "features": np.full((len(prefix), 3), -1, np.float32)}
prediction = engine.predict([query], budget=128)[0]
```

`predict(queries, method="coverage", budget="adaptive", clock_rule="predicted", project=False)`:

- `method`: `coverage` (the search), `legal` (the raw policy over legal moves, no search), or `allie` (the original
  Allie's MCTS at a fixed `budget` in 0..1000, for comparison).
- `budget`: `adaptive` (by rating) or a fixed 64, 128, 256, 512 or 1000 simulations.
- `clock_rule`: `predicted` spends the model's predicted thinking time at each search move; `zero` spends none.
- `project=True` adds a critic-consistency projection of the tree values, at fixed budget 1000 only.

One caller per `Search`: each batch resets the oracle's cache.

## Input

Each query is a dict:

- `prefix`: the game so far in `lichess_tokens_v2`: START (2348), the base-time token, the increment token, four
  digits of White's rating, four of Black's, then the moves played (tokens 378..2345; `allie.search.native.MOVE_ID`
  maps UCI to tokens). 11 to 1024 tokens, from the standard start, not a finished game. No target move.
- `features` (optional): float array `(len(prefix), 3)`. Row `i` describes the player to move after token `i`:
  their clock in seconds, the opponent's clock, and their own previous thinking time. `-1` marks unknown; supply
  known clocks.
- `cell`: `4 * format + band`, format bullet/blitz/rapid/classical = 0..3, band of the mover's rating
  <1400 / 1400-2000 / 2000-2400 / >=2400 = 0..3.

## Output

One dict per query:

| key | |
|---|---|
| `tokens` | legal move tokens |
| `probabilities` | predicted probability of each, aligned with `tokens` |
| `legal_prior` | the raw policy over the legal moves |
| `values` | searched value of each move, win minus loss for the player to move |
| `root_wdl` | the model's win/draw/loss for the player to move |
| `nodes` | new model evaluations spent on this query |
| `simulations` | simulations allocated |
| `prefill_tokens` | prefix tokens evaluated for the root; `batch_prefill_tokens` is the whole batch's |
| `method` | the method used |

## Calibration

[`calibration.json`](calibration.json) holds every fitted constant: `cpuct`, the backup temperature (`backup`), the
rating router (`adaptive`), the output policy per budget (`budget_policies`: per-band `alpha` and `beta`, a gate on
position features, a temperature) and the policies of fixed 64 and 1000 simulations and of the projection
(`old_parameters`). They were fitted on a dense model. `Search(oracle, calibration=...)` takes another fit of the
same form: Allie 2.0's, refit on held-out games, is
[`allie/lichess/calibration-allie-2.0.json`](../lichess/calibration-allie-2.0.json). The default file makes Allie 2.0
worse at every budget. `analysis/search-bench/fit.py` refits it for a model.

## Capacity

`Search` takes at most 128 positions per batch. `MoEOracle` holds a batch's tree in a key-value pool: the batch
needs sum(prefix lengths) + positions x simulations <= `slots` (2^18) and positions x (simulations + 1) <= `rows`
(2^17), and raises when full. Larger batches (set `engine.batch_size`) need larger pools. BF16 results can depend
on batch size and order, so keep both fixed when comparing.

## Parity

`MoEOracle` restates the training forward exactly; only kernels differ (SDPA over the tree's keys instead of flex
attention, eager instead of compiled). On a 1e18-FLOP MoE from the scaling sweep, its move distribution differs from the compiled
evaluator's by a mean KL of 1.3e-3 nats with 97% top-1 agreement, as much as the eager training forward does. A
tree node's prediction matches a fresh prefill of its path to a mean KL of 6e-4. `python -m allie.search.moe_parity
--checkpoint CKPT` reruns the checks.

## Other models

An oracle implements `reset()`, `new_tokens` and `handles(prefixes, features, clock_rule)`, which returns an object
with `root_logits`, `queries`, `per_root_queries` and a call that takes `(node, parent, token, length)` rows and
returns their logits. `cpu.CPUOracle` is a plain reference on the dense port in `model.py`.

## Tests

```sh
uv sync --extra search --extra test
ALLIE_CHESS_INCLUDE=/path/to/chess-library/include OMP_NUM_THREADS=1 uv run pytest tests/search
```

They check the rules against python-chess, the clock and board transitions, and the value backup, and pin the
search's outputs bitwise with a deterministic stand-in model (`tests/search/migration.json`). Without pybind11 or
`chess.hpp` they are skipped.

The search derives from the original [Allie](https://github.com/ippolito-cmu/allie) and keeps its MIT notice in
[ALLIE_LICENSE](ALLIE_LICENSE).
