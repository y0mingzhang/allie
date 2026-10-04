# Allie on Lichess

`allie.lichess` runs Allie 2.0 as a [Lichess bot](https://lichess.org/api#tag/Bot), and predicts moves from
Python or the command line. It needs no GPU: one CPU process holds the model once and plays several games at
a time.

- **Model.** A plain-PyTorch port of the trained network (`model.py`, the reference), run by default through
  `fast.py`: C++ kernels on CPU, compiled for your machine on first use, and CUDA graphs on GPU. Each game keeps
  its own key-value cache, so a move costs one or two new tokens. Concurrent games are batched into one step.
- **Inputs as in training.** The game header carries both ratings and the time control. Each move carries the
  board, both clocks and the mover's previous think time, computed as the training data computes them.
- **Client.** The Bot API over the standard library (`client.py`): challenges, game streams, moves, draws,
  resignation, reconnects and rate limits.

## Setup

1. **Install.** `uv sync --extra bot` (or `uv tool install "allie[bot] @ git+https://github.com/y0mingzhang/allie"`).
   Add `--extra search` for the `strongest` mode's search.
2. **Weights.** By default the bot downloads [`yimingzhang/allie-2.0`](https://huggingface.co/yimingzhang/allie-2.0)
   from Hugging Face on first start (11 GB). To serve a training checkpoint instead, export it once and set
   `model` to the directory:
   ```sh
   allie-bot export results/pretrain/allie-2.0/last.pt exports/allie-2.0
   ```
3. **Create a bot account.** Bot accounts must be new accounts that have never played a game. Create one on
   lichess.org, then make a personal API token with the `bot:play` scope at
   <https://lichess.org/account/oauth/token/create?scopes[]=bot:play>. Upgrade the account once:
   ```sh
   curl -d '' https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer $LICHESS_TOKEN"
   ```
4. **Configure.** Copy [configs/lichess-bot.toml](../../../configs/lichess-bot.toml). The token is never stored in
   the config: put it in the `LICHESS_TOKEN` environment variable.
5. **Run.**
   ```sh
   export LICHESS_TOKEN=lip_...
   allie-bot play --config my-bot.toml
   ```
   Any config key can be overridden with `--set`, for example `--set play.rating=1800`. With
   `--drain-file PATH`, creating that file (or sending SIGUSR1) makes the bot decline new challenges, finish
   its games and exit.

## Predicting moves without the bot

```python
from allie.lichess.api import Allie

allie = Allie.from_pretrained()  # yimingzhang/allie-2.0; int8 on CPU, BF16 on GPU
allie.predict("1. e4 e5 2. Nf3", white_elo=1800, black_elo=1750, time_control="180+2")
allie.analyze(["e2e4", "e7e5"], 1500, 1500, "60+0", clocks=[60, 59])  # also win/draw/loss and think time
allie.play("1. e4 e5 2. Nf3", elo=1200)  # a move sampled as a 1200 player would play it
```

On the command line: `allie-predict "1. e4 e5 2. Nf3" --elo 1800 --tc 180+2` prints the most likely moves,
the win / draw / loss probabilities and the expected think time. The same code ships in the Hugging Face
repository, so `transformers` alone runs it (`AutoModel.from_pretrained("yimingzhang/allie-2.0",
trust_remote_code=True)`, see the model card).

## Playing styles

| `play.mode` | What it does |
|---|---|
| `human` (default) | Samples a move from the model's predicted distribution for a player of `play.rating`, at `play.temperature` (1 = the model's own distribution). `rating = "opponent"` mirrors the opponent's rating, as the original Allie did. |
| `strongest` | Plays the most likely move of a player of `play.rating` (for example 2800). With `play.search` = 5, 8, 25 or 128, it runs `allie.search`'s coverage search with Allie 2.0's own output calibration and plays its most likely move. |

- **Think time.** With `play.think_time`, the bot waits for a think time drawn from the model's own think-time
  prediction, which depends on the position, the clock and the ratings. It never waits longer than
  `play.max_think` of its remaining clock. Compute time counts toward the wait.
- **Resigning.** The bot resigns when the model gives it at least `play.resign_loss` probability of losing for
  `play.resign_moves` of its own moves in a row, from ply 20.
- **Draws.** It accepts a draw offer when its expected score, P(win) + P(draw)/2, is at most `play.draw_accept`.
  From ply 60, it offers a draw when P(draw) is at least `play.draw_offer`.
- **Challenges.** `[challenge]` sets the accepted speeds, base times, increments, rated or casual games, humans
  or bots, and games per opponent. Only standard chess from the starting position is accepted. Beyond
  `max_games`, challenges are declined with "later".

## Chat

With `[chat] enabled = true` (default off), the bot talks in the game chat in natural language, written
by Claude (`claude-sonnet-5-5`, effort low, thinking off) from Allie's own numbers. No engine is involved.

- **One conversation per game.** The persona and policy (cached for an hour) and the game's header come
  first. Each model call appends a user turn with everything since the last call, then the model's
  decision as JSON `{"speak": bool, "text": str}`. Earlier turns are never edited, so each call reads
  the conversation from the prompt cache.
  - **Moves:** each one with Allie's prediction for its mover at their rating (top moves with
    probabilities, the move played marked), the think time typical there and the one taken, and the bot's
    win/draw/loss after it.
  - **Events:** draw offers and the bot's answer, takebacks, resignation, flag, abort, the opponent
    leaving, a rematch.
  - **Chat lines:** verbatim.
  - **The position:** clocks, the opening (Lichess's names), phase, material, FEN, Allie's prediction for
    the side to move.
- **When it calls the model.** Every chat message does (the model may still stay silent), during the
  game and for `linger` seconds after it. Other moments call it with a set probability:
  - `p_moment` for a surprising move or a swing in the win/draw/loss, at least `every` plies after the
    last call;
  - `p_draw` once a draw offer is answered;
  - `p_end` at the end.

  Otherwise the update waits in the next turn.
- **Casual or rated.** In casual games the bot gives its honest opinion from Allie's numbers when asked.
  In rated games it gives nothing that helps the opponent mid-game: it deflects like a human, and a
  message naming a move not yet played is dropped. After the game it reviews from Allie's numbers when
  asked.
- **`!quiet`.** It mutes the bot for that game and their rematches, including a message already being
  written.
- **Never in the way.** The chat reads its own copy of the game stream on a reader thread and keeps its
  own model state, so it does not wait behind a move's think time.
  - Replies are posted at once; unprompted messages wait `gap` seconds after the last one.
  - A call over `timeout` is dropped, and an error only skips a message.
  - A bad key or model turns the model off; a rate limit pauses it a minute.
  - After the game, once the stream closes, it reads new chat lines from the game's chat page.
- **Spend cap.** Each response's tokens are priced into a ledger file (`ledger`) by UTC day and month
  (Sonnet 5.5, $ per million tokens: 2 input, 2.50 / 4 cache write for 5 min / 1 h, 0.20 cache read,
  10 output). At `day_cap` or `month_cap` the chat goes silent until the window turns over. A spend
  limit on the key in the Anthropic Console is the backstop.
- **Setup.** `uv sync --extra chat`, then put an Anthropic API key in `~/.config/allie/anthropic_key`
  (`chmod 600`) or `ANTHROPIC_API_KEY`. Each call is logged with its tokens (new, cached, written,
  output), cost, time to the first token and latency; the key is never logged.
- **Dry run.** `analysis/lichess/chat_dryrun.py` replays recorded games through a real chat, with scripted
  messages, a draw offer and a resignation (`--llm mock` needs no key).

## Cost and accuracy

**Speed and memory.** One cached step, the time a move takes once the opponent's move arrives (median of 35
steps of a game, `analysis/lichess/speed.py`). The bot appends its own move while the opponent thinks, so
each decision reads one new token; a step of 16 games reads one token for each. CPU: int8 weights, the
`fast` backend (the default) unless noted. Memory: the process's resident size.

| Device | Threads | 1 game: ms per move | 16 games: ms per step | 64 games: ms per step | Memory |
|---|---:|---:|---:|---:|---:|
| AMD EPYC 9755 (Zen 5, AVX-512) | 8 | 6.3 | 32 (500 moves/s) | 101 | 6.4 GB |
| AMD EPYC 9755 | 32 | 4.6 | 17 (960 moves/s) | 41 (1,570 moves/s) | 6.4 GB |
| AMD EPYC 9755, PyTorch reference (`backend = "torch"`) | 8 | 19.5 | 185 | | 6.4 GB |
| AMD EPYC 9755, AVX2 only (`ALLIE_MARCH=-march=haswell`) | 4 / 8 | 11.7 / 6.7 | 93 / 48 | | 6.4 GB |
| AMD EPYC 9554 (Zen 4, AVX-512) | 8 / 16 / 32 | 7.6 / 5.0 / 4.4 | 51 / 30 / 20 | 165 / 93 / 56 | 6.4 GB |
| AMD EPYC 7763 (Zen 3, AVX2) | 8 / 16 | 12.7 / 8.0 | 76 / 42 | 246 / 136 | 6.4 GB |
| AMD EPYC 9755, BF16 weights | 16 | 7.8 | 30 | | 11 GB |
| GPU (NVIDIA RTX A6000), CUDA graphs, BF16 | | 7.3 | 25 | | 11 GB |
| GPU (RTX A6000), PyTorch reference | | 46.5 | 186 | | 11 GB |

- **Threads.** The default is PyTorch's thread count within one NUMA node (socket), pinned one per core and
  spread over the node's caches, with the weights moved to that node. One game reads 0.74 GB of weights a
  move, so it is bound by memory bandwidth: speed rises with threads until about 150-160 GB/s, three
  quarters of what these servers stream (`analysis/lichess/speed.py` prints the GB/s it reaches). A laptop
  or desktop streams 50-100 GB/s: about 10-20 ms a move with 6-8 cores. Threads on two sockets are slower
  than on one.
- **Batching.** Games in one step share each expert's weights: on CPU, 16 games cost 3.6 to 5 times one
  game and 64 games 9 to 16 times (fewer threads: more), so a bot with many games should batch them, as
  `allie-bot` does.
- **First use** compiles the kernels for the machine (a few seconds; they are cached under
  `~/.cache/allie`). It needs a C++ compiler (`c++`, or `CXX`); without one the model runs the PyTorch
  reference, with a warning. `ALLIE_MARCH` picks the target (e.g. `-march=haswell` for an AVX2 build) and
  `ALLIE_NUMA=0` leaves the weights' memory where it is.
- **Search.** `strongest` mode's search still runs the PyTorch path for its tree of moves: 5 simulations
  take about 0.4 s a move on CPU, 25 simulations 1.1-1.4 s.
- **First move.** It also reads the 11-token header.
- **How.** `fast.py` runs a whole step (input embedding, board CNN, 24 blocks, head) in one call into C++
  kernels on a pool of threads that stays alive between steps. Matrices are read in place, int8 rows
  widened to FP32 as they stream from memory, one row after another; the matrices of a step are split into
  chunks that threads take as they come free, with a barrier between phases (6 a block for one game). A
  step's tokens are grouped by expert, so each expert is read once however many games route to it. On GPU,
  steps that add one token to each game replay a CUDA graph of `model.py`'s forward, captured once per batch
  size and attention span, with each game's cache a slot of one pool. `backend = "torch"` runs `model.py`
  itself, the reference everything is checked against.

**Accuracy.** `analysis/lichess/parity.py` scores 5,000 positions of the Maia-3 blitz benchmark, 1,250 per
rating band, against the trained model's scores (the training forward on a GPU). With `--decode`, each
position is reached as live play reaches it: the game so far in one step, then its last move as a cached
step. The port is not bitwise identical, because floating-point sums run in a different order, but BF16 is
equal within noise. 95% intervals are bootstrapped over games.

| Setting | CE minus the trained model's (nats) | Top-1 (trained model: 58.92%) |
|---|---:|---:|
| CPU, int8, one game (the default) | +0.0017 [−0.0005, +0.0038] | 58.74% |
| CPU, BF16, one game | −0.0005 [−0.0014, +0.0004] | 58.74% |
| GPU, BF16, one game (CUDA graphs, the default) | −0.0003 [−0.0012, +0.0006] | 58.84% |
| PyTorch reference, CPU, int8, one game | +0.0013 [−0.0008, +0.0035] | 58.66% |
| PyTorch reference, CPU, BF16, one game | −0.0003 [−0.0013, +0.0006] | 58.92% |
| PyTorch reference, GPU, BF16, one game | −0.0006 [−0.0015, +0.0003] | 58.82% |
| PyTorch reference, GPU, BF16, 16 games in one step | −0.0005 [−0.0015, +0.0003] | 58.94% |
| PyTorch reference, GPU, BF16, whole games in one step | −0.0006 [−0.0015, +0.0002] | 58.90% |

- **Fast against the reference**, on the same positions: CPU int8 +0.0003 [−0.0003, +0.0008] nats. A position's
  largest change in any move's probability is 0.004 on average and 0.04 at most, and the top move agrees on
  99.4% of positions (CPU BF16: 99.4%, GPU: 99.6%). The C++ kernels round to BF16 where the reference does;
  the sums run in a different order.
- **int8 against BF16**, on the same positions: +0.0021 [+0.0001, +0.0042] nats with the fast backend
  (the PyTorch reference measured +0.0016, standard error 0.0011); the top move agrees on 97.6% of
  positions. We allowed 0.002 for the CPU default; `int8 = false` keeps BF16.
- **CPU against GPU** (BF16): the top move agrees on 99.3% of positions, and no move's probability differs by
  more than 0.076.

## Offline testing

- `pytest tests/lichess` runs the unit tests on a small random model and a local mock of the Bot API
  (`mock.py`). They cover challenge rules, full games, reconnects, rate limits, draw offers, resignation,
  takebacks, clock features against the training data loader, and board states against the native encoder.
- `allie-bot selfplay --config my-bot.toml --games 4` plays the real model against itself through the mock
  server. `--opponent random` plays a random mover instead. Every move goes through the full protocol, and the
  server checks it.
- `allie-bot bench --config my-bot.toml` measures the time per move.
- `python analysis/lichess/parity.py --model exports/allie-2.0` scores the bot's model on the Maia-3 benchmark
  positions against the training forward's scores (needs `ALLIE_DATA`).
