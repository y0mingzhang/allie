# Allie on Lichess

`allie.lichess` runs Allie 2.0 as a [Lichess bot](https://lichess.org/api#tag/Bot), and predicts moves from
Python or the command line. It needs no GPU: one CPU process holds the model once and plays several games at
a time.

- **Model.** A plain-PyTorch port of the trained network (`model.py`): no Triton kernels and no training code.
  Each game keeps its own key-value cache, so a move costs one or two new tokens. Concurrent games are batched
  into one forward pass.
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

With `[chat] enabled = true` (default off), the bot talks in the game chat in natural language. Claude
(`claude-sonnet-5-5`, effort low, thinking off) writes each message from facts the bot computes: the moves
and position, how likely Allie found each move at the mover's rating, Allie's win / draw / loss estimate
and how it moved, think times from the clocks, and after the game Stockfish's verdict on each side's
costliest move.

- **When it speaks.** A hello that mentions `!quiet`; a compliment when the opponent finds a move Allie
  gave under 15% at their rating that moves the expected score 10 points their way; a graceful word when
  their move costs the bot 15 points, or (once a game) when its expected score sinks 25 points below its
  high and under 40%; an answer when someone writes; a post-game message. At most `remarks` unprompted remarks a game, `every` plies apart, and `gap` seconds between
  messages. Messages that arrive while the model writes get one answer, to the last.
- **Fair play.** Stockfish runs only after the game. During it the model gets the position and Allie's
  numbers, but not the notes that would point the opponent somewhere: their last move when it helped the
  bot, the bot's estimate when its expected score is above 55%, or the move Allie expected instead. It is
  told never to suggest moves or point out threats, and a mid-game message that names a move not yet
  played is dropped. `!quiet` from the opponent mutes the bot for that game and their rematches, including
  a message already being written.
- **Never in the way.** Each game's chat runs on its own thread (the game thread only notes Allie's view
  and queues work); a call that takes over `timeout` (6 s) is dropped, a remark is dropped if the game
  moved on two plies, and an error only skips a message. A game that ends without a final state (a
  crash, shutdown) drops what is queued.
  A bad key or model turns the model off (the hello and `!quiet` reply stay); a rate limit pauses it a
  minute.
- **Spend cap.** Each response's tokens are priced (Sonnet 5.5: $2 input, $2.50 cache write, $0.20
  cache read, $10 output per million) into a ledger file (`ledger`) by UTC day and month. At `day_cap`
  ($1) or `month_cap` ($15) the chat goes silent, logged once, until the window turns over. A spend limit
  on the key in the Anthropic Console is the backstop.
- **Setup.** `uv sync --extra chat`, then put an Anthropic API key in `~/.config/allie/anthropic_key`
  (`chmod 600`) or `ANTHROPIC_API_KEY`. The bot logs each call's tokens (cached and not) and latency,
  never the key.
- **Dry run.** `analysis/lichess/chat_dryrun.py record` replays recorded games through the model as the
  bot feeds it; `replay` prints the chat they would have had, with scripted opponent messages
  (`--llm mock` needs no key).

## Cost and accuracy

**Speed and memory.** One cached step, the time a move takes once the opponent's move arrives (median of 35
steps of a game, `analysis/lichess/speed.py`). The bot appends its own move while the opponent thinks, so
each decision reads one new token. CPU: 8 threads of an AMD EPYC 9354. GPU: one NVIDIA L40S.

| Device | Weights | Experts | ms per move | Memory |
|---|---|---:|---:|---:|
| CPU | int8 (CPU default) | 16 | 29 | 8 GB |
| CPU | int8 | 8 | 24 | 8 GB |
| CPU | BF16 | 16 | 69 | 11 GB |
| GPU, one game | BF16 | 16 | 18 | 11 GB |
| GPU, 16 games in one step | BF16 | 16 | 63 per step, 4 per game | 11 GB |

- **Threads.** On CPU, 6 to 8 threads are best for one game (4 threads: 32 ms); more than the physical cores
  slows it down.
- **Search.** `strongest` mode with 5 simulations takes about 0.4 s a move on CPU (BF16), 25 simulations 1.1-1.4 s.
- **First move.** It also reads the 11-token header: about 0.3 s.
- **Batching.** On CPU, two games at once take 47 ms per step instead of 29 for one, so batching saves little.
  On GPU, a step of 16 games costs 3.5 times a step of one.
- **How.** int8 weights use PyTorch's int8 weight-only kernel with one scale per output row. One or two new
  tokens run each token's 16 experts one by one; on GPU, a few tokens gather their experts' weights into one
  batched product, and a batch of games runs every expert at once. Every path is checked against the same
  reference below.

**Accuracy.** `analysis/lichess/parity.py` scores 5,000 positions of the Maia-3 blitz benchmark, 1,250 per
rating band, against the trained model's scores (the training forward on a GPU). With `--decode`, each
position is reached as live play reaches it: the game so far in one step, then its last move as a cached
step. The port is not bitwise identical, because floating-point sums run in a different order, but BF16 is
equal within noise. 95% intervals are bootstrapped over games.

| Setting | CE minus the trained model's (nats) | Top-1 (trained model: 58.92%) |
|---|---:|---:|
| CPU, BF16, one game | −0.0003 [−0.0013, +0.0006] | 58.92% |
| CPU, int8, one game | +0.0013 [−0.0008, +0.0035] | 58.66% |
| GPU, BF16, one game | −0.0006 [−0.0015, +0.0003] | 58.82% |
| GPU, BF16, 16 games in one step | −0.0005 [−0.0015, +0.0003] | 58.94% |
| GPU, BF16, whole games in one step | −0.0006 [−0.0015, +0.0002] | 58.90% |

- **int8 against BF16**, on the same positions: +0.0016 ± 0.0011 nats (standard error); the top move agrees on
  97.5% of positions. This is within the 0.002 we allowed for the CPU default; `int8 = false` keeps BF16.
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
