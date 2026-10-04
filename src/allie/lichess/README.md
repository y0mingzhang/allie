# Allie on Lichess

`allie.lichess` runs Allie-v3.0 as a [Lichess bot](https://lichess.org/api#tag/Bot). It needs no GPU: one CPU
process holds the model once and plays several games at a time.

- **Model.** A plain-PyTorch port of the trained network (`model.py`): no Triton kernels and no training code.
  Each game keeps its own key-value cache, so a move costs one or two new tokens. Concurrent games are batched
  into one forward pass.
- **Inputs as in training.** The game header carries both ratings and the time control. Each move carries the
  board, both clocks and the mover's previous think time, computed as the training data computes them.
- **Client.** The Bot API over the standard library (`client.py`): challenges, game streams, moves, draws,
  resignation, reconnects and rate limits.

## Setup

1. **Install.** `uv sync --extra bot` (the extra is only for exporting weights). Add `--extra search` for the `strongest` mode's search.
2. **Export the weights** from a training checkpoint, once:
   ```sh
   allie-bot export results/pretrain/allie-v3.0/last.pt exports/allie-v3.0
   ```
   This writes `model.safetensors` (11 GB, BF16) and `config.json`. The bot needs only this directory.
3. **Create a bot account.** Bot accounts must be new accounts that have never played a game. Create one on
   lichess.org, then make a personal API token with the `bot:play` scope at
   <https://lichess.org/account/oauth/token/create?scopes[]=bot:play>. Upgrade the account once:
   ```sh
   curl -d '' https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer $LICHESS_TOKEN"
   ```
4. **Configure.** Copy [configs/lichess-bot.toml](../../../configs/lichess-bot.toml) and set `model` to the
   export directory. The token is never stored in the config: put it in the `LICHESS_TOKEN` environment variable.
5. **Run.**
   ```sh
   export LICHESS_TOKEN=lip_...
   allie-bot play --config my-bot.toml
   ```
   Any config key can be overridden with `--set`, for example `--set play.rating=1800`. With
   `--drain-file PATH`, creating that file makes the bot decline new challenges, finish its games and exit.

## Playing styles

| `play.mode` | What it does |
|---|---|
| `human` (default) | Samples a move from the model's predicted distribution for a player of `play.rating`, at `play.temperature` (1 = the model's own distribution). `rating = "opponent"` mirrors the opponent's rating, as the original Allie did. |
| `strongest` | Plays the most likely move of a player of `play.rating` (for example 2800). With `play.search` = 5, 8, 25 or 128, it runs `allie.search`'s coverage search with Allie-v3.0's own output calibration and plays its most likely move. |

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

## Cost and accuracy

**Speed and memory.** Time to choose a move, from receiving the opponent's move, on one AMD EPYC 9554 node
with 8 threads (median of a 60-ply game). The bot appends its own move while the opponent thinks, so each
decision reads one new token.

| Setting | ms per move | Resident memory |
|---|---:|---:|
| BF16, all 16 experts (default) | 83 | 11 GB |
| int8 weights | 40 | 7.7 GB |
| `strongest` with 5 simulations of search | 440 | 11 GB |
| `strongest` with 25 simulations of search | 1,100-1,400 | 11 GB |

The first move of a game also reads the 11-token header: about 0.3 s. On an AMD EPYC 7763, BF16 takes 65-80 ms
with 8-16 threads, and 8 experts instead of 16 saves about a quarter. More threads than physical cores slow it
down. Two games at once roughly double each decision's time: on a CPU, batching games saves little.

**Accuracy.** `analysis/lichess/parity.py` scores the bot's own code path on 5,000 positions of the Maia-3 blitz
benchmark, 1,250 per rating band, against the trained model's scores (the training forward on a GPU). The
port is not bitwise identical, because floating-point sums run in a different order, but it is equal within
noise. 95% intervals are bootstrapped over games.

| Setting | CE minus the trained model's (nats) | Same top move as the GPU port |
|---|---:|---:|
| GPU, BF16 | −0.0002 [−0.0012, +0.0006] | |
| CPU, BF16 | −0.0003 [−0.0012, +0.0006] | 99.6% |
| CPU, int8 weights | +0.0012 [−0.0009, +0.0034] | 97.6% |
| CPU, 8 of 16 experts | +0.0014 [−0.0003, +0.0031] | 98.6% |

CPU and GPU give the same top move on 99.6% of positions; the largest difference in any move's probability is
0.055. The int8 and 8-expert rows were measured one revision earlier, before gates were rounded to BF16 as in
training.

## Offline testing

- `pytest tests/lichess` runs the unit tests on a small random model and a local mock of the Bot API
  (`mock.py`). They cover challenge rules, full games, reconnects, rate limits, draw offers, resignation,
  takebacks, clock features against the training data loader, and board states against the native encoder.
- `allie-bot selfplay --config my-bot.toml --games 4` plays the real model against itself through the mock
  server. `--opponent random` plays a random mover instead. Every move goes through the full protocol, and the
  server checks it.
- `allie-bot bench --config my-bot.toml` measures the time per move.
- `python analysis/lichess/parity.py --model exports/allie-v3.0` scores the bot's model on the Maia-3 benchmark
  positions against the training forward's scores (needs `ALLIE_DATA`).
