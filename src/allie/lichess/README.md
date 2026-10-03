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

1. **Install.** `uv sync --extra bot`. Add `--extra search` for the `strongest` mode's search.
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

RESULTS

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
