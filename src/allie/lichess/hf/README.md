---
license: mit
library_name: transformers
tags:
  - chess
  - human-behavior
  - mixture-of-experts
---

# Allie 2.1

> **Disclaimer:** the research, code and this card were produced by AI agents (Claude Code and OpenAI Codex),
> directed by Yiming Zhang.

Allie 2.1 predicts the move a human chess player will make. It reads the game so far, both players' ratings,
the time control and the clock. It also predicts how long the player will think and how the game will end.
It follows [Allie 2.0](https://huggingface.co/yimingzhang/allie-2.0) and the original Allie
([paper](https://arxiv.org/abs/2410.03893), [code](https://github.com/ippolito-cmu/allie)): a larger model,
trained on 1.5 times the tokens, with a lower cross-entropy in every time control and rating band we test.

## Quick start

```sh
pip install transformers torch python-chess
```

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("yimingzhang/allie-2.1", trust_remote_code=True)
probs = model.predict("1. e4 e5 2. Nf3", white_elo=1800, black_elo=1750, time_control="180+2")
print(list(probs.items())[:3])  # most likely moves (UCI) for black, a 1750 player
move = model.play("1. e4 e5 2. Nf3", elo=1200)  # a move sampled as a 1200 player would play it
```

- **Moves:** PGN text (with `[%clk]` comments if you have them), or a list of UCI or SAN moves.
- **Time control:** `"180+2"` or `(180, 2)` in seconds. Leave it out if unknown.
- **Clocks:** `clocks=[...]` gives each mover's seconds left after each move. Without them, the model reads
  the clock as unknown, as it did for its clockless training games.
- **More outputs:** `model.analyze(...)` also returns the mover's win / draw / loss probabilities and their
  expected think time.
- **Device:** `device="cuda"` (the default when a GPU is visible) or `device="cpu"`. On CPU the weights
  default to int8 (9.8 GB, about 1.8 times as fast as BF16 for one game); `int8=False` keeps BF16 (18.3 GB).
  `active_experts=8` routes each token through 8 of its 16 experts: faster, slightly less accurate.
- **Speed:** on CPU the model runs Allie's Rust engine when it is installed (Linux on x86-64, with
  [Rust](https://rustup.rs): `pip install "allie-fast @ git+GITHUB_URL#subdirectory=rust/allie-fast"`), and
  otherwise the plain PyTorch code, about three times slower, with a warning; on GPU replayed CUDA graphs.
  `backend="torch"` runs the plain PyTorch reference, `threads=` sets the CPU threads. Calls that extend the
  previous call's game reuse its key-value cache, so following a game move by move costs one new token per call.

| Device | Weights | ms per move | Memory |
|---|---|---:|---:|
| CPU, 12 threads (AMD EPYC 7763) | int8 (CPU default) | 14.3 | 9.8 GB |
| CPU, 12 threads (AMD EPYC 7763) | BF16 | 25.4 | 18.3 GB |
| CPU, 12 threads, 16 games in one step | int8 | 80.2 per step | 9.8 GB |

## Play on Lichess, and the command line

The [allie package](GITHUB_URL) runs the same code as a Lichess bot, and has a command line:

```sh
uv tool install "allie[bot] @ git+GITHUB_URL" --with "allie-fast @ git+GITHUB_URL#subdirectory=rust/allie-fast"
allie-predict "1. e4 e5 2. Nf3" --elo 1800 --tc 180+2 --model yimingzhang/allie-2.1
LICHESS_TOKEN=lip_... allie-bot play --config lichess-bot.toml  # model = "yimingzhang/allie-2.1"
```

`allie-predict` prints the most likely moves, the win / draw / loss probabilities and the think time.
`allie-bot` plays as a Lichess BOT account: it samples moves at the opponent's rating (or a set one), waits
human-like think times, and handles challenges, draws, resignation and reconnects. See the package's
`src/allie/lichess/README.md` for setup.

Allie 2.1 has played on Lichess as [AllieTheChessBot](https://lichess.org/@/AllieTheChessBot) since October 30,
2026. Its `calibrated` mode searches, then samples, to play close to the strength of humans of the rating:
the package's `calibration.py` (one rule per time control) and `behaviour.json` (timing, resign and draw rules)
are fitted for Allie 2.1. To run the bot with Allie 2.0, check out the whole of [commit cad41559](GITHUB_URL/tree/cad41559): its code, rule and default model are Allie 2.0's.

## Model

- **Architecture.** A mixture-of-experts transformer with 1.23B active and 9.19B total parameters: 24 blocks
  of width 2048 (32 heads of 64), from the modded-nanoGPT medium track. The first three blocks have dense MLPs.
  The other 21 route each token to 16 of 256 small experts, plus one shared expert; the first four routed
  blocks centre the router's input. Attention is exclusive self-attention (XSA): each head's output loses a
  learned share of its projection on the token's own value.
- **Inputs.** An 11-token header holds the time control and both ratings, then one token per move. At every
  position the model also gets three clock features (the mover's time left, the opponent's, and the mover's
  previous think time) and a small CNN's reading of the board.
- **Outputs.** One head gives the next move, the move's think time (63 bins) and the game result for the mover.
- **Cost.** 2.45 GFLOPs per move with the game cached (Allie 2.0: 1.39).

## Training

- **Data.** Every Lichess month with clock times from May 2017 to August 2026, except the test month, July
  2026, as for Allie 2.0. Over-the-board and engine games are added. The sampler doubles a game's weight for
  every 200 rating points of its stronger player.
- **Run.** 112B tokens: 213,623 steps of 524,288 tokens, on 8 NVIDIA L40S GPUs for 14.9 days. The learning
  rates warm up over 2,013 steps to 0.35 times Allie 2.0's base rates, then decay linearly to 0.1% of the peak.
  Adam steps on every other step, instead of every step at half the rate as for Allie 2.0.
- **Weights.** The released weights average the final weights with an exponential moving average of the
  weights over the end of training (horizon 17,090 steps, 8% of the run, updated every 64 steps from step
  162,354), half and half. This lowers CE by 0.0008 nats on the main evaluation against the final weights
  alone (0.0007 on September games); unlike Allie 2.0 there is no second anneal.
- **Compute.** 8.3e20 training FLOPs, 2.6 times Allie 2.0's.

## Evaluation

On Lichess games from July 2026, which no model here trained on, unless noted. Cross-entropy (CE) is the
negative log-probability, in nats, of the move the human played; lower is better.

**Blitz benchmark.** 80,000 positions, 20,000 in each band of the mover's rating (<1400, 1400-2000,
2000-2400, ≥2400). Probabilities are renormalised over the legal moves.

| Model | Parameters | GFLOPs per move | CE | Top-1 (%) |
|---|---|---:|---:|---:|
| Maia-3 5M | 5.2M | 0.60 | 1.2955 | 56.95 |
| Maia-3 23M | 22.9M | 2.40 | 1.2475 | 58.34 |
| Maia-3 79M | 78.9M | 9.23 | 1.2269 | 58.80 |
| Original Allie | 305M | 0.61 | 1.2848 | 57.06 |
| Allie 2.0 | 0.69B active / 5.6B total | 1.39 | 1.2030 | 59.29 |
| **Allie 2.1** | 1.23B active / 9.19B total | 2.45 | **1.1779** | **60.12** |

Allie 2.1 minus Maia-3 79M: −0.0489 [−0.0522, −0.0458] nats of CE and +1.32 [+1.11, +1.53] points of top-1
accuracy; minus Allie 2.0: −0.0251 [−0.0272, −0.0230] nats and +0.83 [+0.65, +1.02] points (95% intervals over
games).
[Maia-3](https://arxiv.org/abs/2605.19091) is the state of the art in human-move prediction, from Ashton
Anderson's group; its open weights made this comparison possible.

**Main evaluation.** 16 cells (bullet, blitz, rapid and classical, each in the four rating bands), about
100,000 moves each, CE averaged over cells without legal-move masking: **1.2240** (Allie 2.0: 1.2505), and 1.0603
over the four ≥2400 cells (Allie 2.0: 1.1001); lower in all 16 cells. On the same cells from September 2026
games, after every model's training data: 1.2265 (Allie 2.0: 1.2535).

**This release's code.** The inference here matches the training code within 0.0025 nats. On 5,000 benchmark
positions, each scored as a live game reaches it (the last move added to a cached game), CE minus the training
code's is −0.0015 [−0.0023, −0.0006] nats with the CPU kernels in BF16, −0.0010 [−0.0021, +0.0002] in int8,
and −0.0013 [−0.0022, −0.0005] with the CUDA graphs on GPU.

**int8, the CPU default,** costs +0.0005 nats of CE against BF16 on those 5,000 positions, and the top move
agrees on 99.4% of them, for about half the memory and 1.8 times the speed for one game (1.3 times for 16).
`int8=False` keeps BF16.

**Search.** On 20,000 benchmark positions, 100 simulations of the package's search, its output calibrated
on separate July games, lower CE by 0.0028 [0.0016, 0.0040] nats (Allie 2.0: 0.0052), at 244 GFLOPs per move;
the gain comes from strong players.

## Intended use and limitations

- **Use.** Research on human decision-making in chess, human-like opponents and training partners, and
  analysis of how players of a given rating play a position.
- **Not a strong engine.** It predicts human moves, mistakes included. Its most likely move at a high rating
  is still a human guess, not a calculated best move.
- **Comparison caveats.** Allie reads the clock and the whole game; the released Maia-3 reads neither, and
  trained on blitz only. Training compute and data are not matched.
- **Thin data at the top.** Above rating 2700 the test month has only 43 games.
- **Context.** Games longer than 1,014 plies do not fit.

## License

MIT.
