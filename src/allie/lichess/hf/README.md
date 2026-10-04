---
license: mit
library_name: transformers
tags:
  - chess
  - human-behavior
  - mixture-of-experts
---

# Allie 2.0

> **Disclaimer:** the research, code and this card were produced by AI agents (Claude Code and OpenAI Codex),
> directed by Yiming Zhang.

Allie 2.0 predicts the move a human chess player will make. It reads the game so far, both players' ratings,
the time control and the clock. It also predicts how long the player will think and how the game will end.
It continues the original Allie ([paper](https://arxiv.org/abs/2410.03893),
[code](https://github.com/ippolito-cmu/allie)).

Allie 2.0 is the annealed model: the main training run's final checkpoint after a short second pass at a
low learning rate over 1B tokens of recent games, which lowered CE in every time control. The checkpoint
before that pass stays in this repository under the tag `pre-anneal` (`revision="pre-anneal"`).

## Quick start

```sh
pip install transformers torch python-chess
```

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("yimingzhang/allie-2.0", trust_remote_code=True)
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
  default to int8 (6.4 GB, about 1.6 times as fast as BF16); `int8=False` keeps BF16 (11 GB).
  `active_experts=8` routes each token through 8 of its 16 experts: faster, slightly less accurate.
- **Speed:** on CPU the model runs C++ kernels compiled for your machine on first use (a few seconds; it needs
  a C++ compiler, and without one runs the plain PyTorch code), on GPU replayed CUDA graphs. `backend="torch"`
  runs the plain PyTorch reference, `threads=` sets the CPU threads. Calls that extend the previous call's
  game reuse its key-value cache, so following a game move by move costs one new token per call.

| Device | Weights | ms per move | Memory |
|---|---|---:|---:|
| CPU, 8 threads (AMD EPYC 9755) | int8 (CPU default) | 6.3 | 6.4 GB |
| CPU, 32 threads (AMD EPYC 9755) | int8 | 4.6 | 6.4 GB |
| CPU, 8 threads, AVX2 only | int8 | 6.7 | 6.4 GB |
| CPU, 16 threads (AMD EPYC 9755) | BF16 | 7.8 | 11 GB |
| GPU (NVIDIA RTX A6000), one game | BF16 | 7.3 | 11 GB |
| GPU, 16 games in one step | BF16 | 1.5 per game | 11 GB |

## Play on Lichess, and the command line

The [allie package](GITHUB_URL) runs the same code as a Lichess bot, and has a
command line:

```sh
uv tool install "allie[bot] @ git+GITHUB_URL"
allie-predict "1. e4 e5 2. Nf3" --elo 1800 --tc 180+2
LICHESS_TOKEN=lip_... allie-bot play --config lichess-bot.toml
```

`allie-predict` prints the most likely moves, the win / draw / loss probabilities and the think time.
`allie-bot` plays as a Lichess BOT account: it samples moves at the opponent's rating (or a set one), waits
human-like think times, and handles challenges, draws, resignation and reconnects. See the package's
`src/allie/lichess/README.md` for setup.

## Model

- **Architecture.** A mixture-of-experts transformer with 0.69B active and 5.6B total parameters: 24 blocks
  of width 1536, from the modded-nanoGPT medium track. The first block has a dense MLP. The other 23 route
  each token to 16 of 256 small experts, plus one shared expert.
- **Inputs.** An 11-token header holds the time control and both ratings, then one token per move. At every
  position the model also gets three clock features (the mover's time left, the opponent's, and the mover's
  previous think time) and a small CNN's reading of the board.
- **Outputs.** One head gives the next move, the move's think time (63 bins) and the game result for the mover.
- **Cost.** 1.39 GFLOPs per move with the game cached.

## Training

- **Data.** Every Lichess month with clock times from May 2017 to August 2026, except the test month, July
  2026: 7.9B games. Over-the-board and engine games are added. The sampler doubles a game's weight for
  every 200 rating points of its stronger player.
- **Main run.** 75B tokens: 143,051 steps of 524,288 tokens, on 8 NVIDIA L40S GPUs for 6.6 days.
- **Second anneal.** From the main run's final checkpoint with a fresh optimizer: 1B tokens (1,907 steps) of
  Lichess games from January 2024 to August 2026 (July 2026 still held out). The learning rate peaks at
  0.05, where the main run's schedule stood at about 99% (its peak was 4.0), and decays over the last 30% of
  steps. 2.1 hours on the same GPUs.
- **Compute.** 3.2e20 training FLOPs in all.

## Evaluation

All on Lichess games from July 2026, which no model here trained on. Cross-entropy (CE) is the negative
log-probability, in nats, of the move the human played; lower is better.

**Blitz benchmark.** 80,000 positions, 20,000 in each band of the mover's rating (<1400, 1400-2000,
2000-2400, ≥2400). Probabilities are renormalised over the legal moves.

| Model | Parameters | GFLOPs per move | CE | Top-1 (%) |
|---|---|---:|---:|---:|
| Maia-3 5M | 5.2M | 0.60 | 1.2955 | 56.95 |
| Maia-3 23M | 22.9M | 2.40 | 1.2475 | 58.34 |
| Maia-3 79M | 78.9M | 9.23 | 1.2269 | 58.80 |
| Original Allie | 305M | 0.61 | 1.2848 | 57.06 |
| Allie 2.0 before the anneal | 0.69B active / 5.6B total | 1.39 | 1.2056 | 59.26 |
| **Allie 2.0** | 0.69B active / 5.6B total | 1.39 | **1.2030** | **59.29** |

Allie 2.0 minus Maia-3 79M: −0.0238 [−0.0272, −0.0208] nats of CE and +0.48 [+0.27, +0.70] points of top-1
accuracy (95% intervals over games); the anneal's share is −0.0026 [−0.0033, −0.0019] nats.
[Maia-3](https://arxiv.org/abs/2605.19091) is the state of the art in human-move prediction, from Ashton
Anderson's group; its open weights made this comparison possible.

**Main evaluation.** 16 cells (bullet, blitz, rapid and classical, each in the four rating bands), about
100,000 moves each, CE averaged over cells without legal-move masking: **1.2505**, and 1.1001 over the four
≥2400 cells (1.2533 and 1.1035 before the anneal; lower in all four time controls).

**This release's code.** The inference here matches the training code within noise. On 5,000 benchmark
positions, each scored as a live game reaches it (the last move added to a cached game), CE minus the training
code's is +0.0003 [−0.0007, +0.0012] nats with the C++ kernels on CPU in BF16, and +0.0010 [−0.0011, +0.0032]
in int8.

**int8, the CPU default,** costs +0.0008 [−0.0013, +0.0030] nats of CE against BF16 on those positions, and
the top move agrees on 97.7% of them, for half the memory and about 1.6 times the speed. `int8=False` keeps
BF16.

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
