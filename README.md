# Allie

Allie predicts the move a human chess player will make, given the game so far, both players' ratings and the clock. The current model is a 24-block mixture-of-experts (MoE) transformer with 0.69B active and 5.6B total parameters, trained on 75B tokens of Lichess and over-the-board games.

**On held-out Lichess blitz from July 2026, every model of the public Maia-3 family (5M, 23M and 79M parameters) is beaten on both cross-entropy and top-1 accuracy by a cheaper Allie model.** Against the largest, Maia-3 79M, the full model is ahead by 0.0238 nats and 0.48 accuracy points at 6.6x less compute per move, and a version fine-tuned to use 4 of its 16 experts per token is still ahead at 10x less. On our main all-format evaluation the run scored 1.2533 nats, 0.032 below what our scaling law forecast for it (1.2856).

![Pareto plot: cross-entropy and top-1 accuracy against inference GFLOPs per move for Allie and Maia-3](docs/figures/pareto.png)

| Maia-3 model | its cost (GFLOPs/move) | cheapest Allie model ahead on both metrics | cost | CE difference (nats) | top-1 difference (pp) |
|---|---:|---|---:|---:|---:|
| 5M | 0.60 | distilled student, MoE 181M active / 1.42B total | 0.36 | −0.0410 [−0.0446, −0.0374] | +0.96 [+0.74, +1.19] |
| 23M | 2.40 | big run fine-tuned to use 2 of 16 experts | 0.82 | −0.0270 [−0.0307, −0.0236] | +0.41 [+0.19, +0.64] |
| 79M | 9.23 | big run fine-tuned to use 4 of 16 experts | 0.90 | −0.0180 [−0.0215, −0.0149] | +0.34 [+0.12, +0.56] |

Differences are Allie minus Maia-3 on the same 80,000 positions (negative CE and positive accuracy favour Allie), with 95% intervals from bootstrapping whole games; pp = percentage points. The caveats below matter for reading these numbers.

## Contents

1. [How we measure](#how-we-measure)
2. [Results against Maia-3](#results-against-maia-3)
3. [Caveats](#caveats)
4. [The model](#the-model)
5. [Training](#training)
6. [Scaling laws](#scaling-laws)
7. [Cheaper inference](#cheaper-inference)
8. [What we learned](#what-we-learned)
9. [Next](#next)
10. [Reproducing](#reproducing)

## How we measure

Both evaluations use Lichess games from **July 2026**, a month excluded from all training. Scores are cross-entropy (CE, in nats; lower is better): the negative log-probability the model gives the move the human actually played.

- **Blitz benchmark (head-to-head with Maia-3).** 80,000 positions from July 2026 blitz games, 20,000 in each band of the mover's rating (<1400, 1400-2000, 2000-2400, ≥2400). Every model scores the same positions. Maia-3 is the public 5M, 23M and 79M checkpoints from the University of Toronto's CSSLab. Both models' probabilities are renormalized over the legal moves, and we also report top-1 accuracy (how often the most likely move is the one played). Maia-3 gets its own inputs: the current board, the 7 previous boards and both ratings. Allie gets the whole game so far, both ratings and the clock. Our reimplementation of Maia-3's input pipeline reproduces the authors' own inference code move for move. Intervals resample whole games (2,000 draws); model differences are paired on identical positions.
- **Main evaluation (all formats).** 16 cells: bullet, blitz, rapid and classical, each split into the same four rating bands, with about 100,000 scored moves per cell (1.55M moves from 26,278 games). CE is over all 1,968 move tokens without legal-move masking, averaged over the 16 cells. Every training decision in this project was made on this number. It is not comparable to benchmark numbers.

## Results against Maia-3

All points on the frontier, blitz benchmark:

| Model | GFLOPs/move | CE | top-1 (%) | vs Maia-3 79M: CE | vs 79M: top-1 (pp) |
|---|---:|---:|---:|---:|---:|
| Maia-3 5M | 0.60 | 1.2955 | 56.95 | +0.0686 [+0.0657, +0.0715] | −1.85 [−2.06, −1.64] |
| Maia-3 23M | 2.40 | 1.2475 | 58.34 | +0.0206 [+0.0189, +0.0223] | −0.47 [−0.63, −0.29] |
| Maia-3 79M | 9.23 | 1.2269 | 58.80 | | |
| **Allie big run, after a second anneal** | 1.39 | **1.2030** | **59.29** | **−0.0238 [−0.0272, −0.0208]** | **+0.48 [+0.27, +0.70]** |
| Allie big run, final checkpoint | 1.39 | 1.2056 | 59.26 | −0.0213 [−0.0245, −0.0184] | +0.46 [+0.24, +0.67] |
| final checkpoint, best 8 of 16 experts, no training | 1.06 | 1.2067 | 59.27 | −0.0202 [−0.0235, −0.0172] | +0.47 [+0.26, +0.69] |
| final checkpoint, best 6 of 16 experts, no training | 0.98 | 1.2081 | 59.21 | −0.0188 [−0.0221, −0.0157] | +0.41 [+0.19, +0.63] |
| fine-tuned to use 4 of 16 experts | 0.90 | 1.2088 | 59.14 | −0.0180 [−0.0215, −0.0149] | +0.34 [+0.12, +0.56] |
| fine-tuned to use 2 of 16 experts | 0.82 | 1.2205 | 58.75 | −0.0064 [−0.0100, −0.0033] | −0.06 [−0.27, +0.17] |
| distilled student, MoE 261M active / 2.07B total | 0.52 | 1.2469 | 58.09 | +0.0201 [+0.0168, +0.0231] | −0.71 [−0.92, −0.50] |
| distilled student, MoE 181M active / 1.42B total | 0.36 | 1.2545 | 57.92 | +0.0276 [+0.0240, +0.0309] | −0.89 [−1.12, −0.67] |

The plot below follows the Maia project's own presentation: accuracy and CE against the game's rating (the mean of both players'). The final checkpoint has lower CE than Maia-3 79M in 22 of 23 100-point bins, with intervals below zero in 19; the exception is 2800-2900, which has 7 games. Its accuracy is at or above 79M's in every bin except 800-900 (−0.12 pp) and 2800-2900 (−0.33 pp), both within noise.

![Accuracy and CE against game rating for Maia-3 5M, 23M, 79M and two Allie models](docs/figures/rating.png)

Where the lead comes from (final checkpoint minus Maia-3 79M, blitz unless stated):

| Slice | positions | CE difference | top-1 difference (pp) |
|---|---:|---:|---:|
| bullet (5,000 per rating band) | 20,000 | −0.1412 [−0.1514, −0.1310] | +3.33 [+2.81, +3.83] |
| rapid (5,000 per band) | 20,000 | −0.0163 [−0.0222, −0.0102] | +0.53 [+0.11, +0.94] |
| classical (5,000 per band) | 20,000 | −0.0208 [−0.0278, −0.0144] | +0.45 [−0.00, +0.90] |
| mover has under 10 s left | 2,852 | −0.2229 [−0.2627, −0.1823] | +3.47 [+1.88, +4.91] |
| mover has 60-120 s left | 15,719 | +0.0107 [+0.0034, +0.0182] | −0.53 [−1.06, −0.02] |
| plies 40-59 (middlegame) | 15,997 | +0.0071 [+0.0003, +0.0137] | −0.09 [−0.62, +0.42] |
| Maia-3's own protocol: ply > 20 and at least 30 s left | 48,987 | −0.0057 [−0.0095, −0.0019] | +0.12 [−0.16, +0.40] |

The biggest margins are where Maia-3 is weakest by construction: bullet, which it was not trained on, and time trouble, which it cannot see. Maia-3 79M still holds a small edge in the middlegame and with one to two minutes on the clock, and on its own evaluation protocol the accuracy difference is a tie.

## Caveats

- **Not a matched-compute or matched-data comparison.** Maia-3 was trained on about 2.5 years of Lichess blitz (January 2023 to July 2025); its training compute is not published. Allie trained on all formats from 2017 to 2026 plus over-the-board and engine games, with 3.1e20 FLOPs. The claim is about inference cost per move, not training efficiency.
- **Our data is more recent.** Maia-3's data ends a year before the test month; ours runs up to it. August 2026, the month after the test month, is in our training data (one of 111 Lichess months, 1.2% of the pool's tokens). July 2026 itself is excluded from everything. The second anneal and the expert fine-tunes used only recent months (January 2024 to June 2026, plus August 2026), so part of what they buy is recency.
- **The inputs differ.** The released Maia-3 checkpoints take no clock input, while Allie reads three clock features; our largest gains are in time trouble. Maia-3 sees 8 recent boards; Allie sees the whole game, including the opening.
- **FLOPs, not latency.** Allie's cost is analytic, 2 × active matmul parameters per move with a cached game (attention adds about 0.008 GFLOPs at the mean context); Maia-3's is its measured forward FLOPs per 64-square board. Serving the MoE keeps all 5.6B weights in memory (about 11 GB in BF16) against about 0.3 GB for Maia-3 79M. We do not compare wall-clock latency.
- **Some choices were made on the benchmark.** The second anneal's learning rate was picked from two arms using these positions, and the fine-tune and distillation settings came from pilots on them. The final checkpoint, which predates all of that, is already ahead of Maia-3 79M (−0.0213 nats, +0.46 pp).
- **Thin bins and one seed.** Above game rating 2700 the test month has 36 and 7 games. The big run is a single training run.

## The model

![Diagram of the model: tokens, clock and board inputs, 24 transformer blocks with mixture-of-experts layers, and three outputs](docs/figures/model.png)

A game is a sequence of tokens: an 11-token header (start, base time, increment, and each player's rating as four digits), then one token per move from a vocabulary of the 1,968 possible moves in from-square/to-square notation. At every position two more inputs are added to the token embedding: three clock features (the mover's and the opponent's time left, and the mover's previous thinking time, encoded as Fourier features) and a small CNN over the current board (13 piece planes on 8×8, three 3×3 convolutions with 32 channels). Games are packed into 16K-token rows with attention masked at game boundaries, so each move attends to its own game's history only.

The trunk is 24 transformer blocks of width 1536 with 24 heads of size 64, descended from the [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) medium track: QK-normalized attention with rotary positions on half of each head, gated attention outputs, value embeddings, U-net skip connections, two re-injected input embeddings and a soft-capped output. The first block has a dense SwiGLU MLP (hidden width 4096). In the other 23 blocks, a mixture-of-experts layer takes its place:

- 256 routed SwiGLU experts of hidden width 192 plus one shared expert of hidden width 1024; each token uses 16 routed experts, so the active width (16 × 192 + 1024 = 4096) matches the dense block.
- A router scores every expert with a sigmoid; the top 16 by score plus a per-expert bias are chosen, and their scores, renormalized to sum to 4, weight their outputs. The biases are reset every step to balance load (quantile balancing, as in Kimi K3), with a small sequence-level balance loss on top. There is no capacity limit and no token is dropped.
- Two stabilizers came out of the failures described [below](#what-we-learned): the router sees its input minus a running mean of it (router-input centring), and the sum that normalizes the gates is floored at 1e-12.

One output head serves three targets: the next move (the training objective) and, as auxiliary losses at weight 0.2 each, the thinking time of that move (63 bins) and the game result from the mover's side.

**Inference cost.** With the game's key-value cache, each move is one token forward: 1.39 GFLOPs, of which 0.74 is fixed (attention projections, the shared experts, routers, the board CNN and the head) and each routed expert per token adds 0.041. The weights take 11 GB in BF16.

## Training

![Main-eval CE over training against the scaling-law forecast, and the gap to Maia-3 79M over training](docs/figures/training.png)

**Data.** Every Lichess month with clock annotations from May 2017 to August 2026 except July 2026: 111 months, 7.87B games, 616B tokens. Added to it are over-the-board games (TWIC, PGN Mentor and Lichess broadcasts, 5.1M games) and engine games (CCRL and TCEC, 4.6M games, under 1% of tokens). A training-time sampler draws games from rating-bucketed monthly shards and tokenizes them on the fly. Its weights form an *Elo ramp*: a game is drawn twice as often for every 200 rating points of its stronger player, formats keep their natural shares, and no game is used more than 8 times. That recipe won a series of data-mixing screens at small scale, mainly by improving the strongest players' cells.

**Recipe.** 143,051 steps of 524,288 tokens (75B tokens, 3.1e20 training FLOPs). NorMuon, a Muon variant, trains the weight matrices including all experts; Adam trains embeddings, the head, routers, the board CNN and the clock table, and steps on every step. The learning rate warms up for 2,013 steps, then decays linearly to 0.1% of its peak. Input and output embeddings are tied for the first 307 steps. There is no multi-token prediction. Matrix multiplies run in BF16 with FP32 master weights.

**Systems.** One node of 8 NVIDIA L40S (48 GB). Data parallel with every expert on every GPU and optimizer state sharded across them, micro-batches of four 16K-token rows per GPU, activation recompute, and fused Triton kernels for the experts. Median throughput was 131K tokens/s, about 19% of the GPUs' peak BF16 throughput, for 159 hours (6.6 days) of training in 2-day jobs that resume from checkpoints taken every 1,024 steps.

**Result.** The main eval finished at 1.2533 nats, against 1.3170 for the best model of the scaling sweep and a forecast of 1.2856. On the benchmark, the run passed Maia-3 79M (beyond its interval) only at 70B of 75B tokens, during the last stretch of the learning-rate decay. A second anneal then took the final checkpoint through 1B more tokens of recent months, with a fresh optimizer and a peak learning rate of 0.05 (where the main run's schedule was at 99%). It improved every format: main eval 1.2505 (−0.0028), benchmark CE −0.0026 [−0.0033, −0.0019]. A hotter anneal (peak 0.2) gained nothing.

## Scaling laws

![Isoflop curves for dense and MoE models at three training budgets](docs/figures/scaling.png)

Before the big run we trained 45 models, dense and MoE with the same recipe, at three budgets: 6.3e17, 1.9e18 and 6.2e18 training FLOPs, five to seven sizes each (8 to 24 blocks, widths 384 to 1280), and 11 repeats with a second seed (seed-to-seed standard deviation 0.0029 nats). The MoE layout is the big run's: 256 experts, 16 per token, plus a shared expert. For each family we fit

L(N, D) = E + A (N / 10⁷)^−α + B (D / 10⁸)^−β

with N the active non-embedding parameters and D the training tokens:

| family | E | α | β | fit error (rms) |
|---|---|---|---|---:|
| MoE | 1.27 [1.26, 1.28] | 0.54 [0.42, 0.67] | 1.06 [1.01, 1.11] | 0.0026 |
| dense | 1.27 [1.26, 1.29] | 0.54 [0.44, 0.66] | 1.03 [0.95, 1.10] | 0.0020 |

(90% intervals.) The compute-optimal size grows as C^0.66 for both. Matching the MoE's loss takes 2.17x [1.99, 2.33], 2.31x [2.06, 2.62] and 2.50x [1.99, 3.32] as much dense compute at the three budgets, although the gap between the two families' best models narrows in nats (0.033, 0.025, 0.017).

The law forecast 1.2856 for the big run; it scored 1.2533, 0.032 better: twelve times the fit's rms error, and below the fitted floor E. We do not know why yet. The big run's recipe changes (no multi-token prediction, decay to 0.1%, the router fixes) do not help at small scale: the final recipe re-run at the smallest budget scores 0.0087 worse than the sweep's model. The likelier reading is that a 50x extrapolation in compute exceeds what this fit can pin down (B and β are correlated at 0.98). Memory, not the law, set the big run's size: the law's compute-optimal model at 3.1e20 FLOPs would have 1.8B active parameters trained on 28B tokens (15 per parameter), while 48 GB GPUs capped us at 0.69B active, trained on 109 tokens per parameter.

## Cheaper inference

![CE against the number of routed experts per token, and the gain from tree search against training compute](docs/figures/inference.png)

**Use fewer experts.** The big run computes 16 routed experts per token, but the best 8 by router score carry nearly everything: keeping only those (dropping the rest, without renormalizing the gates) costs +0.0011 nats on the benchmark, 6 cost +0.0025 and 4 cost +0.0095. A 0.5B-token fine-tune that routes through only the best K, distilled from the 16-expert model, recovers about two thirds of the loss at K = 4 and three quarters at K = 2. Below about 0.75 GFLOPs the fixed cost dominates and only smaller models help.

**Distill into smaller models.** Training small MoE students on a 50/50 mix of the played move and the big run's predicted distribution beats the same fine-tune on played moves alone, and the gain grows with tokens: for the 0.52-GFLOP student, −0.0017 nats at 0.25B tokens, −0.0031 at 0.75B and −0.0063 [−0.0069, −0.0056] at 1.5B. Teacher-only targets and temperature 2 did not help.

**Search buys little at this scale.** Our tree search over the model's own move, outcome and thinking-time predictions (in [search/](search/README.md)) helps less the better the model: at 128 simulations, the MoE sweep's best models gained 0.022, 0.019 and 0.013 nats at 6.3e17, 1.9e18 and 6.2e18 training FLOPs, and the big run gains 0.0055 [0.0038, 0.0070], almost all of it for players rated 2000 and above. Five simulations on the annealed model buy 0.0025 nats at 6x the compute.

## What we learned

![Small-scale ablations of training choices](docs/figures/ablations.png)

- **Router collapse at width 1536.** The first attempt at the big run collapsed during warmup: in the first MoE layer, the attention output feeding the router came to be dominated by one direction shared by all tokens, routing became nearly the same for every token, and up to 116 of 256 experts starved. The MoE sweep, at widths up to 1152, never showed it. Halving the router's learning rate only delayed it. What fixed it was subtracting a running mean from the router's input and stepping the Adam-trained parameters (router included) on every step instead of every other step, the modded-nanoGPT default. Both cost about 0.004-0.006 nats at small scale, where routers are healthy.

![Starved experts in the first MoE layer over the first 3,125 steps: first attempt against the final run](docs/figures/router.png)

- **A NaN at step 60,081.** One token had all 16 selected router scores near zero (summing to 6e-20), and the backward pass of the gate normalization overflowed. Flooring that sum at 1e-12 fixed it; the floor only touches tokens whose scores sum below it (about 0.2% of that layer's tokens at that step), and the run resumed from step 59,392. The early routers' weights had grown about 15x over training; the next run computes the gate normalization in log space instead.
- **Inherited defaults deserve an audit.** modded-nanoGPT is tuned for short GPT-2 runs. One change at a time at small scale (figure above): multi-token prediction and the 5% learning-rate floor were worth dropping, halving weight decay hurt, and the time-control tokens are worth keeping even with the clock features present. Each difference is within about two seed standard deviations, so these are directions, not precise sizes.
- **Ratings at every token: no gain.** Our use of the header ratings fades over a game while Maia-3's does not, so we tried adding both ratings to every token. It sped up early training and gained nothing by the end (+0.0004 to +0.0055 across five variants). The fading is the moves themselves revealing a player's strength.
- **A low-learning-rate second anneal helps; a hot one does not.** See [Training](#training).

## Next

- **Next run (planned).** MoE 1.2B active / ~9.2B total (24 blocks, width 2048, the first three blocks dense), 112B tokens, about 18 days on the same 8-GPU node with experts sharded across GPUs. At small scale, three dense first blocks cost nothing where the current router fixes cost 0.010 nats, and a constant-then-cooldown schedule was 0.026 nats worse than linear decay, so the run keeps the decay.
- **Strong-player data limits tokens.** Past about 110B tokens, the Elo ramp must either repeat games of players rated 2400+ more often or give them a smaller share, and both cost on the strongest cells. Beyond the next run, compute should go to parameters (GPUs with more memory) and to more strong-player data, such as more over-the-board games, rather than to more tokens.
- **Distillation from the next run** into students that beat Maia-3 23M at a fraction of its cost.

## Reproducing

The code is in two directories; training data, checkpoints and per-run results live outside the repository.

| Path | What it is |
|---|---|
| [scripts/chessdata.py](scripts/chessdata.py), [scripts/extdata.py](scripts/extdata.py) | Build the structured game stores from Lichess monthly PGN dumps and over-the-board / engine collections |
| [scripts/chessmix.py](scripts/chessmix.py), [scripts/chess_vocab.py](scripts/chess_vocab.py) | Training-time sampler (data mixing policies, tokenization, packing) and the token vocabulary |
| [scripts/modded_train.py](scripts/modded_train.py) | Distributed, resumable trainer, including fine-tuning from a checkpoint, distillation and fewer-experts training |
| [scripts/modded_medium_core.py](scripts/modded_medium_core.py), [scripts/modded_moe.py](scripts/modded_moe.py), [scripts/modded_smoe.py](scripts/modded_smoe.py), [scripts/modded_board.py](scripts/modded_board.py), [scripts/modded_arch.py](scripts/modded_arch.py) | Model, optimizers, MoE layer and its kernels, board CNN, architecture switches |
| [scripts/strateval.py](scripts/strateval.py), [scripts/eval_strat.py](scripts/eval_strat.py) | Build the 16-cell main evaluation and score a checkpoint on it |
| [scripts/modelexp.py](scripts/modelexp.py) | Freezes an experiment (code, data selection, recipe) into a study directory and runs and scores it on Slurm |
| [search/](search/README.md) | Inference engine with tree search, calibrated output policy and export |
| [docs/make_figures.py](docs/make_figures.py) | Regenerates every figure here from the result files |

Experiments are frozen before they run: `modelexp.py plan` copies the trainer source, data selection and recipe tables into a study directory with their hashes, and the trainer refuses to resume a run whose source or settings changed. The big run's study is `results/recipe10x/bigfix-24x1536d75m4shipv2nf-c8s200f0v4`. The Maia-3 benchmark scoring code lives with its results in `results/recipe10x/maia3-bench`. `results/` is a link to the group's storage and is not tracked. Figures: `.venv/bin/python docs/make_figures.py`.

Code derived from modded-nanoGPT is under its MIT license ([scripts/modded_medium_LICENSE](scripts/modded_medium_LICENSE)); the search engine is under [search/ALLIE_LICENSE](search/ALLIE_LICENSE).
