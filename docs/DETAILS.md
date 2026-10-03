# Allie: details

Numbers behind the [README](../README.md). CE is in nats; intervals are 95%, bootstrapping whole games (2,000 draws), and differences are paired on identical positions.

## Blitz benchmark

Allie-v3.0 is the final training checkpoint; Allie-v3.0 (annealed) continues it with a second anneal (see [Training](#training)). Differences on the same 80,000 positions:

| Allie model minus | Allie-v3.0: CE | top-1 (pp) | Allie-v3.0 (annealed): CE | top-1 (pp) |
|---|---:|---:|---:|---:|
| Maia-3 5M | −0.0899 [−0.0938, −0.0863] | +2.31 [+2.07, +2.54] | −0.0925 [−0.0964, −0.0887] | +2.33 [+2.10, +2.59] |
| Maia-3 23M | −0.0419 [−0.0454, −0.0387] | +0.92 [+0.71, +1.14] | −0.0444 [−0.0480, −0.0412] | +0.95 [+0.75, +1.17] |
| Maia-3 79M | −0.0213 [−0.0245, −0.0184] | +0.46 [+0.24, +0.67] | −0.0238 [−0.0272, −0.0208] | +0.48 [+0.27, +0.70] |

## Where the differences come from

Allie-v3.0 minus Maia-3 79M, blitz unless stated (bullet, rapid and classical from 5,000 positions per rating band):

| Slice | positions | CE difference | top-1 difference (pp) |
|---|---:|---:|---:|
| bullet | 20,000 | −0.1412 [−0.1514, −0.1310] | +3.33 [+2.81, +3.83] |
| rapid | 20,000 | −0.0163 [−0.0222, −0.0102] | +0.53 [+0.11, +0.94] |
| classical | 20,000 | −0.0208 [−0.0278, −0.0144] | +0.45 [−0.00, +0.90] |
| mover has under 10 s left | 2,852 | −0.2229 [−0.2627, −0.1823] | +3.47 [+1.88, +4.91] |
| mover has 60-120 s left | 15,719 | +0.0107 [+0.0034, +0.0182] | −0.53 [−1.06, −0.02] |
| plies 40-59 | 15,997 | +0.0071 [+0.0003, +0.0137] | −0.09 [−0.62, +0.42] |
| Maia-3's protocol: ply > 20, at least 30 s left | 48,987 | −0.0057 [−0.0095, −0.0019] | +0.12 [−0.16, +0.40] |

The rating plot reweights every scored blitz move of the main evaluation (402,108 positions in 6,247 games) to the natural player mix. In 100-point bins of game rating, Allie-v3.0's CE interval is below Maia-3 79M's in 19 of 23; the exceptions are 600-700, 700-800, 1000-1100 and 2800-2900 (7 games), and only the last has a higher point estimate. Its accuracy is lower only at 800-900 (−0.12 pp) and 2800-2900 (−0.33 pp), both within noise.

## Evaluation

- **Benchmark positions.** The main evaluation's 402,108 scored blitz moves, rebuilt into games (every target move replays as legal); a fixed random sample of 20,000 per rating band, drawn before any model ran. Maia-3 ([paper](https://arxiv.org/abs/2605.19091), [models](https://huggingface.co/collections/MaiaChess/maia3), [code](https://github.com/CSSLab/maia-chess), AGPL-3.0) runs unmodified from its public 5M, 23M and 79M checkpoints. It receives the current board and the 7 previous ones in its side-to-move orientation, and the mover's and opponent's ratings; our scorer reproduces the authors' engine move for move and probability for probability. Allie receives the whole game, both ratings and the clock, in one forward pass per game.
- **Main evaluation.** 16 cells of format × mover rating with about 100,000 scored moves each (1.55M moves, 26,278 games; classical ≥2400 has all 37,180 available moves). Rated human games only; BOT players and games that leak into validation are excluded. CE is over all 1,968 move tokens without legal masking, and the macro is the unweighted mean of the 16 cells.

## The model

- Tokens: an 11-token header (start, base time, increment, each rating as four digits) and one token per move, from-square/to-square plus promotion. Games are packed into 16K-token rows with attention masked at game boundaries.
- Inputs added at every position: the mover's and opponent's time left and the mover's previous thinking time (Fourier features of log seconds; 10% of training games omit them), and a board CNN (13 piece planes, three 3×3 convolutions with 32 channels, a 1×1 squeeze, castling and en-passant features).
- Trunk: 24 blocks of width 1536, 24 heads of 64. From modded-nanoGPT: QK-normalized attention with rotary positions on half of each head, gated attention outputs, value embeddings, U-net skips, two re-injected input embeddings, soft-capped logits, embeddings tied for the first 307 steps.
- MoE (blocks 2-24): 256 routed SwiGLU experts of hidden width 192 plus a shared expert of width 1024 (active width 16 × 192 + 1024 = 4096, matching the dense block 1). Sigmoid router scores; the top 16 of score plus bias are used, gates renormalized to sum to 4. Biases are reset every step by quantile balancing (Kimi K3), with a small sequence-level balance loss; no capacity limit, no dropped tokens. Router input centred by a running mean (decay 0.9); gate sum floored at 1e-12.
- Head: the next move (the objective), plus the move's thinking time (63 bins) and the game result from the mover's side, each at loss weight 0.2.
- Cost per move with a cached game: 1.39 GFLOPs, of which 0.74 is fixed (attention, shared experts, routers, board CNN, head) and each routed expert adds 0.041.

## Training

- **Data.** 111 Lichess months (May 2017 to August 2026 without July 2026; 7.87B games, 616B tokens), plus over-the-board games (TWIC, PGN Mentor, Lichess broadcasts; 5.1M) and engine games (CCRL, TCEC; 4.6M, under 1% of tokens). The sampler draws games from rating-bucketed monthly shards and tokenizes on the fly. Its *Elo ramp* table doubles a game's weight per 200 points of its stronger player, keeps formats at their natural shares and caps any game at 8 uses; it won a series of small-scale data-mixing screens, mainly on the strongest players' cells.
- **Optimization.** NorMuon (a Muon variant) for all weight matrices including experts; Adam, stepping every step, for embeddings, head, routers, board CNN and clock table. Warmup 2,013 steps, then linear decay to 0.1% of the peak learning rate. No multi-token prediction. BF16 matrix multiplies with FP32 master weights.
- **Systems.** 8 NVIDIA L40S (48 GB) on one node; data parallel with optimizer state sharded, four 16K-token rows per GPU per micro-batch, activation recompute, fused Triton expert kernels. Median 131K tokens/s, about 19% of peak BF16 throughput; 159 hours in 2-day jobs resuming from checkpoints every 1,024 steps.
- **Second anneal (Allie-v3.0 annealed).** From Allie-v3.0 with a fresh optimizer: 1B tokens of January 2024 to June 2026 plus August 2026, peak learning rate 0.05 (the main schedule's value at 99%), decaying over the last 30%. Main eval −0.0028, every format better; benchmark −0.0026 [−0.0033, −0.0019]. Peak 0.2 gained nothing.

## Scaling laws

45 runs at 6.3e17, 1.9e18 and 6.2e18 training FLOPs, dense and MoE, five to seven sizes per budget (8 to 24 blocks, widths 384 to 1280), plus 11 second-seed repeats (seed standard deviation 0.0029). Fits of L(N, D) = E + A (N/10⁷)^−α + B (D/10⁸)^−β, with N active non-embedding parameters and D tokens (90% intervals):

| family | E | α | β | rms error |
|---|---|---|---|---:|
| MoE | 1.27 [1.26, 1.28] | 0.54 [0.42, 0.67] | 1.06 [1.01, 1.11] | 0.0026 |
| dense | 1.27 [1.26, 1.29] | 0.54 [0.44, 0.66] | 1.03 [0.95, 1.10] | 0.0020 |

Compute multiplier of MoE over dense: 2.17x [1.99, 2.33], 2.31x [2.06, 2.62] and 2.50x [1.99, 3.32] at the three budgets. The compute-optimal size grows as C^0.66. Allie-v3.0 (N = 0.69B, D = 75B, 109 tokens per parameter) was forecast at 1.2856 and scored 1.2533. B and β are correlated at 0.98 in the fit.

Allie-v3.0's recipe re-run at the sweep's sizes, minus the sweep recipe at the same size (main eval / its ≥2400 cells, nats): +0.0082 / +0.0104 at 6.3e17 FLOPs (12 blocks, width 512), +0.0058 / +0.0086 and +0.0042 / +0.0074 at 6.2e18 (18 × 896 and 20 × 1024). The offset shrinks by about 0.003 per decade of compute and extrapolates to about −0.002 at Allie-v3.0's 3.1e20 FLOPs, so the 0.032 below the forecast comes from the law's form (its floor), not from the recipe.

## Inference cost

- **Fewer experts, no retraining** (Allie-v3.0; the dropped experts' gates are left out, not renormalized): K = 12, 8, 6, 4, 2 cost +0.0001, +0.0011, +0.0025, +0.0095, +0.0544 nats. Renormalizing the kept gates is far worse (+0.0178 at K = 8).
- **Search** (20,000 positions, 128 simulations with a per-model calibration fit on separate July 2026 games): the gain is +0.0055 [+0.0038, +0.0070] for Allie-v3.0, almost all from players rated 2000+; five simulations on Allie-v3.0 (annealed) buy 0.0025 nats at 6x the compute.

## Training stability

- **Router collapse.** In the first attempt, the first block's attention output came to carry a large near-constant vector, so the first MoE layer's router logits became a fixed per-expert offset; the capped balancing biases could not compensate, and starved experts rose from step ~1,200 to 116 of 256. Short test runs from scratch with the same data order isolated the fix: half the router learning rate plus a shorter multi-token phase collapsed 100-300 steps later; centring alone left the load uneven (largest expert load 3.8x the mean at step 2,600, against 2.1x with both fixes); centring plus Adam on every step stayed healthy. The final run uses all three, drops multi-token prediction (whose phase had inflated training loss by 0.4 nats) and decays to 0.1% instead of 5%; it stayed at 0-2 starved experts.
- **NaN at step 60,081.** One token's 16 selected sigmoid scores summed to 5.7e-20; the backward pass of the gate normalization overflowed into non-finite gradients. A floor of 1e-12 on that sum touches about 0.2% of that layer's tokens and leaves every other token bitwise unchanged. The run resumed from step 59,392, losing about 50 minutes. The next run is planned to normalize gates in log space instead.

## Reproducing

| Module | What it is |
|---|---|
| `allie.data.fetch`, `fastbuild`, `store`, `external`, `annotate` | Download Lichess months and build the store (games bucketed by format and rating, pre-shuffled into shards); the same for over-the-board and engine games, deduplicated against the evaluation games |
| `allie.data.pin`, `history`, `inventory` | Freeze a data selection, count its games per bucket, build sampling tables such as the Elo ramp |
| `allie.data.mix`, `vocab`, `packed` | Training-time sampler (mixing policies, on-the-fly tokenization, packing), the vocabulary, the original packed validation rows |
| `allie.model.network`, `nanogpt`, `attention`, `moe`, `moe_kernels`, `board`, `shard`, `arch` | The model (`nanogpt` is the modded-nanoGPT-derived core with its optimizers), the MoE layer and its Triton kernels, the board CNN, expert sharding, architecture switches |
| `allie.train.trainer`, `schedule`, `checkpoints`, `state`, `runtime`, `provenance` | Resumable distributed trainer (`--init-from` continues from a checkpoint's weights with a fresh optimizer, as the second anneal did), learning-rate schedule, checkpoints, and the source hashes a checkpoint records |
| `allie.eval.build`, `score`, `maia3` | Build and score the main evaluation; the Maia-3 benchmark |
| `allie.search` | Tree-search engine; `moe_oracle` serves MoE checkpoints to it |
| `allie.experiments.modelexp`, `isoflop`, `dmix`, `readout` | Frozen studies on Slurm, scaling-law fits, the sweep readout |
| `allie.cli` | `allie-train` (a config file to a torchrun of the trainer) and `allie-eval` |
| `configs/` | `allie-v3.0.json` (the trainer's arguments for Allie-v3.0), `recipes/` (its sampling table), `allie-v3.0/` (the run's record: study plan, round file, launchers, data pin, game counts) |
| `tests/` | `test_checks.py` runs each check in `tests/checks/` as its own program; `tests/search/` tests the engine |
| `analysis/` | The analysis scripts behind the search and frontier results, as run |

**Experiments.** A round file declares runs; `allie-exp plan ROUND WAVE` freezes the package (`source/allie/`), recipe tables, data pin and history counts into a study directory with their hashes, and `allie-exp submit` / `task` run it on Slurm: the trainer under torchrun, resumed across jobs, then the evaluator. The trainer refuses to resume a run whose source or settings changed.

**Checkpoints from before the package layout** (Allie-v3.0 included) record flat source file names (`modded_medium.py`, ...). The evaluator, the Maia-3 scorer and the MoE search oracle load them from their run's frozen flat source (`--source`); a checkpoint trained with the package loads with the package, checked against `allie.train.provenance`.

**Not in git.** Training data, checkpoints and per-run results (scores, logs, reports) live on group storage under `ALLIE_DATA` and `results/`.
