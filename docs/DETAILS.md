# Allie: details

> Researched and written by AI agents (Claude Code and OpenAI Codex), with Yiming Zhang setting goals and making calls.

The numbers behind the [README](../README.md).

CE is in nats. Intervals are 95%, from 2,000 bootstrap draws over whole games. Differences between models are paired on identical positions.

## Blitz benchmark

Allie-v3.0 is the final training checkpoint. Allie-v3.0 (annealed) continues it with a second anneal (see [Training](#training)).

Differences on the same 80,000 positions:

| Allie model minus | Allie-v3.0: CE | top-1 (pp) | Allie-v3.0 (annealed): CE | top-1 (pp) |
|---|---:|---:|---:|---:|
| Maia-3 5M | −0.0899 [−0.0938, −0.0863] | +2.31 [+2.07, +2.54] | −0.0925 [−0.0964, −0.0887] | +2.33 [+2.10, +2.59] |
| Maia-3 23M | −0.0419 [−0.0454, −0.0387] | +0.92 [+0.71, +1.14] | −0.0444 [−0.0480, −0.0412] | +0.95 [+0.75, +1.17] |
| Maia-3 79M | −0.0213 [−0.0245, −0.0184] | +0.46 [+0.24, +0.67] | −0.0238 [−0.0272, −0.0208] | +0.48 [+0.27, +0.70] |

### The original Allie

The original Allie ([ICLR 2025](https://arxiv.org/abs/2410.03893)) is scored from its raw policy, on the same positions, in its own input format: both ratings and a time-control token before the moves, with no clock.

- **Checkpoint.** `medium/best.pt` from [yimingzhang/allie-models](https://huggingface.co/datasets/yimingzhang/allie-models): a GPT-2-medium shape with 305M parameters, trained on 2022 Lichess blitz. It is a retrain of the paper's recipe, not the exact model behind the paper's stored outputs: it reproduces its training run's validation loss (1.3535) and the paper's test accuracy (55.73% against 55.7% reported).
- **Benchmark.** CE 1.2848 [1.2748, 1.2941], top-1 57.06% [56.70, 57.42]. By rating band, CE 1.4256, 1.2947, 1.2320 and 1.1870 from <1400 to ≥2400. On Maia-3's protocol (ply > 20, at least 30 s left): 1.2778 and 57.36%.
- **Main evaluation.** 1.3526, against Allie-v3.0's 1.2533. Only blitz is in its training distribution: bullet, rapid and classical time controls map to its unknown token.
- **Compute.** 0.61 GFLOPs per move with a key-value cache, counted as for the other models. Its released decoder has no cache and re-reads the game, about 28 GFLOPs per move at the benchmark's mean ply; its time-adaptive search averages 50 such passes per move.
- **Mapping.** 3 of 6,191 benchmark games (35 positions) have a time control outside its 24 blitz tokens and get the unknown token. The model clamps ratings to 500-3000.

Our previous model, Allie v2 (a dense 1.42B Qwen3 on the same token format, without clock inputs), scores 1.2343 [1.2247, 1.2439] and 58.43% [58.06, 58.79] on the benchmark at 2.85 GFLOPs per move.

## Where the differences come from

Allie-v3.0 minus Maia-3 79M, on slices of the benchmark. Bullet, rapid and classical use 5,000 positions per rating band; the other rows are blitz.

| Slice | positions | CE difference | top-1 difference (pp) |
|---|---:|---:|---:|
| bullet | 20,000 | −0.1412 [−0.1514, −0.1310] | +3.33 [+2.81, +3.83] |
| rapid | 20,000 | −0.0163 [−0.0222, −0.0102] | +0.53 [+0.11, +0.94] |
| classical | 20,000 | −0.0208 [−0.0278, −0.0144] | +0.45 [−0.00, +0.90] |
| mover has under 10 s left | 2,852 | −0.2229 [−0.2627, −0.1823] | +3.47 [+1.88, +4.91] |
| mover has 60-120 s left | 15,719 | +0.0107 [+0.0034, +0.0182] | −0.53 [−1.06, −0.02] |
| plies 40-59 | 15,997 | +0.0071 [+0.0003, +0.0137] | −0.09 [−0.62, +0.42] |
| Maia-3's protocol: ply > 20, at least 30 s left | 48,987 | −0.0057 [−0.0095, −0.0019] | +0.12 [−0.16, +0.40] |

The rating plot uses every scored blitz move of the main evaluation: 402,108 positions in 6,247 games. They are reweighted to the natural player mix and binned by game rating, in 100-point bins.

Allie-v3.0's CE interval is below Maia-3 79M's in 19 of 23 bins. The exceptions are 600-700, 700-800, 1000-1100 and 2800-2900 (7 games); only the last has a higher point estimate.

Its accuracy is lower in only two bins, both within noise: 800-900 (−0.12 pp) and 2800-2900 (−0.33 pp).

## Search

The search runs over the model's own move, outcome and thinking-time predictions. It used 20,000 benchmark positions, 5,000 per rating band. Each budget's output is calibrated on separate July 2026 games, disjoint from the benchmark.

| Allie-v3.0 | GFLOPs/move | CE | CE gain | top-1 gain (pp) |
|---|---:|---:|---:|---:|
| raw policy | 1.40 | 1.2216 | | |
| 5 simulations | 8.30 | 1.2189 | +0.0026 [+0.0014, +0.0039] | +0.15 [−0.08, +0.37] |
| 128 simulations | 176 | 1.2161 | +0.0055 [+0.0038, +0.0070] | +0.18 [−0.04, +0.40] |

On the same positions, Maia-3 5M, 23M and 79M score 1.3086, 1.2597 and 1.2395.

The gain comes from strong players. With 128 simulations it is −0.0001 below 1400, +0.0014 at 1400-2000, +0.0062 at 2000-2400 and +0.0143 at 2400 and above.

Search helps less as models grow. 128 simulations gain 0.022, 0.019 and 0.013 nats on the scaling sweep's best MoE models at its three budgets, and 0.0055 on Allie-v3.0.

## Evaluation

**Benchmark positions.**

- The positions are the main evaluation's 402,108 scored blitz moves, rebuilt into games. Every target move replays as legal.
- We drew a fixed random sample of 20,000 per rating band before any model ran.

**Maia-3.**

- Maia-3 ([paper](https://arxiv.org/abs/2605.19091), [models](https://huggingface.co/collections/MaiaChess/maia3), [code](https://github.com/CSSLab/maia-chess), AGPL-3.0) runs unmodified from its public 5M, 23M and 79M checkpoints.
- It receives the current board and the 7 previous ones, in its side-to-move orientation, and the mover's and opponent's ratings.
- Our scorer reproduces the authors' engine move for move and probability for probability.
- Allie receives the whole game, both ratings and the clock, in one forward pass per game.

**Main evaluation.**

- 16 cells of format × mover rating, with about 100,000 scored moves each: 1.55M moves in 26,278 games. Classical ≥2400 has all 37,180 available moves.
- Rated human games only. BOT players and games that leak into validation are excluded.
- CE is over all 1,968 move tokens, without legal masking. The macro is the unweighted mean of the 16 cells.

## The model

- **Tokens.** An 11-token header: start, base time, increment, and each rating as four digits. Then one token per move: from-square and to-square, plus promotion. Games are packed into 16K-token rows, with attention masked at game boundaries.
- **Clock inputs.** At every position: the mover's and opponent's time left, and the mover's previous thinking time, as Fourier features of log seconds. 10% of training games omit them.
- **Board input.** A small CNN: 13 piece planes, three 3×3 convolutions with 32 channels, a 1×1 squeeze, and castling and en-passant features.
- **Trunk.** 24 blocks of width 1536, with 24 heads of 64. From modded-nanoGPT: QK-normalized attention, rotary positions on half of each head, gated attention outputs, value embeddings, U-net skips, two re-injected input embeddings and soft-capped logits. Embeddings are tied for the first 307 steps.
- **MoE (blocks 2-24).** 256 routed SwiGLU experts of hidden width 192, plus a shared expert of width 1024. The active width is 16 × 192 + 1024 = 4096, matching the dense block 1.
- **Routing.** Sigmoid router scores; the top 16 of score plus bias are used, with gates renormalized to sum to 4. Biases are reset every step by quantile balancing (Kimi K3), with a small sequence-level balance loss. There is no capacity limit and no token is dropped. The router input is centred by a running mean (decay 0.9), and the gate sum is floored at 1e-12.
- **Head.** The next move is the objective. Side targets, each at loss weight 0.2: the move's thinking time (63 bins) and the game result from the mover's side.
- **Cost.** 1.39 GFLOPs per move with the game cached. Of this, 0.74 is fixed (attention, shared experts, routers, board CNN, head) and each routed expert adds 0.041.

## Training

**Data.**

- 111 Lichess months: May 2017 to August 2026, without July 2026. That is 7.87B games and 616B tokens.
- Over-the-board games (TWIC, PGN Mentor, Lichess broadcasts; 5.1M) and engine games (CCRL, TCEC; 4.6M, under 1% of tokens).
- The sampler draws games from rating-bucketed monthly shards and tokenizes them on the fly.
- Its *Elo ramp* table doubles a game's weight per 200 points of its stronger player, keeps formats at their natural shares, and caps any game at 8 uses. It won a series of small data-mixing screens, mainly on the strongest players' cells.

**Optimization.**

- NorMuon (a Muon variant) for all weight matrices, experts included.
- Adam, stepping every step, for the embeddings, head, routers, board CNN and clock table.
- Warmup of 2,013 steps, then linear decay to 0.1% of the peak learning rate. No multi-token prediction.
- BF16 matrix multiplies with FP32 master weights.

**Systems.**

- 8 NVIDIA L40S (48 GB) on one node. Data parallel, with optimizer state sharded.
- Four 16K-token rows per GPU per micro-batch, activation recompute, fused Triton expert kernels.
- Median 131K tokens/s, about 19% of peak BF16 throughput.
- 159 hours, in 2-day jobs that resume from checkpoints saved every 1,024 steps.

**Second anneal (Allie-v3.0 annealed).**

- Starts from Allie-v3.0 with a fresh optimizer.
- 1B tokens of January 2024 to June 2026, plus August 2026.
- Peak learning rate 0.05 (the main schedule's value at 99%), decaying over the last 30%. A peak of 0.2 gained nothing.
- Main eval −0.0028, with every format better. Benchmark −0.0026 [−0.0033, −0.0019].

## Scaling laws

The sweep has 45 runs at 6.3e17, 1.9e18 and 6.2e18 training FLOPs, dense and MoE. Each budget has five to seven sizes, from 8 to 24 blocks and widths 384 to 1280. Eleven second-seed repeats put the seed standard deviation at 0.0029.

We fit L(N, D) = E + A (N/10⁷)^−α + B (D/10⁸)^−β. N is active non-embedding parameters and D is tokens. Intervals are 90%:

| family | E | α | β | rms error |
|---|---|---|---|---:|
| MoE | 1.27 [1.26, 1.28] | 0.54 [0.42, 0.67] | 1.06 [1.01, 1.11] | 0.0026 |
| dense | 1.27 [1.26, 1.29] | 0.54 [0.44, 0.66] | 1.03 [0.95, 1.10] | 0.0020 |

- The MoE's compute multiplier over dense is 2.17x [1.99, 2.33], 2.31x [2.06, 2.62] and 2.50x [1.99, 3.32] at the three budgets.
- The compute-optimal size grows as C^0.66.
- B and β are correlated at 0.98 in the fit.
- Allie-v3.0 (N = 0.69B, D = 75B, 109 tokens per parameter) was forecast at 1.2856 and scored 1.2533.

**Recipe re-run.** We re-ran Allie-v3.0's recipe at the sweep's sizes and compared it with the sweep recipe at the same size. On the main eval and its ≥2400 cells:

| Budget | Shape | Main eval | ≥2400 cells |
|---|---|---:|---:|
| 6.3e17 FLOPs | 12 blocks × 512 | +0.0082 | +0.0104 |
| 6.2e18 FLOPs | 18 blocks × 896 | +0.0058 | +0.0086 |
| 6.2e18 FLOPs | 20 blocks × 1024 | +0.0042 | +0.0074 |

The offset shrinks by about 0.003 per decade of compute. It extrapolates to about −0.002 at Allie-v3.0's 3.1e20 FLOPs. So the 0.032 below the forecast comes from the law's form (its floor), not from the recipe.

## Ablations

One change at a time to an MoE with 39M active parameters, trained on 2.5B tokens with the same seed and data order. Each row is the change in main-eval CE against the unchanged run (1.3754); negative is better. One seed-to-seed standard deviation is 0.0029, so read directions, not sizes.

| Change | Main-eval CE | In Allie-v3.0 |
|---|---:|:---:|
| Decay the learning rate to 0.1% of peak, not 5% | −0.0019 | yes |
| No multi-token prediction | −0.0011 | yes |
| Both ratings at every token (best of 5 settings) | +0.0004 | |
| Half the weight decay | +0.0026 | |
| No time-control tokens | +0.0028 | |
| Adam on every step, not every other | +0.0043 | yes |
| Router-input centring | +0.0056 | yes |

The two router changes cost a little at this scale, where routers stay healthy. Allie-v3.0 keeps them because at width 1536 they prevent router collapse.

## Training stability

**Router collapse.**

- In the first attempt, the first block's attention output came to carry a large, near-constant vector. The first MoE layer's router logits became a fixed per-expert offset, and the capped balancing biases could not compensate.
- Starved experts rose from about step 1,200 to 116 of 256.
- Short test runs from scratch, with the same data order, isolated the fix:
  - Half the router learning rate plus a shorter multi-token phase collapsed 100-300 steps later.
  - Centring alone left the load uneven: the largest expert load was 3.8x the mean at step 2,600, against 2.1x with both fixes.
  - Centring plus Adam on every step stayed healthy.
- The final run uses all three. It also drops multi-token prediction, whose phase had inflated training loss by 0.4 nats, and decays to 0.1% instead of 5%. It stayed at 0-2 starved experts.

**NaN at step 60,081.**

- One token's 16 selected sigmoid scores summed to 5.7e-20, and the backward pass of the gate normalization overflowed into non-finite gradients.
- A floor of 1e-12 on that sum touches about 0.2% of that layer's tokens and leaves every other token bitwise unchanged.
- The run resumed from step 59,392, losing about 50 minutes. The next run is planned to normalize gates in log space instead.

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
| `tests/` | `test_checks.py` runs each check in `tests/checks/` as its own program; `test_equivalent.py` guards how Allie-v3.0 loads; `tests/search/` tests the engine |
| `analysis/` | The analysis scripts behind the search and frontier results, as run |

**Experiments.** A round file declares runs. `allie-exp plan ROUND WAVE` freezes the package (`source/allie/`), recipe tables, data pin and history counts into a study directory, with their hashes. `allie-exp submit` and `task` then run it on Slurm: the trainer under torchrun, resumed across jobs, then the evaluator. The trainer refuses to resume a run whose source or settings changed.

**Checkpoints from before the package layout** record flat source file names (`modded_medium.py`, ...). The evaluator, the Maia-3 scorer and the MoE search oracle load them from their run's frozen flat source (`--source`). Allie-v3.0 is the exception: it loads with the package, because this version of the package was checked to score it bitwise as its frozen source does (`allie.eval.score.EQUIVALENT`). A checkpoint trained with the package loads with the package, checked against `allie.train.provenance`.

**Not in git.** Training data, checkpoints and per-run results (scores, logs, reports) live on group storage under `ALLIE_DATA` and `results/`.
