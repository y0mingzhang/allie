# Allie: details

Numbers behind the [README](../README.md). CE is in nats; intervals are 95%, bootstrapping whole games (2,000 draws), and differences are paired on identical positions.

## Blitz benchmark frontier

| Model | GFLOPs/move | CE | top-1 (%) | vs Maia-3 79M: CE | vs 79M: top-1 (pp) |
|---|---:|---:|---:|---:|---:|
| Maia-3 5M | 0.60 | 1.2955 | 56.95 | +0.0686 [+0.0657, +0.0715] | −1.85 [−2.06, −1.64] |
| Maia-3 23M | 2.40 | 1.2475 | 58.34 | +0.0206 [+0.0189, +0.0223] | −0.47 [−0.63, −0.29] |
| Maia-3 79M | 9.23 | 1.2269 | 58.80 | | |
| Allie big run, after a second anneal | 1.39 | 1.2030 | 59.29 | −0.0238 [−0.0272, −0.0208] | +0.48 [+0.27, +0.70] |
| Allie big run, final checkpoint | 1.39 | 1.2056 | 59.26 | −0.0213 [−0.0245, −0.0184] | +0.46 [+0.24, +0.67] |
| final, best 8 of 16 experts, no training | 1.06 | 1.2067 | 59.27 | −0.0202 [−0.0235, −0.0172] | +0.47 [+0.26, +0.69] |
| final, best 6 of 16 experts, no training | 0.98 | 1.2081 | 59.21 | −0.0188 [−0.0221, −0.0157] | +0.41 [+0.19, +0.63] |
| fine-tuned to use 4 of 16 experts | 0.90 | 1.2088 | 59.14 | −0.0180 [−0.0215, −0.0149] | +0.34 [+0.12, +0.56] |
| fine-tuned to use 2 of 16 experts | 0.82 | 1.2205 | 58.75 | −0.0064 [−0.0100, −0.0033] | −0.06 [−0.27, +0.17] |
| student, MoE 261M active / 2.07B total | 0.52 | 1.2469 | 58.09 | +0.0201 [+0.0168, +0.0231] | −0.71 [−0.92, −0.50] |
| student, MoE 181M active / 1.42B total | 0.36 | 1.2545 | 57.92 | +0.0276 [+0.0240, +0.0309] | −0.89 [−1.12, −0.67] |

Relative to Maia-3 23M, the fine-tuned 2-expert model has lower CE and higher accuracy (−0.0270 nats, +0.41 pp); relative to Maia-3 5M, so does the 0.36-GFLOP student (−0.0410, +0.96 pp).

## Where the differences come from

Final checkpoint minus Maia-3 79M, blitz unless stated (bullet, rapid and classical from 5,000 positions per rating band):

| Slice | positions | CE difference | top-1 difference (pp) |
|---|---:|---:|---:|
| bullet | 20,000 | −0.1412 [−0.1514, −0.1310] | +3.33 [+2.81, +3.83] |
| rapid | 20,000 | −0.0163 [−0.0222, −0.0102] | +0.53 [+0.11, +0.94] |
| classical | 20,000 | −0.0208 [−0.0278, −0.0144] | +0.45 [−0.00, +0.90] |
| mover has under 10 s left | 2,852 | −0.2229 [−0.2627, −0.1823] | +3.47 [+1.88, +4.91] |
| mover has 60-120 s left | 15,719 | +0.0107 [+0.0034, +0.0182] | −0.53 [−1.06, −0.02] |
| plies 40-59 | 15,997 | +0.0071 [+0.0003, +0.0137] | −0.09 [−0.62, +0.42] |
| Maia-3's protocol: ply > 20, at least 30 s left | 48,987 | −0.0057 [−0.0095, −0.0019] | +0.12 [−0.16, +0.40] |

The rating plot reweights every scored blitz move of the main evaluation (402,108 positions in 6,247 games) to the natural player mix. In 100-point bins of game rating, the final checkpoint's CE interval is below Maia-3 79M's in 19 of 23; the exceptions are 600-700, 700-800, 1000-1100 and 2800-2900 (7 games), and only the last has a higher point estimate. Its accuracy is lower only at 800-900 (−0.12 pp) and 2800-2900 (−0.33 pp), both within noise.

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
- **Second anneal.** From the final checkpoint with a fresh optimizer: 1B tokens of January 2024 to June 2026 plus August 2026, peak learning rate 0.05 (the main schedule's value at 99%), decaying over the last 30%. Main eval −0.0028, every format better; benchmark −0.0026 [−0.0033, −0.0019]. Peak 0.2 gained nothing.

## Scaling laws

45 runs at 6.3e17, 1.9e18 and 6.2e18 training FLOPs, dense and MoE, five to seven sizes per budget (8 to 24 blocks, widths 384 to 1280), plus 11 second-seed repeats (seed standard deviation 0.0029). Fits of L(N, D) = E + A (N/10⁷)^−α + B (D/10⁸)^−β, with N active non-embedding parameters and D tokens (90% intervals):

| family | E | α | β | rms error |
|---|---|---|---|---:|
| MoE | 1.27 [1.26, 1.28] | 0.54 [0.42, 0.67] | 1.06 [1.01, 1.11] | 0.0026 |
| dense | 1.27 [1.26, 1.29] | 0.54 [0.44, 0.66] | 1.03 [0.95, 1.10] | 0.0020 |

Compute multiplier of MoE over dense: 2.17x [1.99, 2.33], 2.31x [2.06, 2.62] and 2.50x [1.99, 3.32] at the three budgets. The compute-optimal size grows as C^0.66. The big run (N = 0.69B, D = 75B, 109 tokens per parameter) was forecast at 1.2856 and scored 1.2533. B and β are correlated at 0.98 in the fit.

## Cheaper inference

- **Fewer experts, no training** (final checkpoint; the dropped experts' gates are left out, not renormalized): K = 12, 8, 6, 4, 2 cost +0.0001, +0.0011, +0.0025, +0.0095, +0.0544 nats. Renormalizing the kept gates is far worse (+0.0178 at K = 8).
- **Fewer-experts fine-tunes:** 0.5B tokens of recent months, routing through only the best K and distilled 50/50 toward the 16-expert model: K = 4 1.2088 (from 1.2151), K = 2 1.2205 (from 1.2600).
- **Students** (MoE 261M active / 2.07B total, 0.52 GFLOPs): distillation minus a played-moves-only control on identical rows, −0.0017 [−0.0023, −0.0011], −0.0031 [−0.0038, −0.0023] and −0.0063 [−0.0069, −0.0056] at 0.25B, 0.75B and 1.5B tokens. Teacher-only targets and temperature 2 did not help.
- **Search** (20,000 positions, 128 simulations with a per-model calibration fit on separate July 2026 games): the gain is +0.0055 [+0.0038, +0.0070] for the big run, almost all from players rated 2000+; five simulations on the annealed model buy 0.0025 nats at 6x the compute.

## Training stability

- **Router collapse.** In the first attempt, the first block's attention output came to carry a large near-constant vector, so the first MoE layer's router logits became a fixed per-expert offset; the capped balancing biases could not compensate, and starved experts rose from step ~1,200 to 116 of 256. Short test runs from scratch with the same data order isolated the fix: half the router learning rate plus a shorter multi-token phase collapsed 100-300 steps later; centring alone left the load uneven (largest expert load 3.8x the mean at step 2,600, against 2.1x with both fixes); centring plus Adam on every step stayed healthy. The final run uses all three, drops multi-token prediction (whose phase had inflated training loss by 0.4 nats) and decays to 0.1% instead of 5%; it stayed at 0-2 starved experts.
- **NaN at step 60,081.** One token's 16 selected sigmoid scores summed to 5.7e-20; the backward pass of the gate normalization overflowed into non-finite gradients. A floor of 1e-12 on that sum touches about 0.2% of that layer's tokens and leaves every other token bitwise unchanged. The run resumed from step 59,392, losing about 50 minutes. The next run is planned to normalize gates in log space instead.

## Reproducing

| Path | What it is |
|---|---|
| `scripts/hf_fetch.py`, `scripts/fastbuild.py`, `scripts/chessdata.py` | Download Lichess months and build the store: games bucketed by format and rating, pre-shuffled into shards |
| `scripts/extdata.py` | The same store for over-the-board and engine games, deduplicated against the evaluation games |
| `scripts/datapin.py`, `scripts/history_counts.py`, `scripts/pool_inventory.py` | Freeze a data selection, count its games per bucket, build sampling tables such as the Elo ramp |
| `scripts/chessmix.py`, `scripts/chess_vocab.py` | Training-time sampler (mixing policies, on-the-fly tokenization, packing) and the vocabulary |
| `scripts/modded_train.py` | Resumable distributed trainer; `--init-from` fine-tunes, `--kd-teacher` distills, arch `moe_keep` trains fewer-experts models |
| `scripts/modded_*.py` | Model, optimizers, MoE layer and Triton kernels, board CNN, architecture switches |
| `scripts/strateval.py`, `scripts/eval_strat.py` | Build and score the main evaluation |
| `scripts/modelexp.py` | Freeze a study (source, data, recipe, hashes), run and score it on Slurm |
| `scripts/test_*.py` | Unit and equivalence tests (some need a GPU or real data rows) |
| `runs/` | The released model's frozen study, and the anneal, fine-tune and student launchers |
| `bench/` | Maia-3 benchmark scoring, fewer-experts inference, rating plot, scaling readout, search on the MoE |
| `search/` | Inference engine with tree search, calibration and export |
| `docs/make_figures.py` | All README figures from the result files |

**Workflow.** Build the stores (`hf_fetch.py`, then `fastbuild.py build` and `finalize` per month; `extdata.py` for the other sources), pin them (`datapin.py`) and build the sampling table (`pool_inventory.py`). Build the evaluation set once (`strateval.py`). Declare runs in a round file, freeze them with `modelexp.py plan ROUND WAVE` and launch with `modelexp.py submit` (a Slurm array) or `modelexp.py task` on a node. A study runs `modded_train.py` under `torchrun`, resumes it across jobs and scores it with `eval_strat.py` at the end; the trainer refuses to resume a run whose source or settings changed. The released model's exact trainer arguments are in `runs/bigrun/resume-config.json`.

**Not in git.** Training data, checkpoints and per-run results (scores, logs, reports) live on the group's storage under `ALLIE_DATA` and `results/`.
