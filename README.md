# Allie

> **Disclaimer:** I did not do any of this research. The experiments, code, analysis and this write-up were produced by AI agents (Claude Code and OpenAI Codex), with me setting goals and making calls. — Yiming

Allie predicts the move a human chess player will make. It reads the game so far, both players' ratings and the clock.

The released model is **Allie-v3.0**, a mixture-of-experts (MoE) transformer with 0.69B active and 5.6B total parameters. It was trained on 75B tokens of Lichess and over-the-board games. It continues the original Allie ([paper](https://arxiv.org/abs/2410.03893), [code](https://github.com/ippolito-cmu/allie)).

**On held-out Lichess blitz from July 2026, Allie-v3.0 has lower cross-entropy and higher top-1 accuracy than each Maia-3 model, with 6.6x less inference compute per move than Maia-3 79M.** On our main evaluation, which covers all time controls, it scored 1.2533 nats: 0.032 below its scaling-law forecast.

Maia-3 ([Chessformer, ICLR 2026](https://arxiv.org/abs/2605.19091); [models](https://huggingface.co/collections/MaiaChess/maia3)) comes from Ashton Anderson's group at the University of Toronto. It is the state of the art in human-move prediction, after [Maia](https://arxiv.org/abs/2006.01855) (KDD 2020) and [Maia-2](https://arxiv.org/abs/2409.20553) (NeurIPS 2024). Its open weights and inference code are what made a like-for-like comparison possible.

The comparison is not symmetric. Allie reads the clock and the released Maia-3 does not, our training data is more recent, and our compute is counted rather than measured. See [Caveats](#caveats).

![Cross-entropy against inference GFLOPs per move for Maia-3, the original Allie, Allie-v3.0 and Allie-v3.0 with search](docs/figures/pareto.png)

*Legal-move cross-entropy (lower is better) against inference compute, on the 20,000 benchmark positions where we ran search, so the values differ slightly from the table's 80,000. The "+ search" line adds 5 and 128 simulations of tree search to Allie-v3.0, with its output calibrated on a separate dev set. The original Allie is its raw policy. Compute is counted forward-pass FLOPs, not measured latency.*

| Model | Parameters | GFLOPs/move | CE (nats) | top-1 (%) |
|---|---|---:|---:|---:|
| Maia-3 5M | 5.2M | 0.60 | 1.2955 | 56.95 |
| Maia-3 23M | 22.9M | 2.40 | 1.2475 | 58.34 |
| Maia-3 79M | 78.9M | 9.23 | 1.2269 | 58.80 |
| Original Allie (raw policy) | 305M | 0.61 | 1.2848 | 57.06 |
| Allie-v3.0 | 0.69B active / 5.6B total | 1.39 | 1.2056 | 59.26 |
| Allie-v3.0 (annealed) | 0.69B active / 5.6B total | 1.39 | 1.2030 | 59.29 |

Every model sees the same 80,000 blitz positions. Allie-v3.0 minus Maia-3 79M is −0.0213 [−0.0245, −0.0184] nats of CE and +0.46 [+0.24, +0.67] points of top-1 accuracy, with 95% intervals over games. The other pairs and the annealed version are in [docs/DETAILS.md](docs/DETAILS.md).

**Search.** Tree search over the model's own predictions helps a little. With 128 simulations per move, Allie-v3.0's CE drops by 0.0055 nats [0.0038, 0.0070], at about 126 times the compute. Almost all of the gain comes from players rated 2000 and above.

**The original Allie** ([ICLR 2025](https://arxiv.org/abs/2410.03893)) is scored the same way, from its raw policy: the released 305M-parameter checkpoint, trained on 2022 Lichess blitz, which reproduces the paper's test accuracy. It sits between Maia-3 5M and 23M on this benchmark. It has no clock input and saw only 2022 data.

Its 0.61 GFLOPs per move assume a key-value cache, as for the other models. Its released decoder re-reads the game for every move, about 28 GFLOPs, and its time-adaptive search runs about 50 such passes per move. Here every model is scored on its policy alone.

## How we measure

Both evaluations use Lichess games from July 2026, a month left out of all training. Both score cross-entropy (CE): the negative log-probability, in nats, of the move the human played.

- **Blitz benchmark.** 80,000 positions, 20,000 in each band of the mover's rating: <1400, 1400-2000, 2000-2400 and ≥2400. Probabilities are renormalized over the legal moves. Maia-3 runs from its public checkpoints, on inputs built exactly as the authors' code builds them.
- **Main evaluation.** 16 cells: bullet, blitz, rapid and classical, each in the same four rating bands, with about 100,000 moves per cell. CE is averaged over the cells, without legal-move masking. Every training choice was decided on it.

## Comparison with Maia-3

![CE minus Maia-3 79M's against game rating, for Maia-3 5M and 23M, the original Allie and Allie-v3.0](docs/figures/rating.png)

*CE minus Maia-3 79M's by game rating, on every scored blitz move of the main evaluation; below zero is better. Allie-v3.0 has a 95% band; Maia-3 5M, Maia-3 23M and the original Allie are reference lines.*

Allie-v3.0 has lower CE than Maia-3 79M in 22 of 23 rating bins. In 19 of them, the whole interval is below zero.

The gaps are largest where the setup favours Allie:

- **Bullet** (−0.141 nats). Maia-3 was trained on blitz only.
- **Under 10 seconds on the clock** (−0.223 nats). Allie reads the clock; the released Maia-3 does not.

Maia-3 79M is better in the middlegame (plies 40-59) and with one to two minutes left, by 0.007 and 0.011 nats. On Maia-3's own protocol, which keeps moves after ply 20 with at least 30 seconds left, the accuracy difference is within noise.

## Caveats

- **Not matched compute or data.** Maia-3 trained on about 2.5 years of Lichess blitz (2023-2025); its compute is not published. Allie trained on all formats from 2017 to 2026, with 3.1e20 FLOPs. Our claim is about inference cost.
- **Recency.** Our data runs up to the test month: August 2026 (1.2% of tokens) is in training, July 2026 is not.
- **Different inputs.** The released Maia-3 has no clock input and sees 8 boards. Allie reads the clock and the whole game.
- **FLOPs, not latency.** Allie's cost is computed, not timed. Serving keeps all 5.6B weights in memory (11 GB in BF16), against about 0.3 GB for Maia-3 79M.
- **The anneal was tuned on these positions.** Its learning rate was picked between two values on the benchmark. Allie-v3.0 itself predates that choice.
- **Thin data at the top.** Above game rating 2700, the test month has only 43 games. Allie-v3.0 is a single training run.

## The model

![Diagram of the model: tokens, clock and board inputs, 24 transformer blocks with mixture-of-experts layers, and three outputs](docs/figures/model.png)

*Allie-v3.0: three inputs, 24 transformer blocks with mixture-of-experts layers, and one head with three outputs.*

A game is a sequence of tokens: an 11-token header with the time control and both ratings, then one token per move. At every position, the model also gets three clock features and a small CNN's reading of the board. Attention sees only the game's own history.

The trunk comes from the [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) medium track: 24 blocks of width 1536. The first block has a dense MLP. The other 23 have a mixture-of-experts layer: a router picks 16 of 256 small experts for each token, and one shared expert sees every token.

One head predicts the next move and, as side targets, how long the move took and the game result. A move costs 1.39 GFLOPs when the game so far is cached.

## Training

![Allie-v3.0's blitz-benchmark CE over training tokens, with reference lines for Maia-3, the original Allie and the scaling-law forecast](docs/figures/training.png)

*Allie-v3.0's legal-move CE on the full 80,000-position benchmark during training, against Maia-3, the original Allie and the scaling-law forecast. The law forecasts main-evaluation CE (1.2856). The dashed line converts it to benchmark CE with a linear fit across this run's own checkpoints, made after the run: a unit conversion, not an independent forecast.*

**Data.** Every Lichess month with clock times from May 2017 to August 2026, except July 2026: 7.9B games and 616B tokens. Over-the-board and engine games are added. The sampler doubles a game's weight for every 200 rating points of its stronger player, and uses no game more than 8 times.

**Optimization.** 143,051 steps of 524,288 tokens, on one node of 8 NVIDIA L40S GPUs for 6.6 days. NorMuon trains the weight matrices and Adam trains the rest. The learning rate decays linearly to 0.1% of its peak.

**Result.** The final checkpoint, Allie-v3.0, scored 1.2533 on the main evaluation; the best model of our scaling sweep scored 1.3170. On the benchmark, its CE went below Maia-3 79M's only after about 70B of its 75B tokens.

**Annealed version.** Allie-v3.0 (annealed) continues from Allie-v3.0 with a second, short anneal at a low learning rate, on 1B tokens of recent months. It improved every format and reached 1.2505 on the main evaluation.

## Scaling laws

![Isoflop curves for dense and MoE models at three training budgets](docs/figures/scaling.png)

*Main-evaluation CE against active parameters for dense (grey) and MoE (blue) models, one panel per training budget. Each curve is a quadratic fit in log N through the four sizes around that budget's minimum; points are those sizes' seed means.*

Before Allie-v3.0, we trained 45 dense and MoE models at three compute budgets. The MoE reached lower loss at every budget. A dense model needs 2.2x to 2.5x the compute to match it, although the gap between the best models narrows with scale: 0.033, 0.025 and 0.017 nats.

We fitted a law L(N, D) = E + A·N^−α + B·D^−β to these runs. It forecast 1.2856 for Allie-v3.0, which scored 1.2533. That is below even the fitted floor, E = 1.27.

This is not a recipe effect. Re-run at the sweep's sizes, Allie-v3.0's recipe is slightly worse than the sweep's: +0.008 nats at 6.3e17 FLOPs and +0.004 to +0.006 at 6.2e18. That gap extrapolates to about −0.002 at Allie-v3.0's compute. So the fitted floor looks too high, and more compute keeps paying.

Memory, not the law, set the model's size. The law's optimum for this compute is 1.8B active parameters on 28B tokens.

## Lessons

- **Router collapse.** The first attempt at Allie-v3.0 collapsed during warmup: routing became the same for every token, and up to 116 of 256 experts starved. Two changes fixed it: centring the router's input, and stepping Adam on every step instead of every other step.
- **A rare overflow.** At step 60,081, one token's router scores summed to 6e-20 and the backward pass overflowed. Flooring that sum at 1e-12 fixed it.
- **Small ablations.** Testing modded-nanoGPT's defaults one at a time, dropping multi-token prediction and decaying the learning rate nearly to zero helped. The two router fixes cost a little at small scale, but Allie-v3.0's width needs them. See [DETAILS](docs/DETAILS.md#ablations).

## Next

The next run is an MoE with 1.2B active and about 9.2B total parameters: 24 blocks of width 2048, the first three dense. It will train on 112B tokens for about 18 days on the same node, with experts sharded across GPUs.

Strong-player data now limits how far we can go. Past about 110B tokens, games of 2400+ players must either repeat more or take a smaller share, and both hurt the strongest cells.

## Reproducing

The code is one Python package, [`allie`](src/allie):

- `data`: game stores, the training-time sampler, tokenization
- `model`: the transformer, the mixture-of-experts layer and its kernels, the board CNN
- `train`: trainer, schedule, checkpoints
- `eval`: the main evaluation and the Maia-3 benchmark
- `search`: the tree-search engine
- `lichess`: a Lichess bot, with a CPU inference engine for Allie-v3.0
- `experiments`: frozen studies on Slurm, scaling-law fits

It needs Linux and CUDA 12.8 GPUs. `ALLIE_DATA` points at the game stores and evaluation sets; `ALLIE_PROJECT_ROOT` is where `results/` receives runs and scores (default: this checkout).

1. **Install.** `uv sync` installs the package and PyTorch 2.10. Add `--extra search` for tree search and `--extra test` for the tests; `uv run pytest` skips checks whose GPU or data is missing. The commands below run in that environment (`uv run ...` or an activated `.venv`).
2. **Get the data.** Lichess games come from the monthly database on Hugging Face (`Lichess/standard-chess-games`). We use every month with clock annotations from May 2017 to August 2026, except the test month, July 2026. For each month:
   ```sh
   python -m allie.data.fetch 2024-01 raw/2024-01                  # download, sha256-verified
   python -m allie.data.fastbuild build --hf raw/2024-01 --out $ALLIE_DATA/data-v1/2024-01
   python -m allie.data.fastbuild finalize --out $ALLIE_DATA/data-v1/2024-01
   ```
   Over-the-board games (TWIC, PGN Mentor, Lichess broadcasts) and engine games (CCRL, TCEC) go through `python -m allie.data.external fetch SOURCE`, then `parse` and `finalize`. `allie.data.history` counts the games per bucket, and `allie.data.inventory` builds sampling tables such as the Elo ramp. Allie-v3.0's months, stores, counts and table are in [configs/](configs/allie-v3.0.json).
3. **Build the main evaluation.** `python -m allie.eval.build` builds the July 2026 evaluation, 16 cells, leaving out the original Allie's dev and test games.
4. **Train Allie-v3.0.** Run `allie-train configs/allie-v3.0.json --nproc 8` on one node of eight 48 GB GPUs (on NVIDIA L40S: 131K tokens/s, 6.6 days). The run stops after each 2-day chunk (`max_seconds`), and the same command resumes it.
5. **Evaluate.** `allie-eval --checkpoint results/pretrain/allie-v3.0/last.pt` writes the 16 cells to `results/lm-eval/allie-v3.0/strat-v1.json`. The Maia-3 benchmark samples its positions with `allie.eval.maia3.positions` and `legal`. It then scores Maia-3 with `allie.eval.maia3.score_maia3` (with the Maia-3 code at `MAIA3_REPO`) and Allie with `allie.eval.maia3.score_moe`.
6. **Search.** See [src/allie/search/README.md](src/allie/search/README.md).
7. **Play on Lichess.** `allie-bot` runs Allie-v3.0 as a Lichess bot on a CPU. See [src/allie/lichess/README.md](src/allie/lichess/README.md).
8. **Figures.** `python docs/make_figures.py` regenerates every figure here from the result files.

**Weights.** TODO: the Allie-v3.0 checkpoint is not published yet.

The project is MIT-licensed ([LICENSE](LICENSE)); upstream notices are in [LICENSES.md](LICENSES.md). More detail, including the record of how Allie-v3.0 was run, is in [docs/DETAILS.md](docs/DETAILS.md#reproducing).
