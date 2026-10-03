# Allie

Allie predicts the move a human chess player will make from the game so far, both players' ratings and the clock. The released model, **Allie-v3.0**, is a mixture-of-experts (MoE) transformer with 0.69B active and 5.6B total parameters, trained on 75B tokens of Lichess and over-the-board games. It continues the original Allie ([paper](https://arxiv.org/abs/2410.03893), [code](https://github.com/ippolito-cmu/allie)).

**On held-out Lichess blitz from July 2026, Allie-v3.0 reaches lower cross-entropy and higher top-1 accuracy than each Maia-3 model, using 6.6x less inference compute per move than Maia-3 79M.** On our all-format main evaluation it scored 1.2533 nats, 0.032 below its scaling-law forecast.

Maia-3 ([Chessformer, ICLR 2026](https://arxiv.org/abs/2605.19091); [models](https://huggingface.co/collections/MaiaChess/maia3)), from Ashton Anderson's group at the University of Toronto, is the state of the art in human-move prediction, after [Maia](https://arxiv.org/abs/2006.01855) (KDD 2020) and [Maia-2](https://arxiv.org/abs/2409.20553) (NeurIPS 2024). Its open release of weights and inference code is what made a like-for-like comparison possible. The comparison is not symmetric: Allie reads the clock and the released Maia-3 does not, our training data is more recent, and our compute is counted, not measured (see [Caveats](#caveats)).

![Cross-entropy and top-1 accuracy against inference GFLOPs per move for Allie-v3.0 and Maia-3](docs/figures/pareto.png)

| Model | Parameters | GFLOPs/move | CE (nats) | top-1 (%) |
|---|---|---:|---:|---:|
| Maia-3 5M | 5.2M | 0.60 | 1.2955 | 56.95 |
| Maia-3 23M | 22.9M | 2.40 | 1.2475 | 58.34 |
| Maia-3 79M | 78.9M | 9.23 | 1.2269 | 58.80 |
| Allie-v3.0 | 0.69B active / 5.6B total | 1.39 | 1.2056 | 59.26 |
| Allie-v3.0 (annealed) | 0.69B active / 5.6B total | 1.39 | 1.2030 | 59.29 |

Blitz benchmark, the same 80,000 positions for every model. On those positions, Allie-v3.0 minus Maia-3 79M is −0.0213 [−0.0245, −0.0184] nats and +0.46 [+0.24, +0.67] points of top-1 accuracy (95% intervals over games); for the annealed version, −0.0238 [−0.0272, −0.0208] and +0.48 [+0.27, +0.70]. More numbers: [docs/DETAILS.md](docs/DETAILS.md).

## How we measure

Both evaluations use Lichess games from July 2026, a month excluded from all training, and score cross-entropy (CE): the negative log-probability, in nats, of the move the human played. The **blitz benchmark** has 80,000 positions, 20,000 per band of the mover's rating (<1400, 1400-2000, 2000-2400, ≥2400), with probabilities renormalized over legal moves. Maia-3 runs from its public checkpoints on its own inputs (the current and 7 previous boards and both ratings), built exactly as the authors' code builds them. The **main evaluation**, which decided every training choice, averages CE over 16 cells (bullet, blitz, rapid and classical, each in the four rating bands, about 100,000 moves per cell) without legal-move masking.

## Comparison with Maia-3

![Accuracy and CE against game rating for Maia-3 5M, 23M, 79M and two Allie models](docs/figures/rating.png)

Allie-v3.0 has lower CE than Maia-3 79M in 22 of 23 rating bins (intervals below zero in 19) and equal or higher accuracy in all but two, both within noise. The differences are largest where the setup favours Allie: bullet (−0.141 nats), which lies outside Maia-3's blitz training data, and positions with under 10 s on the clock (−0.223), where Allie reads the clock and the released Maia-3 does not. Maia-3 79M predicts better in the middlegame (plies 40-59, Allie +0.007 nats) and with one to two minutes left (+0.011), and on Maia-3's own protocol (ply > 20, at least 30 s left) the accuracy difference is within noise: +0.12 pp [−0.16, +0.40].

## Caveats

- **Not matched compute or data.** Maia-3 trained on about 2.5 years of Lichess blitz (January 2023 to July 2025; compute not published), Allie on all formats from 2017 to 2026 plus over-the-board and engine games, with 3.1e20 FLOPs. The claim is about inference cost.
- **Recency.** Our data runs up to the test month, and August 2026 (1.2% of tokens) is in training; July 2026 is excluded everywhere. The annealed version's extra tokens came only from 2024-2026 months.
- **Different inputs.** The released Maia-3 checkpoints have no clock input and see 8 boards; Allie reads the clock and the whole game. The largest differences are in time trouble.
- **FLOPs, not latency.** Allie's cost is analytic (2 × active parameters per move, with a cached game). Serving keeps all 5.6B weights resident (11 GB in BF16), against about 0.3 GB for Maia-3 79M.
- **The anneal was tuned on these positions:** its learning rate was chosen between two values on the benchmark. Allie-v3.0 itself predates that choice.
- **Thin data at the top.** Above game rating 2700 the test month has 43 games, and Allie-v3.0 is a single training run.

## The model

![Diagram of the model: tokens, clock and board inputs, 24 transformer blocks with mixture-of-experts layers, and three outputs](docs/figures/model.png)

A game is a token sequence: an 11-token header (time control and both ratings as digits), then one token per move out of 1,968 possible moves. At every position, three clock features and a small CNN over the current board are added to the token embedding, and attention sees only the game's own history. The trunk descends from the [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) medium track: 24 blocks of width 1536. The first block has a dense MLP; the other 23 have a mixture-of-experts layer with 256 small experts, 16 chosen per token by a sigmoid router with load-balancing biases, plus one shared expert. One head predicts the next move and, as auxiliary targets, how long the move took and the game result. A move costs 1.39 GFLOPs with the game's key-value cache.

## Training

![Main-eval CE over training against the scaling-law forecast, and the gap to Maia-3 79M over training](docs/figures/training.png)

The data is every Lichess month with clock times from May 2017 to August 2026 except July 2026 (7.9B games, 616B tokens), plus over-the-board and engine games. A sampler doubles a game's weight for every 200 rating points of its stronger player, using no game more than 8 times. Training ran 143,051 steps of 524,288 tokens with NorMuon for the weight matrices and Adam for the rest, decaying the learning rate linearly to 0.1% of its peak, on one node of 8 NVIDIA L40S GPUs at 131K tokens/s for 6.6 days. The final checkpoint, Allie-v3.0, reached 1.2533 on the main eval (best sweep model: 1.3170); on the benchmark, its CE fell below Maia-3 79M's only at 70B of its 75B tokens. Allie-v3.0 (annealed) continues it with a second, low-learning-rate anneal on 1B tokens of recent months, which improved every format (main eval 1.2505).

## Scaling laws

![Isoflop curves for dense and MoE models at three training budgets](docs/figures/scaling.png)

Before Allie-v3.0 we trained 45 dense and MoE models at three budgets. The MoE reaches lower loss at every budget: dense models need 2.2x, 2.3x and 2.5x the compute to match it, although the gap between the fitted optima narrows (0.033, 0.025, 0.017 nats). A fitted law L(N, D) = E + A·N^−α + B·D^−β forecast 1.2856 for Allie-v3.0; it scored 1.2533, below even the fitted floor E = 1.27. This is not a recipe effect: re-run at the sweep's sizes, Allie-v3.0's recipe is slightly worse than the sweep's (+0.008 nats at 6.3e17 FLOPs, +0.004 to +0.006 at 6.2e18), a gap that extrapolates to about −0.002 at Allie-v3.0's compute. The fitted floor looks too high: more compute keeps paying. Memory, not the law, set the model's size (the law's optimum: 1.8B active parameters on 28B tokens).

## Inference cost

![CE against the number of routed experts per token, and the gain from tree search against training compute](docs/figures/inference.png)

Allie-v3.0 uses 16 of its 256 routed experts per token, but the best 8 by router score carry nearly everything: computing only those, with no retraining, costs 0.0011 nats on the benchmark at 1.06 GFLOPs per move; 6 cost 0.0025 and 4 cost 0.0095. Tree search over the model's own move, outcome and thinking-time predictions helps less as models grow: 128 simulations gain 0.022, 0.019 and 0.013 nats on the sweep's MoE models and 0.0055 on Allie-v3.0.

## Training stability

![Starved experts in the first MoE layer over the first 3,125 steps: first attempt against Allie-v3.0](docs/figures/router.png)

The first attempt to train Allie-v3.0 collapsed during warmup. In the first MoE layer one direction came to carry most of the router's input, routing became the same for every token, and up to 116 of 256 experts starved. Halving the router's learning rate only delayed it. Subtracting a running mean from the router's input, and stepping the Adam-trained parameters on every step rather than every other step (modded-nanoGPT's default), stopped it. At step 60,081 one token's 16 router scores summed to 6e-20 and the backward pass of the gate normalization overflowed; flooring that sum at 1e-12 fixed it.

## Ablations

![Small-scale ablations of training choices](docs/figures/ablations.png)

modded-nanoGPT is tuned for short GPT-2 runs, so we tested its defaults one at a time on a small MoE. Dropping multi-token prediction and decaying the learning rate nearly to zero helped; halving weight decay and removing the time-control tokens hurt. Giving the model both ratings at every token, not just in the header, gained nothing: the moves themselves reveal a player's strength. The two router fixes cost 0.004-0.006 nats at this scale, where routers stay healthy; we keep them because Allie-v3.0's width needs them. Every difference is within about two seed standard deviations, so read directions, not sizes.

## Next

The next run is an MoE with 1.2B active / ~9.2B total parameters (24 blocks of width 2048, the first three dense), 112B tokens, about 18 days on the same node with experts sharded across GPUs. Strong-player data now limits tokens: past about 110B, games of 2400+ players must either repeat more or take a smaller share, and both hurt the strongest cells. Beyond it, compute should go to parameters and to more strong-player data.

## Reproducing

The code is one Python package, [`allie`](src/allie): `data` (game stores, the training-time sampler, tokenization), `model` (transformer, mixture-of-experts layer and kernels, board CNN), `train` (trainer, schedule, checkpoints), `eval` (the main evaluation and the Maia-3 benchmark), `search` (the tree-search engine) and `experiments` (frozen studies on Slurm, scaling-law fits). It needs Linux and CUDA 12.8 GPUs. Two environment variables point at storage: `ALLIE_DATA` (game stores and evaluation sets) and `ALLIE_PROJECT_ROOT` (whose `results/` receives runs and scores; default: this checkout).

1. **Install.** `uv sync` installs the package and PyTorch 2.10; add `--extra search` for tree search and `--extra test` for the tests (`uv run pytest`: checks skip when their GPU or data is missing). The commands below run in that environment (`uv run ...` or an activated `.venv`).
2. **Get the data.** Lichess games come from the monthly database on Hugging Face (`Lichess/standard-chess-games`), every month with clock annotations from May 2017 to August 2026 except the test month, July 2026. For each month:
   ```sh
   python -m allie.data.fetch 2024-01 raw/2024-01                  # download, sha256-verified
   python -m allie.data.fastbuild build --hf raw/2024-01 --out $ALLIE_DATA/data-v1/2024-01
   python -m allie.data.fastbuild finalize --out $ALLIE_DATA/data-v1/2024-01
   ```
   Over-the-board (TWIC, PGN Mentor, Lichess broadcasts) and engine games (CCRL, TCEC) go through `python -m allie.data.external fetch SOURCE`, then `parse` and `finalize`. `allie.data.history` counts the games per bucket and `allie.data.inventory` builds sampling tables such as the Elo ramp. Allie-v3.0's months, stores, counts and table are in [configs/](configs/allie-v3.0.json).
3. **Build the main evaluation:** `python -m allie.eval.build` (July 2026, 16 cells, excluding the original Allie dev and test games).
4. **Train Allie-v3.0:** `allie-train configs/allie-v3.0.json --nproc 8` on one node of eight 48 GB GPUs (NVIDIA L40S: 131K tokens/s, 6.6 days). The run stops after each 2-day chunk (`max_seconds`); the same command resumes it.
5. **Evaluate:** `allie-eval --checkpoint results/pretrain/allie-v3.0/last.pt` writes the 16 cells to `results/lm-eval/allie-v3.0/strat-v1.json`. The Maia-3 benchmark samples its positions with `allie.eval.maia3.positions` and `legal`, then scores Maia-3 (`allie.eval.maia3.score_maia3`, with the Maia-3 code at `MAIA3_REPO`) and Allie (`allie.eval.maia3.score_moe`).
6. **Search:** see [src/allie/search/README.md](src/allie/search/README.md).
7. **Figures:** `python docs/make_figures.py` regenerates every figure here from the result files.

**Weights.** TODO: the Allie-v3.0 checkpoint is not published yet.

The project is MIT-licensed ([LICENSE](LICENSE)); upstream notices are in [LICENSES.md](LICENSES.md). More detail, including the record of how Allie-v3.0 was run, is in [docs/DETAILS.md](docs/DETAILS.md#reproducing).
