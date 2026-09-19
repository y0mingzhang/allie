# Allie

Human chess move prediction from move history and player ratings. This repository contains the current move-sequence model, Qwen controls, and the completed size/data scaling study.

The completed [inference track](search/README.md) adds a resident search engine,
Allie MCTS references, and [transfer results and Pareto plots](search/TRANSFER_REPORT.md)
for the later 34M and 129M recipes with board and clock inputs. The architecture
and scaling sections below describe the earlier move-only study.

The objective is low next-move cross-entropy (CE), including moves played by players rated at least 2400 and 2600. Lower is better. The current scaling study uses no board features, distillation, or search.

## Architecture

Ours derives from the medium track of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt/tree/ecbb586296d3dac36fd206211f25d63bad4a6b35), with chess-specific tokenization, causal masks, losses and checkpoint recovery. The checked-in implementation matches the frozen source used for the small-model scaling runs; hashes are in [analysis/provenance.json](analysis/provenance.json).

```text
Game header, including both Elo ratings → move tokens
    → token embeddings + gated previous-token mixing
    → transformer blocks: normalized attention + 4× ReLU² MLP
    → output head → distribution over 1,968 move tokens
```

Each game has an 11-token header. Original 1,024-token input rows can contain multiple games. Ours masks attention at both row and game boundaries, so a move uses only its causal game context. The configured attention windows cover the full game prefix within a row. Strength conditioning remains in the input.

The model retains these medium-track features:

- Head dimension 64, Q/K normalization and rotary positions.
- Gated attention, token-value embeddings in early and late layers, and three gated skip connections.
- Learned mixtures of residual and input embeddings, plus a learned residual subtraction before the output head.
- Input/output embeddings start tied and split at update 65. The padded vocabulary has 2,432 entries; CE is normalized over the original 1,968 move IDs.
- An early auxiliary objective predicts up to three future plies from the same logits, within a game. It is disabled after 64 updates.

Primary source: [model adapter](scripts/modded_medium.py), [transformer and optimizers](scripts/modded_medium_core.py), [trainer](scripts/modded_train.py), [schedule](scripts/modded_wsd.py).

Training uses Muon for attention/MLP matrices and selected gates, with Adam for embeddings, the output head and other parameter groups. Matrix parameters and updates use mixed FP32/BF16 storage; forward matrix products use BF16. FP8 is disabled in the reported experiments. The selected recipe uses a full linear LR decline, a schedule multiplier of 4→0.2, and 32-step momentum warmup. Checkpoints preserve the model, optimizer, RNG, data order and schedule state.

### Qwen control

[scaled_native_qwen.py](scripts/scaled_native_qwen.py) uses native Qwen3 blocks: RMSNorm, RoPE, grouped-query attention, head dimension 128, and SwiGLU. Controls start from random weights with unit Q/K norm scales. They use Muon, FP32 Adam state for normalization vectors, a 0.01→0.0005 linear LR schedule, and 32 warmup updates.

Qwen trains on all 2,350 token IDs with full causal packed-row attention. Ours trains on move targets with game-isolated attention. Both are evaluated with the same raw move-token normalization. This compares complete training recipes, not architecture alone.

The historical big Qwen checkpoint is a separate reference: 1.419B parameters and 44.407B processed tokens. Its WSD schedule, batch size and initialization differ from these controls. There is evidence of inherited pretrained Q/K norm scales; the historical checkpoint-to-run binding is not cryptographically established.

## Data and evaluation

Original corpus: `yimingzhang/lichess_tokens_v2`, revision `20a899ddf344ccaea74e273509a60e5a511125f8`. Original splits and packed rows are retained.

| Quantity | Count |
|---|---:|
| Training shards | 100 |
| Training rows | 54,368,123 |
| Input tokens per corpus pass | 55,672,957,952 |
| ≥2400 move targets | 4.084B; 8.74% of moves |
| ≥2600 move targets | 1.565B; 3.35% of moves |
| Validation rows | 5,371 |
| Validation move / ≥2400 / ≥2600 targets | 4,616,637 / 396,483 / 151,660 |

Counts are stored occurrences, not deduplicated positions. Expert rating is the rating of the player making the target move. Evaluation uses raw CE over the 1,968 move IDs, without legal masking. Final tests remain unopened.

## Scaling study

The initial grid was two recipes × three shapes × four independent token budgets:

| Layers × width | Ours parameters | Qwen parameters |
|---|---:|---:|
| 8 × 128 | 3,752,557 | 2,506,368 |
| 12 × 256 | 14,419,457 | 11,829,504 |
| 16 × 512 | 60,296,597 | 57,477,632 |

Each shape trained for 134M, 268M, 537M and 1.074B tokens. Global batch was 512 rows × 1,024 tokens. Each endpoint had its own complete LR schedule.

The final fit has **42 endpoints**: those 24 runs, six prospective 16×384 checks, and twelve historical 128M/483M endpoints extending to 3.758B tokens. Six additional seed repeats diagnose variability and are excluded from the fit. The historical big Qwen checkpoint and our 1.064B short run are also excluded.

For each recipe and metric, fit independently:

\[
L(N,D)=E+A(N/10^7)^{-\alpha}+B(D/10^8)^{-\beta}.
\]

Here N is total parameter count and D is processed training tokens. Fitting minimizes unweighted squared CE residuals with positive exponents and a nonnegative floor.

| Recipe | Metric | E | A | B | α | β | Fit RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| Ours | Move | 1.30980 | 0.40453 | 1.03273 | 0.6326 | 1.0244 | 0.03025 |
| Qwen | Move | 1.29305 | 0.53400 | 1.11295 | 0.5494 | 0.9422 | 0.03564 |
| Ours | ≥2400 | 1.13361 | 0.51189 | 1.16990 | 0.5633 | 0.9243 | 0.02525 |
| Qwen | ≥2400 | 1.11378 | 0.66675 | 1.26838 | 0.4988 | 0.8598 | 0.03203 |

[scaling_fit.json](analysis/scaling_fit.json) contains full precision coefficients, all three metrics, the 42 observations, residuals and omitted-group checks.

### Optimal allocation and compute multiplier

Let C=ND, with nominal training FLOPs ≈6C. At fixed C, the fitted optimum satisfies

\[
\alpha A(N/10^7)^{-\alpha}=\beta B(D/10^8)^{-\beta}.
\]

Consequently N scales as C^(β/(α+β)), and D as C^(α/(α+β)). D/N is not constant. Each metric has its own optimal allocation.

Compute multiplier (CM) is Qwen compute divided by our compute at equal loss. Distinguish an optimally allocated Qwen forecast from the actual historical Qwen run:

| Question | Move CM | ≥2400 CM |
|---|---:|---:|
| Ours at ND=10¹⁸ vs optimally allocated Qwen | 2.29× | 2.33× |
| Ours at one equivalent L40S node-day vs optimally allocated Qwen | 1.40× | 1.67× |
| Match historical checkpoint's measured loss vs its actual nominal compute | 7.25× | 1.01× |

One equivalent node-day assumes eight L40S GPUs, dense BF16 peak 362.05 TFLOP/s per GPU and 35% MFU: 8.76×10¹⁹ model FLOPs, or ND=1.46×10¹⁹. This is a conversion assumption, not measured runtime. [NVIDIA specifications](https://www.nvidia.com/en-us/data-center/l40s/).

### How accurate is it?

Prospective intermediate-size move predictions had RMSE 0.014 for ours and 0.025 for Qwen using the initial tiny-grid fit. Omitting the largest size or horizon from the final fit produces roughly 0.04 move-CE error. At two repeated points, seed SD is only 0.0033/0.0025, while final-fit errors are 0.079/0.086.

On the original tiny grid, nearly all fitting error is unavoidable for any additive size-plus-data formula. More coefficient optimization cannot remove it. The actual big Qwen checkpoint has move CE **1.3422**, versus **1.3317** predicted; expert ≥2400 CE is **1.1622**, versus **1.1768** predicted.

The laws are useful approximations, not validated guarantees at billion-parameter, long-token allocations. Their fitted floors drive the shrinking projected CM. Historical anchors also differ in shape and execution details. Logged FLOP counters for our 8/12-layer models overcount some operations and are excluded from this analysis.

## Reproduce the calculations

The calculator requires only Python's standard library and the checked-in JSON files:

```bash
python analysis/scaling.py --nd 1e18 --metric move
python analysis/scaling.py --node-days 1 --metric expert2400
python analysis/scaling.py --target-original --metric expert2400
```

To refit the saved observations with the original fitting code:

```bash
python -m pip install -r requirements-analysis.txt
python analysis/refit.py --metric move
```

Raw evaluations, datasets, checkpoints and job records remain outside Git under `/data/group_data/dei-group/yimingz3/allie`. The local `results` symlink points there. Snapshot provenance is in [analysis/provenance.json](analysis/provenance.json).

## Training and cluster operations

The training stack is cluster-specific. Ours uses the pinned PyTorch 2.10.0+cu128 environment in `/data/group_data/dei-group/yimingz3/allie/envs/modded-torch210`; Qwen uses PyTorch 2.8.0+cu128 with the native dependency overlay in `envs/qwen-reproduction-deps-v1`. The native Picotron source is hash-pinned under `results/recipe10x/qwen-reproduction/source-v1` and is required by `historical_qwen_runtime.py`.

`scripts/prepare_tiny_scaling.py` records the exact training commands and freezes sources. Existing study directories are immutable; rerunning preparation must not overwrite completed runs. Resume historical checkpoints with their frozen source and runtime, not a newly edited working tree. `scripts/stage_corpus.py` restores a verified node-local cache from durable data. `scripts/budget.py ledger` reports cumulative Slurm usage.

IsoFLOP v1 (ours vs chess-v2 Qwen, 26 runs, 3e16–3e17) is complete: `results/recipe10x/isoflop-v1/RESULTS.md` supersedes the scaling fit above. [GOAL.md](GOAL.md) records the research state; [CONTROLLER.md](CONTROLLER.md) describes the running controller. Cleanup does not authorize a new training run.

Next hypothesis discussed: change expert sampling proportions at fixed compute. A prior 3× expert loss-weighting trial at 128M/470M tokens worsened both overall and expert CE; it did not test additional expert-example exposure. No new mixture experiment has been launched.

The modded-nanogpt-derived code retains its [MIT license](scripts/modded_medium_LICENSE).
