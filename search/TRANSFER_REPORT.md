# Search transfer to the final recipe

**Search still helps the larger model, but its CE gain shrinks.** This finite
stage is complete. The GPU has been released.

| Same frozen 1,000-budget method | 34M checkpoint | 129M checkpoint |
|---|---:|---:|
| Legal macro / expert CE | 1.47068 / 1.38850 | 1.35437 / 1.24028 |
| Search macro / expert CE | 1.41895 / 1.28654 | 1.33858 / 1.18006 |
| CE reduction vs legal | 0.05172 / 0.10196 | 0.01579 / 0.06022 |
| Training-equivalent CM vs raw | 2.22× / 3.50× | 1.53× / 3.98× |
| Average new NN evaluations | 969.3 | 966.5 |

CM is conditional on the old scaling-law shape, shifted to each checkpoint.
The expert curve gets flatter, so its CM can stay high while its CE gain shrinks.
The large/small expert CM ratio is 1.13, with 95% interval 0.77–1.80: no established
increase. These intervals exclude law-fit and training-seed uncertainty.

The frozen adaptive router is the useful cost/quality reference: on 129M it uses
461.4 NN evaluations, with macro/expert CE 1.33748/1.17985 and conditional
CM 1.57×/4.00×. That is 52% fewer NN evaluations than fixed 1,000, without a
resolved CE difference. The tested Allie and fixed-ply references are worse;
this does not rule out every adaptive-MCTS design.

Across model sizes, the larger legal policy also beats the small model's heavy
search at about 12 versus 58 analytical inference GFLOPs per query, including
root prefill. Training cost, memory and CPU-hosting latency are separate.

![Node versus quality](../results/search-v1/transfer-v1/pareto.png)

- [Full write-up: all methods, paired intervals and limitations](../results/search-v1/transfer-v1/REPORT.md)
- [Node frontier PDF](../results/search-v1/transfer-v1/pareto.pdf)
- [CM versus pretraining scale](../results/search-v1/transfer-v1/transfer-cm.png)
- [Inference-FLOP frontier](../results/search-v1/transfer-v1/inference-flops.png)
- [Production implementation, tests and historical reproduction](README.md)

Limits: one checkpoint per scale; reused 8,192-position golden sample anchored
to the full canonical evaluation; no fresh independent confirmation; imagined
clock updates are not an established best rule. On August development, the
large-model gain was null, unlike July golden. All are disclosed in the report.
The original 10×/10× target remains unmet. Transfer cost: 0.903 allocated GPU-hours;
cumulative search usage: 15.728 GPU-hours, with all earlier charges retained.
