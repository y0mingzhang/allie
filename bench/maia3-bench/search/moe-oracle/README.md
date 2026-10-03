# MoE search oracle

`search/` ships a model port for the dense ship recipe only. `oracle.py` lets the unchanged
production search run modded MoE checkpoints, such as the sweep MoE 1e18 s42
(`results/pretrain/sw-1e18-s16-18x896-c8s200f0v4-s42`).

- The model is the checkpoint's own training source (`sweep-1e18-c8s200f0v4/source-ours`,
  hash-checked, foreign `modded_*` modules rejected), with the same modules, weights, Triton MoE
  kernels and TF32 policy as the trainer. `forward()` restates `GPT.forward` (eval path) with
  attention, the smear predecessor and rotary positions as inputs.
- Tree KV cache: roots prefill once into a slot pool. Each search node adds one token that
  attends to its path (root prefix plus ancestors) through a per-node slot table. Both of this
  model's windows cover a whole game (ws 11/23 x 128 >= 1025), so attention is plain causal.
  Rotary positions are in-game positions; `prefill(origins=...)` offsets them, for tests only.
- The interface is `ShipOracle`'s (`reset()`, `handles(prefixes, features, clock_rule)`,
  `new_tokens`). `MoEHandles` repeats ShipHandles' causal board and clock bookkeeping, using
  `search.board`.

```python
from oracle import MoEOracle          # this directory on sys.path
from search import Search
engine = Search(MoEOracle(".../last.pt"), threads=8)
engine.batch_size = 512               # bypasses Search's <=128 check; see capacity below
engine.predict(queries, budget=128)
```

Capacity: a batch needs sum(prefix lengths) + roots x simulations <= `slots` (2^18) and
roots x (simulations + 1) <= `rows` (2^17); the oracle raises when full. Group batches by these
bounds (gen_targets.py does).

Runtime: the training runtime (`modded_runtime_stage.py`, torch 2.10) plus
`envs/search-overlay` (scipy, pybind11) on PYTHONPATH; see `run.sbatch`.

## Parity (`parity.py` -> `parity.json`, `smoke.py` -> `smoke.json`)

Reference: the training forward on 143 golden games in packed 1024-token rows, TF32 on. KL is
KL(ref || oracle) over the move softmax in nats; "logp" is the signed mean change in the played
move's log-probability (± token-wise s.e.; chains are correlated, so use game-clustered
intervals for any inference).

| check | n | KL mean | KL p99 | top-1 agree | logp |
|---|---:|---:|---:|---:|---:|
| A. `forward()` restatement on the training's own attention vs the eager training forward | 8 rows | bitwise equal | | | |
| B. tree nodes (16-move chains) vs a fresh prefill of the same prefix | 1850 | 6.0e-4 | 2.9e-3 | 98.6% | +0.0014 ± 0.0008 |
| C. oracle at matched rotary positions vs the eager training forward | 1850 | 5.9e-4 | 2.9e-3 | 98.9% | +0.0004 ± 0.0008 |
| C. the same vs the compiled training forward (score_moe, evaluate) | 1850 | 1.25e-3 | 5.3e-3 | 97.0% | +0.0007 ± 0.0011 |
| D. oracle as search runs it (in-game positions) vs compiled | 1850 | 1.26e-3 | 5.6e-3 | 97.0% | +0.0022 ± 0.0011 |
| for scale: eager vs compiled training forward | 1850 | 1.28e-3 | 5.6e-3 | 97.0% | +0.0003 ± 0.0012 |
| for scale: compiled forward vs itself with games repacked 37 tokens later | 1850 | 4.5e-4 | 2.4e-3 | 99.3% | +0.0004 ± 0.0007 |

The restated math is exact. What remains is kernel numerics: SDPA instead of flex attention,
and eager instead of compiled. In mean KL, the oracle is no further from the compiled evaluator
than the training code's own eager forward is; its max KL is higher (0.027 vs 0.0087). These are sensitivity comparisons, not bitwise evaluator parity: treat the oracle as a
numerically distinct teacher and measure search gains against its own legal baseline. Codex
cleared it for research throughput and target generation on that basis (2026-09-24).

Search smoke (256 golden positions, 128 simulations, 128 roots per batch, 32,064 nodes): every node's clock features
and board, as fed to the model, match an independent CPUHandles-style replay from its token path
(0 feature error, 0 board mismatches). A repeated predict is bitwise identical, and two roots
with the same moves but different clocks get different outputs.

At the generation batch size (`smoke-b512-wide.json`: 1,023 golden positions plus a twin, 512
roots per batch, 127,700 nodes) the bookkeeping replay is again exact (0 feature error, 0 board
mismatches) and a repeat is bitwise identical. 16,384 sampled nodes against a fresh prefill of
their paths: KL mean 4-5e-4, p99 2.9-3.6e-3, max 0.039, top-1 agreement 99.3%. The worst nodes
are spread over waves of 104 to 9,094 nodes, at depths 1-4; no pattern points to a cache error.
