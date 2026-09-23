# Design lock + final scaling sweep (user GO 2026-09-22 ~20:00 ET; model track)

The sweep launches once the data recipe is decided (the data track's 3e17 confirm, scored ~01:00-02:00 ET).
- Evidence: ledger.md, section DESIGN LOCK + SWEEP PLAN.
- Numbers: sweep-plan/s16.py (s16.out); earlier S18-vs-C numbers in sc_mem.py / sweep_size.py.
- Freezing: sweep-plan/round.py.template and freeze.py.

## 1. Design: S16 (user's choice); S18 is the tested fallback
S16 is a MoE with 256 experts, top-16:
- a shared expert of 1/4 of the active MLP; each routed expert = (the other 3/4) / 16, about d/8;
- every width (d_model, MLP, shared, expert) a multiple of 16 (moe_round 16, no round-8 fallback);
- router: sigmoid scores with Kimi K3 quantile balancing, moe_seq 1e-3, router LR x0.1.

S16 is untested. It sits between two designs that tie at 3e17 (golden macro):
- C: 12 picks, experts 1.5x wide, 1.3496;
- S18: 18 picks, 1.3499, the fallback.

S16 is exact on the dense MLP width from d = 512 up. At d = 448 the expert rounds 56 -> 64 (+10.7% active MLP); the
FLOP-matched steps absorb it (6681 vs dense 7312 steps at 1e17). With strict x16, S18 would be off by -7.5% to +9.4% at
d 512-768, which is why x16 favours 16 picks.

| size | dense MLP | S16 shared / expert | S16 active / total | S18 expert, residual | S18 active / total | dense |
|---|---|---|---|---|---|---|
| 10x448 | 1200 | 304 / 64 (+10.7%) | 0.036 / 0.22B | 48, -2.7% | 0.034 / 0.17B | 0.033B |
| 12x512 | 1360 | 336 / 64 | 0.049 / 0.31B | 64, +9.4% | 0.052 / 0.31B | 0.048B |
| 14x640 | 1712 | 432 / 80 | 0.084 / 0.56B | 64, -7.5% | 0.081 / 0.46B | 0.082B |
| 16x768 | 2048 | 512 / 96 | 0.132 / 0.93B | 80, -4.7% | 0.128 / 0.79B | 0.129B |
| 18x896 | 2384 | 592 / 112 | 0.195 / 1.42B | 96, -2.7% | 0.192 / 1.24B | 0.191B |
| 20x1024 | 2736 | 688 / 128 | 0.278 / 2.07B | 112, -1.2% | 0.276 / 1.83B | 0.273B |
| 22x1152 | 3072 | 768 / 144 | 0.380 / 2.89B | 128, 0 | 0.380 / 2.59B | 0.374B |
| 24x1280 | 3408 | 848 / 160 | 0.505 / 3.90B | 144, +0.9% | 0.508 / 3.53B | 0.497B |

Big-run memory per GPU, from moe-perf/memmodel.py calibrated on node8c (8 x L40S budget 44.0 GB, 8 x H100 80.2 GB):

| shape | design | expert | active / total | 8 x L40S 32K / 64K micro | L40S ZeRO-3 experts 64K | 8 x H100 64K | persistent |
|---|---|---|---|---|---|---|---|
| width 1536, 24 layers | S16 | 192 | 0.72 / 5.60B | 30.5 / 38.8 | 29.5 | 38.8 | 21.0 |
| | S18 | 176 | 0.73 / 5.17B | 28.9 / 37.3 | 28.8 | 37.3 | 19.4 |
| | C | 256 | 0.72 / 7.34B | 37.0 / 45.1 (no) | 32.7 | 45.1 | 27.3 |
| width 2048, 24 layers | S16 | 256 | 1.26 / 9.94B | 50.1 / 60.7 (no) | 44.2 (no, 0.2 over) | 60.7 | 37.2 |
| | S18 | 224 | 1.25 / 8.79B | 45.7 / 56.5 (no) | 42.1 | 56.5 | 33.0 |
| | C | 336 | 1.25 / 12.84B | 61.1 / 71.4 (no) | 49.7 (no) | 71.4 | 47.8 |

- At width 1536 S16 fits both micros; at width 2048 it fits on H100 only. On L40S it misses by 0.2 GB with ZeRO-3
  experts, which is inside the model's error, so it is unresolved.
- Routing speed at the big-run shape is unmeasured for any design. S16 dispatches 16 pairs per token (S18 18, C 12) on
  192-wide experts at width 1536. This is parked for the pre-big-run throughput step.

## 2. Sweep = the integration test
- Bundle, identical in every run:
  - S16 or dense;
  - quantile router (MoE runs);
  - fp32 small masters;
  - 16K-token rows, micro 1 row;
  - x0 on, clock-input LR x5, MTP as shipped;
  - RECIPE on the v4 pool (inv4 pin, history mode, 75B final_tokens);
  - pool_frac = the run's own tokens / 75B (modelexp.planned, as in the data track);
  - seed 42; FLOP-matched steps; WSD fractions as frozen;
  - frozen from branch sweep 890aa38 (main 570f88b + data-recipe-3e17 45fbbc6, disjoint files).
- Ladder: a 10x448, b 12x512, c 14x640, d 16x768, e 18x896, f 20x1024, g 22x1152, h 24x1280.

| budget | S16 | dense | prior optimum (old bundle) | tokens (B), smallest to largest size |
|---|---|---|---|---|
| 1e17 | a b c d | a b c d | b / b | S16 3.50-0.87, dense 3.83-0.89 |
| 3e17 | c d e f | c d e f | d / d | 4.09-1.17, 4.21-1.19 |
| 1e18 | d e f g | e f g h | MoE flat d-f / dense >= f | 8.48-2.82, 5.74-2.14 |

- Adaptive rule: if a cell's best size is at an end, add the next ladder size beyond it. At most one per cell; if that
  size is best again, flag it.
- Seed 43 at each family's best 1e18 size (2 runs) once 1e18 reads out. Later waves are frozen from the same commit and
  recipe.
- Health: step-500 bar per run (NaN, divergence, starved > 5% or rising), reporting both even and odd strata.
- Readout:
  - per family and budget: the isoFLOP golden macro / expert macro;
  - L(N, D) refit per family on the new bundle, which replaces the old law and closes audit M03;
  - N*(C), and the node-week CM of MoE vs dense with bootstrap errors;
  - tok/s per shape.

## 3. Cost and schedule
Costs are anchored on the measured S18 / dense runs, +-30%. S16 has fewer picks and wider experts than S18: assumed the
same speed; the pre-launch check measures it.

| cell (4 runs) | per run | cell total |
|---|---|---|
| 1e17 S16, 1 x A6000 | ~8.6 A6000-h, ~9 h wall | 34 |
| 1e17 dense, 1 x A6000 | ~4.4 A6000-h, ~4.5 h | 18 |
| 3e17 S16, 4 x L40S | ~13.6 L40S-h, 3.4 h | 54 |
| 3e17 dense, 4 x L40S | ~6.4 L40S-h, 1.6 h | 26 |
| 1e18 S16, 4 x H100 or 4 x L40S | 12 H100-h, 3 h / 45 L40S-h, 11 h | 48 / 180 |
| 1e18 dense, 4 x H100 or 4 x L40S | 5.7 H100-h, 1.5 h / 21.5 L40S-h, 5.4 h | 23 / 86 |
| seeds (S16 + dense at 1e18) | | 18 H100-h / 67 L40S-h |
| pre-launch check (2 x 128 steps + scoring) | | ~4 L40S-h (+ ~2 H100-h) |

- Total for 26 runs: ~230 GPU-h with 1e18 and the seeds on orchard (~140 babel + ~90 H100); ~470 babel-only.
  Adaptive sizes add ~15-60.
- Lanes. One GPU type per budget, so no isoFLOP parabola mixes hardware:
  - 1e17: 1 x A6000 on dei-group, tagged requeue-ok (8 at once, under the governor).
  - 3e17: scripts/race_submit.sh, 4 x L40S, general + preempt, one race per run (--array=i).
  - 1e18 + seeds: orchard 4 x H100 (omake.py, then orun.sh; 8 x 4 GPUs on <= 4 nodes; never raced with babel).
    Fallback if the v4 pool is not on GCS in time: babel 4 x L40S, raced. All 1e18 runs go on one type or the other.
- Wall clock after launch:
  - with orchard: ~12-14 h. 1e17 (~9 h), 3e17 (4-10 h, depending on preempt slots) and 1e18 (3 h) run in parallel;
    seeds and adaptive sizes add ~3 h.
  - babel only: ~1.5-2 days.

## 4. Prep status (2026-09-22 20:55 ET)
- Codex:
  - plans, data, rounding and FLOP matching CLEAR (20:15);
  - orchard provenance fixes CLEAR (20:25): sweep 48e85c6 (orun strict hashes + oargs argv binding; deployed).
- Launch: `sweep-plan/launch.sh LABEL [orchard|babel]`, with LABEL = c8s200f0v4 or control. It freezes the three studies
  and races every run: 1e17 on dei A6000, 3e17 on L40S, 1e18 on orchard + autoscore or on babel L40S.
- Babel pre-launch check: sweep-check3-c8s200f0v4 (legs 128 -> 256 with a resume, then scored), queued; it starts when
  general frees.
- Orchard v4 upload: paused at 20:37 (8 of 58 months done). It slowed the data confirm runs (control 430K -> 222K tok/s).
  - Full rest: ~625 GB, ~6.4 h at 27 MB/s.
  - Subset, not built: 145 GB for f <= 0.113, 187 GB for f <= 0.19, ~1.5-1.9 h.
  - Or 1e18 on babel.
  - The H100 check (sweep-checkh) waits on the data.
- NFS: every babel run opens >= ~220 GB of shards (small buckets' shard-0000s), so stagger starts, co-locate runs for the
  page cache, and watch sampler_wait.
- Held old-bundle runs: all 5 cancelled by main.
