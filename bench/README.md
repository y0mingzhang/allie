# Evaluation and analysis scripts

Copies of the research scripts behind the numbers in the top-level README, in their original directory layout (they import each other by relative path). They read and write the group's storage directly (absolute paths under `/data/group_data/dei-group/yimingz3/allie` and `/home/yimingz3/src/allie/results`), so outside that cluster they document the method rather than run as is; the training code itself honours `ALLIE_DATA` and `ALLIE_PROJECT_ROOT`.

| Script | What it does | README result |
|---|---|---|
| `maia3-bench/build_positions.py` | Rebuilds the blitz games of the main eval from its packed rows and draws the fixed 20,000-per-band sample; `verify` replays games and checks every target move is legal | the 80,000-position benchmark |
| `maia3-bench/legal.py` | Legal-move lists for the sampled positions | legal-move renormalization |
| `maia3-bench/score_maia3.py`, `check_uci.py` | Scores a Maia-3 checkpoint on CPU with its native inputs; cross-checks against the authors' own engine | Maia-3 rows |
| `maia3-bench/score_moe.py` | Scores an Allie MoE checkpoint on one GPU through its frozen training source | Allie rows |
| `maia3-bench/build_formats.py`, `build_rating.py` | Companion samples: bullet / rapid / classical, and every blitz move for the rating plot | format rows, rating plot |
| `maia3-bench/aggregate.py`, `report_bigrun.py` | Game-bootstrap intervals and the slice tables (rating, time control, clock, ply, format) | "Where the lead comes from" |
| `maia3-bench/flops.py` | Per-move FLOPs (Maia-3 measured with torch's FLOP counter, ours analytic) | GFLOPs/move |
| `distill-v2/topk_moe.py`, `topk_score.py` | Runs a checkpoint with only its best K of 16 routed experts per token | fewer experts, no training |
| `distill-v2/report.py` | Frontier table and paired differences for every scored model | results table, Pareto plot |
| `bigrun-progress/plot_rating.py` | Accuracy and CE by game rating, reweighted to the natural player mix | rating plot |
| `sweep/readout.py` | IsoFLOP fits, the L(N, D) law and the MoE compute multipliers of the 45-run sweep | scaling section |
| `maia3-bench/search/` | Tree search on the MoE: `moe-oracle/oracle.py` serves the model to `search/`'s engine; `run.py`, `fit.py` (per-model calibration on a disjoint dev split), `report.py`, `bigrun.py`, `ladder.py` | search gains |

The fewer-experts fine-tunes, the distilled students and the second anneal were trained by the launchers in [../runs/distill-v2](../runs/distill-v2).
