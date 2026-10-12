# Analysis scripts

The scripts behind the results the docs cite, kept as they ran (they read and write our group storage):

- `search-bench/`: tree search on the Maia-3 benchmark positions with the MoE oracle (`run.py`), per-model output
  calibration on a disjoint dev split (`fit.py`, `build_dev.py`), tables and plots (`report.py`, `bigrun.py`,
  `ladder.py`), benchmark positions and fixed-budget search (`adapter.py`). `docs/make_figures.py` reads
  `report.py ann2`'s `report-ann2.json` (Allie 2.0) and `run.py`'s `ann2/legal/*.npz`.
- `frontier_report.py`: the benchmark tables and cost plot of every scored model against Maia-3 and Allie 2.0
  (`tables.md`, `report.json`, `pareto.png`), including the annealed model's numbers in DETAILS.
- `recent_share.py`: a recipe table's estimated token share from recent Lichess months per format x Elo cell, and the
  passes a `data.mix` recent tail (`recentNN(YYYY-MM[,p=P])`) gives each bucket's recent games against the table's cap.
- `elo_strength/`: the calibrated bot's rule (`allie.lichess.calibration`). Positions from held-out games
  (`calib_positions.py`), Stockfish on every legal move (`calib_mpv.py`, with `play.py`'s engine wrapper), the
  coverage ladders and the output calibration of `calibration-allie-2.0.json` (`calib_search.py`,
  `calib_outcal.py`), the KL export's pruned budgets (`calib_kl_subsets.py`), and the joint fit with the README
  figure's data (`calib_unified.py` on `calib_cells.py`, `calib_fit.py`, `calib_rule.py`, `calib_plotdata.py`).
- `lichess/`: the bot's checks and measurements: its model against the training forward (`parity.py`), speed
  (`speed.py`), search cost under load (`search_load.py`, `search_profile.py`), tree reuse between moves
  (`reuse_fraction.py`), human think times and resignations behind `behaviour.json` (`human.py`) and chat dry
  runs (`chat_dryrun.py`).
