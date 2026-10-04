# Analysis scripts

The scripts behind two results, kept as they ran (they read and write our group storage):

- `search-bench/`: tree search on the Maia-3 benchmark positions with the MoE oracle (`run.py`), per-model output
  calibration on a disjoint dev split (`fit.py`, `build_dev.py`), tables and plots (`report.py`, `bigrun.py`,
  `ladder.py`), benchmark positions and fixed-budget search (`adapter.py`). `docs/make_figures.py` reads
  `report.py bigrun`'s `report-bigrun.json` and `run.py`'s `bigrun/legal/*.npz`.
- `frontier_report.py`: the benchmark tables and cost plot of every scored model against Maia-3 and Allie 2.0
  (`tables.md`, `report.json`, `pareto.png`).
