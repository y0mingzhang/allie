# Analysis scripts

The scripts behind two results, kept as they ran (they read and write our group storage):

- `search-bench/`: tree search on the Maia-3 benchmark positions with the MoE oracle (`run.py`), per-model output
  calibration on a disjoint dev split (`fit.py`, `build_dev.py`), tables and plots (`report.py`, `bigrun.py`,
  `ladder.py`), the SGLang adapter for dense models (`adapter.py`).
- `frontier_report.py`: the benchmark table of every scored model (`report.json`, which `docs/make_figures.py` reads).
