# Rebasing onto the package layout

`main` moved the code from flat `scripts/` modules (plus `search/`, `bench/`, `runs/`) into one package,
`src/allie`. Behavior is unchanged: the same 30-step 2-GPU run trains bitwise identically from both layouts, and the
evaluator and MoE search give identical scores (details in the commit message).

- **Branches of `main`** made before it: `git rebase main`. Rename detection maps edits of a moved file onto its new
  path, since only import lines, paths and docstrings changed. Exceptions to resolve by hand: the Maia-3 benchmark
  builders `legal.py`, `build_formats.py` and `build_rating.py` now run inside `main()` (importing them no longer
  rewrites benchmark files), `topk_score.py` patches the package's MoE instead of a flat copy, and new files a branch
  adds under `scripts/` need moving (directory-rename detection may put them in `tests/checks/`).
- **`lab`** predates the release trim (15b051f) and keeps files `main` dropped, so it takes the layout by
  cherry-pick without directory renames (else git moves lab-only `scripts/` files into `tests/checks/`):
  `git -c merge.directoryRenames=false cherry-pick <this commit>`. Then `git add` the files the trim added (`analysis/`,
  `configs/`, `src/allie/eval/maia3/`, `pyproject.toml`, ...: "deleted by us"), and resolve `data/mix.py`,
  `eval/score.py`, `experiments/{modelexp,dmix}.py` to this side (the trim's `ALLIE_DATA` overrides) and
  `model/network.py` keeping lab's `moe_dense_first` assert, plus README and DETAILS. Lab-only callers of moved
  modules stay in `scripts/` and need their imports or paths updated: `bigrun.py`, `health.py`, `replay.py`,
  `runindex.py` (and `runindex_loop.sh`), `pass_tokens.py`, `isoflop_strat.py`, `test_memsnap_oom.py`, the
  `chessdata*.sbatch` and `extdata.sbatch` launchers, `codex-controller.sbatch` and `archive/`.

## Paths

| Before | After |
|---|---|
| `scripts/chess_vocab.py` | `src/allie/data/vocab.py` |
| `scripts/chessdata.py` | `src/allie/data/store.py` |
| `scripts/chessdata_annotate.py` | `src/allie/data/annotate.py` |
| `scripts/fastbuild.py`, `scripts/fastbuild.c` | `src/allie/data/fastbuild.py`, `src/allie/data/fastbuild.c` |
| `scripts/extdata.py` | `src/allie/data/external.py` |
| `scripts/chessmix.py` | `src/allie/data/mix.py` |
| `scripts/lm_data.py` | `src/allie/data/packed.py` |
| `scripts/datapin.py` | `src/allie/data/pin.py` |
| `scripts/history_counts.py` | `src/allie/data/history.py` |
| `scripts/pool_inventory.py` | `src/allie/data/inventory.py` |
| `scripts/hf_fetch.py` | `src/allie/data/fetch.py` |
| `scripts/modded_arch.py` | `src/allie/model/arch.py` |
| `scripts/modded_medium_core.py` | `src/allie/model/nanogpt.py` |
| `scripts/modded_medium.py` | `src/allie/model/network.py` |
| `scripts/modded_attn.py` | `src/allie/model/attention.py` |
| `scripts/modded_moe.py` | `src/allie/model/moe.py` |
| `scripts/modded_smoe.py` | `src/allie/model/moe_kernels.py` |
| `scripts/modded_shard.py` | `src/allie/model/shard.py` |
| `scripts/modded_board.py`, `board_encode.cpp`, `move-table.json` | `src/allie/model/board.py`, `board_encode.cpp`, `move-table.json` |
| `scripts/modded_medium_LICENSE` | `src/allie/model/NOTICE` |
| `scripts/modded_train.py` | `src/allie/train/trainer.py` |
| `scripts/modded_wsd.py` | `src/allie/train/schedule.py` |
| `scripts/modded_checkpoints.py` | `src/allie/train/checkpoints.py` |
| `scripts/lm_checkpoint.py` | `src/allie/train/state.py` |
| `scripts/modded_runtime.py`, `modded_runtime_stage.py` | `src/allie/train/runtime.py`, `runtime_stage.py` |
| `scripts/modded_memsnap.py` | `src/allie/train/memsnap.py` |
| (new) | `src/allie/train/provenance.py`: the source hashes a checkpoint records |
| `scripts/strateval.py` | `src/allie/eval/build.py` |
| `scripts/eval_strat.py` | `src/allie/eval/score.py` |
| `bench/maia3-bench/build_positions.py`, `build_formats.py`, `build_rating.py` | `src/allie/eval/maia3/positions.py`, `formats.py`, `rating_sample.py` |
| `bench/maia3-bench/{legal,check_uci,score_maia3,score_moe,flops,aggregate}.py` | `src/allie/eval/maia3/` (same names) |
| `bench/maia3-bench/report_bigrun.py` | `src/allie/eval/maia3/report.py` |
| `bench/bigrun-progress/plot_rating.py` | `src/allie/eval/maia3/plot_rating.py` |
| `bench/distill-v2/topk_moe.py`, `topk_score.py` | `src/allie/eval/maia3/topk_moe.py`, `topk_score.py` |
| `scripts/modelexp.py` | `src/allie/experiments/modelexp.py` |
| `scripts/isoflop_fit.py`, `scripts/dmix_fit.py` | `src/allie/experiments/isoflop.py`, `dmix.py` |
| `bench/sweep/readout.py` | `src/allie/experiments/readout.py` |
| `search/` (top-level package `search`) | `src/allie/search/` (`allie.search`) |
| `bench/maia3-bench/search/moe-oracle/oracle.py`, `parity.py`, `README.md` | `src/allie/search/moe_oracle.py`, `moe_parity.py`, `MOE_ORACLE.md` |
| `bench/maia3-bench/search/*.py` | `analysis/search-bench/` |
| `bench/distill-v2/report.py` | `analysis/frontier_report.py` |
| `scripts/test_X.py` | `tests/checks/X.py`, run by `tests/test_checks.py` (`test_chessmix_recent` is `mix_recent`, `test_modded_zero` is `zero_masters`) |
| `search/tests/` | `tests/search/` |
| `runs/bigrun/` | `configs/allie-2.0/` (`recipes/` → `configs/recipes/`), plus `configs/allie-2.0.json` |
| `runs/distill-v2/ann.sbatch` | `configs/allie-2.0/anneal.sbatch` (`kd.sbatch` dropped) |

## Imports

Package modules import each other absolutely: `import modded_moe` becomes `from allie.model import moe`,
`from modded_medium import Config` becomes `from allie.model.network import Config`, `import chessmix as cm` becomes
`from allie.data import mix as cm`. Inside the package, modules imported whole keep a non-colliding alias:
`model_arch`, `moe_layer`, `moe_kernels`, `expert_shard`, `board_cnn`, `attn_kernels`, `network`. Nothing inserts
into `sys.path` any more; run with the package installed (`uv sync`) or `PYTHONPATH=src`, and launch modules with
`python -m` (`torchrun ... -m allie.train.trainer`, `-m allie.eval.score`).

## Running

- **Studies.** `allie-exp plan ROUND WAVE [COMMIT]` (or `PYTHONPATH=src python -m allie.experiments.modelexp ...`)
  freezes the package into `STUDY/source/allie/` (no more `source-ours/`, `evaluator-ours/` or driver copies at the
  study root); the study's `run.sbatch` runs `python -m allie.experiments.modelexp task` with
  `PYTHONPATH=STUDY/source`. Round files should `from allie.experiments import modelexp as mx`; old ones that
  `import modelexp` still load (modelexp registers itself under both names). Studies frozen before the layout keep
  their flat copies and run as before.
- **Recipes.** A study's frozen `STUDY/recipes/` is still found (now relative to `source/allie/data/mix.py`);
  otherwise `ALLIE_RECIPES` (new), else `ALLIE_DATA/results/recipe10x/recipes` as before.
- **Checkpoints.** Package checkpoints record `source_sha256` keyed by package-relative paths (`model/moe.py`, ...).
  Resuming a pre-package run with the package needs `--resume-new-source` (its recorded sources differ), as any
  source change did. The evaluator, the Maia-3 MoE scorer, the top-K scorer and the MoE search oracle take
  `--source` (a run's frozen flat source) for pre-package checkpoints and the package otherwise.
- **Benchmark scripts** write their outputs under `ALLIE_PROJECT_ROOT/results/recipe10x/...` (they used to write next
  to themselves) and read the Maia-3 code from `MAIA3_REPO` (default: the clone under `ALLIE_DATA/maia3-bench`).
  `ALLIE_PROJECT_ROOT` defaults to the checkout, as for the trainer; `allie.experiments.modelexp` keeps its own
  default, `/home/yimingz3/src/allie`.
