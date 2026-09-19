# Scaled-checkpoint search transfer and Pareto comparison

Owner: Codex, codex/search-v1 worktree. Status: COMPLETE (2026-09-19). The finite transfer evaluation and Pareto write-up are finished. No experiment or GPU submission remains authorized by this goal. A controller restart does not resume it.

Current objective: (1) test whether the previously frozen model-only search gains hold on Claude's finished scaled final data+model recipe, including whether training-equivalent CM diminishes; (2) produce a readable write-up and plots comparing representative methods on average actual NN nodes versus macro/expert CE.

This replaces the old open-ended10x/10x search objective. That stage is closed with its target unmet. Do not restart broad idea hunting or wait for10x before evaluating the scaled checkpoint.

Checkpoints (read-only):
- Original stage: r2-3e16-control-t20-w20-pf052h-s42 (8x512).
- Same-recipe small control: mo1-3e16-dense-pf052h-s42 (8x512; second seed available).
- Scaled ship recipe: msh-3e17-model-pf115h-s42 (16x768,128.8M total parameters,2.61B tokens,4974steps). Frozen source/evaluator results/recipe10x/model-v1-ship3e17. Board CNN, SwiGLU, no key offset, cf3 clock input, time/WDL heads.

First verify the extended inference port against each frozen evaluator, including full-prefix/cached branching, causal board states, clock features and fixed position conventions. No dropping side inputs. Describe imagined child-clock updates explicitly and test their consistency. Keep all new artifacts separate from original frozen studies and export.

Evaluate frozen algorithms/calibration first. If calibration-only refits are needed, use the existing game-disjoint August fitting/CV split, label it separately, freeze before July golden and use no new search hyperparameter sweep. Golden = existing16 format x Elo cell means; expert = four>=2400 cells. Raw/legal baselines, move CE, accuracy, uncertainty and port discrepancies for every model. Same-recipe small-to-large comparison separates scale from recipe changes. Any missing matched law means CM is explicitly conditional, not an observed training multiplier. The old numerical10x targets do not apply to the new checkpoint.

Representative comparison: raw/legal/calibrated direct, shallow fixed-ply, released/repaired Allie adaptive MCTS, current fixed-budget root-coverage search, adaptive allocation, optional critic consistency. Use a small declared budget grid, actual NN nodes, prefix work and measured runtime. Mark point frontiers separately from statistically supported dominance. Freeze methods before evaluation; avoid selecting winners on golden. Existing sample reuse must remain disclosed; any fresh sample uses disjoint games and explicit population/anchor semantics.

No external memory, retrieval, Stockfish, neural-weight changes or new model training. Allowed: fixed model, current game/header/pre-move inputs, chess rules, current query's tree/KV. Use durable storage under /data/group_data/dei-group/yimingz3/allie/worktrees/search-v1. Main/Claude files and checkpoints are read-only.

Resources used: one GPU, job10505672, initially preempt RTX PRO 6000 and then general L40S after Claude confirmed the available normal-QoS share. All three allocations total0.902778GPU-hours; cumulative search use15.727778GPU-hours preserves the prior14.825. All requested results completed; the GPU exited and the pending automatic requeue was cancelled. Own results/search-v1/STOP is present. Peer/tunnel/controller jobs were untouched.

Finish when the scaled transfer evaluation and representative frontier write-up are complete, or explicitly paused/blocked. Keep durable progress and resumable commands in search/transfer/RECOVERY.md. Generate the requested comparison plots as exportable artifacts; no unrelated images.

Completed artifact: search/TRANSFER_REPORT.md and results/search-v1/transfer-v1/REPORT.md, with PNG/PDF/SVG plots. Frozen1,000 search retains a gain at129M, but the CE reduction versus legal falls from0.0517/0.1020 to0.0158/0.0602 (macro/expert). Rung-anchored conditional CM2.22/3.50→1.53/3.98; the expert CM change is unresolved and depends on the assumed law slope. The live adaptive router uses461 rather than966 average NN evaluations at129M, with no resolved quality difference. The prior10×/10× research target remains unmet and its API goal remains paused; this completed finite task does not claim that target.
