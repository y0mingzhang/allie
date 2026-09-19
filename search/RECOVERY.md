# Search recovery — 2026-09-19 19:37 UTC

## Authority

Goal active, explicitly resumed by Yiming. Find one frozen method reaching both10× golden macro and10× expert-macro training-equivalent CM, with an improved average NN-node/quality frontier. Current exploratory best is only~3.08×/~5.23×. Do not reset or mark complete. Reproduce on B₂ only after this fixed-checkpoint goal.

Latest constraint: **NO EXTERNAL MEMORY**. No retrieval, game books, cross-game player profiles or episodic datastore. Allowed: fixed model, current-game context, rules and query-local tree/KV. Cached NN outputs for the same queries are research artifacts, not external-memory inputs. No neural-weight changes or Stockfish.

Read controller/STOP and results/search-v1/STOP before work. Neither existed at last check. Never cancel tunnel/controller/Claude jobs or edit main. Own worktree: /data/group_data/dei-group/yimingz3/allie/worktrees/search-v1. Own status/ledger: results/search-v1/status.json and job-receipts/. Preserve all charges, including idle allocations.

## Current execution

- ONE preempt GPU10504460, RTX6000Ada onbabel-x5-20. Eight-hour allocation started2026-09-19 19:29:41UTC; may be preempted.10503933 is TIMEOUT. No other search GPU submitted.
- Resident engine PID3839958; ready.json identifies job10504460. Node-local runtime staged successfully. Warm process startup43.3s, plus the separate initial archive copy.
- Runtime: /scratch/yimingz3/allie/search-runtime/sglang-0.5.9-torch2.9.1-cu128/bin/python.
- Logs: results/search-v1/logs/engine-10504460.log. Queue: engine-queue/*.request.json, *.result.json, *.error.json.
- Requests through099 DONE.100-deep-frontier-live is running; fixed1000/state512 completed, state1000/state2000 are the remaining work at this note. Do not duplicate requests. Each method writes atomic256-position blocks.
-099 smoke covered all4methods,256positions each;29.8s.100 requires its passed receipt.
- Own observer search.advance --watch reconciles sacct without model wakeups. Ledger~13.19GPUh at19:37UTC; never reset old entries.
- General/dei belong to Claude. Phone before any future submission; inspect STOP, live jobs, checkpoint/data and ledger first. No current submission needed.

## Current experiment

Study aug-deep-frontier-v1 extends the4096-position August ladder to128/256/512/1000/4000/16000 simulations. Backup count scale16 for low budgets and16*B/1000 above1000.98 completed in42.2s, using cached trees only.

Routing uses root/128-search features, Elo, format and pre-move clock; no future values or target moves. Router training targets are nested game-cross-fitted calibration losses, excluding the training row's own game and any outer held-out fold. All3caps selected the state router by both CV metrics.

Cached checking results (macro/expert CE; actual NN nodes):
- fixed1000:1.408927/1.228153;971.9nodes.
- state512:1.410414/1.227677;505.6nodes.
- state1000:1.407582/1.225820;1011.0nodes.
- state2000:1.403768/1.218879;1846.8nodes.
- fixed4000:1.403314/1.218542;3864.0nodes.

The state2000 point nearly matches fixed4000 at half the nodes, but is not established dominance. The actual live study aug-deep-frontier-live-v1 now runs fixed1000 plus all3routers with frozen coefficients and no retuning. It resumes the same tree/KV after128; all early work counts. It reports policy differences and CE drift relative to cached results. Sources freeze in plan.json; do not edit deep_frontier_live.py or its dependencies mid-study.

CPU audit search/test_mixed_depth.py passed in10.6s: heterogeneous budgets128..16000, zero root, stage128 then grow, exact Q under each normalized temperature, node counts and no-op continuation. Result: runtime/cpu-audit/mixed-depth-check.json. This deterministic-oracle proof does not replace the live BF16 batching audit.

Next: inspect100 result/error; report all4arms and cached/live drift, including actual nodes and wall time. Check normalized-temperature behavior and equal-cost controls before promotion. No new golden evaluation is queued. If live reproduces the development advantage, freeze a small set of frontier candidates and controls before golden; do not select among them using golden.

## Recent completed evidence

- Deep-scale4096,1000/4000/16000,8budget/backup arms: budget-normalized16K wins CV. Checking gain vs1K is0.00671macro/0.01422expert at~15.8× nodes; expert interval crosses zero. No golden CM conversion.
- Causal quota replay: root instability and visit-scaled instability allocations both worsened expert CE at~970nodes. All64chunks recovered original1000 Q/counts exactly. No live promotion.
- Lapse components, rating precision smoothing and value uncertainty corrections did not establish incremental checking gains. No promotion.
- Tactical extension at equal expected nodes and asymmetric own/opponent backup also failed to promote. This rules out promoting the tested variants, not entire families.
- Fit-only diagnostic: early search changes predict later value changes well but barely predict per-move CE improvement. Value convergence is not human prediction.

## Fixed evidence and reporting

Checkpoint: durable main results/pretrain/r2-3e16-control-t20-w20-pf052h-s42/checkpoints/step-00002274-9a5e6eb1/model.pt, SHA3739d0d90f2ea826874ddb29a0ec8f99e28a0ef843ab8f0e1e6d9f4f7f9b5cc5. Frozen source: results/recipe10x/data-v1-round2/source-ours.8×512; no clock input. Serving export under own results/search-v1/serving-export.

Golden original raw anchor1.524007755681445/1.476428895813994.10× targets1.3832044316765544/1.2992929937638005. training-cm-laws.json immutable. Golden sample8192=512/cell; full raw+paired differences; game bootstrap. Existing golden has research reuse; final success requires fresh disjoint games. Test/test_expert stay closed.

August means game month, not experiment time. Both fitting and checking splits are reused development and potentially model-training-seen; split by game. The current4096 cohort is a subset of the16384 inventory. Do not compare their absolute CE across cohorts or convert August with the July golden law. After each experiment show a table and update REPORT/LATEST. No images unless requested.

Useful files: results/search-v1/{LATEST.md,REPORT.md,FRONTIER.md,GOLDEN_METHODS.md}; search/ALLOCATION_NOTES.md. Algorithm-family explainer is complete at results/search-v1/explainer/index.html and marks retrieval excluded.

## Peers / environment

All Claude review requests known at this note are closed. Main files untouched. Phone tool: mcp__phone_a_friend__phone; Claude sees only phone messages. Do not answer acknowledgements.

Controller Python /home/yimingz3/miniconda3/bin/python3.11 has numpy/scipy/torch, but lacks pybind11 and chess. CPU native audits use Torch's bundled pybind11 headers and isolated runtime/cpu-audit binaries; no environment installation. GPU runtime has the normal dependencies. Never compile a3.11binary over the engine's3.12binary.
