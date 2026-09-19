# Search recovery — 2026-09-19 after preemption

## Authority

Goal active, explicitly resumed by Yiming. Find one frozen method reaching both10× golden macro and10× expert-macro training-equivalent CM, with an improved average NN-node/quality frontier. Current exploratory best is only~3.08×/~5.23×. Do not reset or mark complete. Reproduce on B₂ only after this fixed-checkpoint goal.

Latest constraint: **NO EXTERNAL MEMORY**. No retrieval, game books, cross-game player profiles or episodic datastore. Allowed: fixed model, current-game context, rules and query-local tree/KV. Cached NN outputs for the same queries are research artifacts, not external-memory inputs. No neural-weight changes or Stockfish.

Read controller/STOP and results/search-v1/STOP before work. Neither existed at last check. Never cancel tunnel/controller/Claude jobs or edit main. Own worktree: /data/group_data/dei-group/yimingz3/allie/worktrees/search-v1. Own status/ledger: results/search-v1/status.json and job-receipts/. Preserve all charges, including idle allocations.

## Current execution

- ONE job10504460 search-engine RUNNING on A10080G/babel-y9-24; restarted2026-09-19 19:56:19UTC. Previous Ada allocation was preempted. Keep both charges. Attempted pending Features update was rejected after restart; no change took effect.
- Current enginePID2530027; ready_unix1789848043. Cold stage161.9s plus process startup98.1s. Logs engine-10504460.log. No other search GPU submitted. General/dei belong to Claude.
-101 numerical audit,102/104 header intervention,103/105/106 critic precision all DONE. No new golden queued.
-107 portfolio smoke PASSED all7 components, fresh single1000 Q/root/counts exact versus same-A100104 control.108 full portfolio RUNNING/queued. Atomic256-root blocks; inspect queue result/error before any new work. Never duplicate.
-Existing search.advance --watch preserves every allocation charge. Previous Python gate/observer sessions are finished; do not restart them.

## Current findings and next action

- A100 fixed1000 repeat is bit-identical on all4096 roots. Fixed permutation Δmacro/expert−.000134/−.000321; adaptive2000 permutation−.000052/−.000041. Both below.001 aggregate threshold, while individual policies vary. GPU migration Ada→A100 shifts expert+.00208(control)/−.00158(adaptive); compare same-hardware controls. aug-deep-order-audit-v1/{REPORT.md,results.json,paired-ci.json}.
- A100 checking fixed1000 CE1.408722/1.228093(~972NN), adaptive2000 CE1.403339/1.217313(~1848NN). Pairedmacro−.00538 CI[−.01021,−.00044]; expert−.01078 CI[−.02476,+.00216]. More expensive, so no dominance claim.
- CPU convergence router study DONE: add64→128 causal convergence features between POSITIONS. Both CV metrics choose unchanged state router at512/1000/2000. Same-state baselines recovered exactly. No live promotion. aug-convergence-router-v1/REPORT.md.
-103 fixed-prefix precision diagnostic: only3WDL readout rows recomputedFP32, other logits bit-identical and hook restored in finally.99th-percentile batch-size value differences~.038→.0097, maximum~.059→.013. One target-selected debugging tree, not representative quality evidence. No new inference default adopted.
-104 aug-conditioned-search-v1: actual root policy/quotas fixed. Entire imagined continuation uses actual/common+200/common+400/equal2400. Shift preserves rating gap; separate hypothetical-header prefill prevents mismatched KV.1000simulations, all NN calls and additional root/prefill work charged. Fourarms,4096August positions, modern output fit inside gameCV. Inspect result/error and per-cell deltas; no automatic golden promotion.

104 is complete: all actual-control Q/root/counts equal101; no resolved header win.106 full FP32 critic search complete: macro1.408817/expert1.229608 vs BF16 1.408829/1.228056, ~972nodes,109vs105seconds for4096. Both checking intervals include0; no default change.

108 aug-portfolio-search-v1: fresh single1000, half500 c2.5, half500 c2.5 globally permuted, c.5/5 and c1/4 components. Fixed equal-Q portfolios, both inherited scale16 and count-normalized8 variants. Identical root quotas for unpermuted500 arms make visit-weighted and plain means identical; assertions check this. Refit each output inside the same gameCV. All duplicate NN calls and both prefills charged. Same-c permutation is a numerical-ensemble control. No union cache infrastructure yet. Report ALL arms and per-cell/game CIs, no golden CM on August. Inspect results before deciding next algorithm.

All development is reused August, potentially training-seen. Current goal10x/10x remains unmet. Keep July raw anchors/laws fixed, fresh disjoint golden confirmation required for final claim. Do not turn stability probes into a new quality claim.

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
