# Search recovery — stage paused2026-09-19

## Authority

Goal PAUSED by user-requested stage wrap-up. Own results/search-v1/STOP exists. No new experiments, GPU submissions or B₂ reproduction without explicit user instruction. Workbench10504460 released after the queue completed. See results/search-v1/STAGE_REPORT.md. Historical execution notes below are preserved for reproducibility, not instructions to restart. Find one frozen method reaching both10× golden macro and10× expert-macro training-equivalent CM, with an improved average NN-node/quality frontier. Current exploratory best is only~3.083×/~5.317×. Do not reset or mark complete. Reproduce on B₂ only after this fixed-checkpoint goal.

Latest constraint: **NO EXTERNAL MEMORY**. No retrieval, game books, cross-game player profiles or episodic datastore. Allowed: fixed model, current-game context, rules and query-local tree/KV. Cached NN outputs for the same queries are research artifacts, not external-memory inputs. No neural-weight changes or Stockfish.

Read controller/STOP and results/search-v1/STOP before work. Own STOP exists; the shared controller STOP was not touched. Never cancel tunnel/controller/Claude jobs or edit main. Own worktree: /data/group_data/dei-group/yimingz3/allie/worktrees/search-v1. Own status/ledger: results/search-v1/status.json and job-receipts/. Preserve all charges, including idle allocations.

## Current execution

- Historical allocation: job10504460 search-engine, now CANCELLED at stage wrap-up, previously RUNNING on A10080G/babel-y9-24; restarted2026-09-19 19:56:19UTC. Previous Ada allocation was preempted. Keep both charges. Attempted pending Features update was rejected after restart; no change took effect.
- Current enginePID2530027; ready_unix1789848043. Cold stage161.9s plus process startup98.1s. Logs engine-10504460.log. No other search GPU submitted. General/dei belong to Claude.
-101 numerical audit,102/104 header intervention,103/105/106 critic precision all DONE. No new golden queued.
-107/108 portfolio DONE; every independent-Q pair loses to single1000; both CV metrics choose unchanged. No caching/union infrastructure built.
-CPU Bellman-consistency screen DONE, expanded16384 DONE, promising. GPU request109 golden-bellman DONE: frozen lambda3 plus fresh same-tree unchanged control, old cached control. One shared1000sim tree, no extraNN for projection; records full compact trees for follow-up output-only mechanisms. Do not duplicate or edit its frozen files.

-Existing search.advance --watch preserves every allocation charge. Previous Python gate/observer sessions are finished; do not restart them.

## Current findings and next action

- A100 fixed1000 repeat is bit-identical on all4096 roots. Fixed permutation Δmacro/expert−.000134/−.000321; adaptive2000 permutation−.000052/−.000041. Both below.001 aggregate threshold, while individual policies vary. GPU migration Ada→A100 shifts expert+.00208(control)/−.00158(adaptive); compare same-hardware controls. aug-deep-order-audit-v1/{REPORT.md,results.json,paired-ci.json}.
- A100 checking fixed1000 CE1.408722/1.228093(~972NN), adaptive2000 CE1.403339/1.217313(~1848NN). Pairedmacro−.00538 CI[−.01021,−.00044]; expert−.01078 CI[−.02476,+.00216]. More expensive, so no dominance claim.
- CPU convergence router study DONE: add64→128 causal convergence features between POSITIONS. Both CV metrics choose unchanged state router at512/1000/2000. Same-state baselines recovered exactly. No live promotion. aug-convergence-router-v1/REPORT.md.
-103 fixed-prefix precision diagnostic: only3WDL readout rows recomputedFP32, other logits bit-identical and hook restored in finally.99th-percentile batch-size value differences~.038→.0097, maximum~.059→.013. One target-selected debugging tree, not representative quality evidence. No new inference default adopted.
-104 aug-conditioned-search-v1: actual root policy/quotas fixed. Entire imagined continuation uses actual/common+200/common+400/equal2400. Shift preserves rating gap; separate hypothetical-header prefill prevents mismatched KV.1000simulations, all NN calls and additional root/prefill work charged. Fourarms,4096August positions, modern output fit inside gameCV. Inspect result/error and per-cell deltas; no automatic golden promotion.

104 is complete: all actual-control Q/root/counts equal101; no resolved header win.106 full FP32 critic search complete: macro1.408817/expert1.229608 vs BF16 1.408829/1.228056, ~972nodes,109vs105seconds for4096. Both checking intervals include0; no default change.

108 aug-portfolio-search-v1 DONE: 14arms, bothCV choose single1000. Pairs cost~975NN vs972 and regress macro+.006–.008/expert+.013–.017 with paired intervals excluding0. Exact all-block parent identity and identical unpermuted root quotas. All duplicate queries/prefills charged.

New Bellman projection: search/engine/bellman_projection.cpp, tested independently against dense Gaussian normal equations; terminal constraints, lambda0 identity, and exclusion of future nodes pass. Minimize critic measurement error plus human-policy Bellman residuals. Unknown aggregate variance adds r², output clipped[-1,1]; Gaussian variances are not calibrated uncertainty. Raw critic still drives expansion, so this is output-only. No clock input in this checkpoint.
-4096 screen: lambda3 selected bothCV, checking macro/expert deltas-.00240/-.00494; expert CI includes0. Source for original completed screen is git dfa2245.
-Expanded16384 (includes4096, reused development): lambda0/1/3/10; bothCV select3. Macro delta-.001343 CI[-.002611,-.000081]; expert-.004644 CI[-.007936,-.001559]. Native projection+backup1.873CPU seconds over16384; no newNN, clipping0.0037%. Reports under aug-bellman-projection{,-expanded}-v1/REPORT.md.
-Expanded analysis initially stopped on1.55e-12 fitted-metric identity gap. Exact per-block Q identity passed. Recorded old source/plan and explicit analysis-revision.json allowing only driver change; tolerance1e-9, all final metric gaps <=1.55e-12. No algorithm/parameter/data change or rescoring selection.
-109 golden-bellman-projection-v1 uses own live GPU because old golden score files did not retain compact trees. Parameters frozen from expanded lambda3/unchanged; fresh parent avoids hardware/batch drift. Raw softmax restricted to move vocabulary378:2346, legal and search reported separately. Golden remains exploratory; no target achieved yet. Result: fresh parent CE1.444230/1.339094, CM3.0538/5.1484; lambda3 CE1.443654/1.336986, CM3.0830/5.3168. Paired deltas-.000576 CI[-.001790,+.000575] and-.002107 CI[-.005720,+.001598]; uncertain incremental effect. Both969.8114NN; projection0.774CPU seconds/8192; blocks204.3s, total241.1s. Old cached parent reproduced exactly. Report/registry/frontier updated. No110queued and goal remains active.

Mechanism ablation completed, aug-bellman-mechanisms-v1: root-only CE1.445059/1.315753 is null; no-root1.443863/1.311531 and upward-only1.443749/1.310667 retain the full1.443771/1.311241 gain. Both CV criteria retain full. Dense-reference tests and unchanged/full action-value identity passed. No golden opened.

Final CPU experiment search.bellman_variance COMPLETED. Seven arms, both CV criteria retain uniform lambda3. Confidence weighting did not improve move CE. Existing August outcome labels prove a useful diagnostic: root expected-score BCE improves .587065/.546359→.583225/.541840, paired intervals favor both; labels never enter predictions or method selection. Reports are in aug-bellman-{mechanisms,variance}-v1. No110GPU request exists. No unfinished research process needs recovery.

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
