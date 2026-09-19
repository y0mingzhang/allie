# Search recovery — 2026-09-19 after preemption

## Authority

Goal active, explicitly resumed by Yiming. Find one frozen method reaching both10× golden macro and10× expert-macro training-equivalent CM, with an improved average NN-node/quality frontier. Current exploratory best is only~3.08×/~5.23×. Do not reset or mark complete. Reproduce on B₂ only after this fixed-checkpoint goal.

Latest constraint: **NO EXTERNAL MEMORY**. No retrieval, game books, cross-game player profiles or episodic datastore. Allowed: fixed model, current-game context, rules and query-local tree/KV. Cached NN outputs for the same queries are research artifacts, not external-memory inputs. No neural-weight changes or Stockfish.

Read controller/STOP and results/search-v1/STOP before work. Neither existed at last check. Never cancel tunnel/controller/Claude jobs or edit main. Own worktree: /data/group_data/dei-group/yimingz3/allie/worktrees/search-v1. Own status/ledger: results/search-v1/status.json and job-receipts/. Preserve all charges, including idle allocations.

## Current execution

Update:10504460 restarted on A10080G/babel-y9-24 at2026-09-19 19:56:19UTC. Runtime archive is staging in node-local scratch; old ready.json remains stale until startup finishes. Attempt to widen pending job Features was rejected because the job had started; no change took effect.

- ONE job10504460 search-engine, preempt. It completed request100 on RTX6000Ada/babel-x5-20, then was preempted. Same job auto-requeued and PENDING; old ready.json/PID3839958 is stale. Never infer live access from it. No second search job submitted.
- Request101-deep-order-audit is queued, not yet started. All through100 are DONE. It will resume on the existing job. Check receipts before doing anything.
- Logs results/search-v1/logs/engine-10504460.log; queue engine-queue/. Observer search.advance --watch reconciles repeated sacct allocations without model wakeups. Preserve all charges. General/dei belong to Claude.

## Current experiment and next action

aug-deep-frontier-v1 cached screen098:128/256/512/1000/4000/16000 budgets, nested game-cross-fitted calibration targets for routing. Features use only raw root and paid128 search. State routers win both CV metrics at all3caps. Same4096 August cohort, no CM conversion.

aug-deep-frontier-live-v1 requests099/100 DONE. Checking results:
- fixed1000:1.408161/1.226010;971.9NN;66.6s all4096.
- state512:1.409917/1.225791;505.6NN;39.6s.
- state1000:1.407338/1.225857;1012.7NN;103.8s.
- state2000:1.403933/1.218888;1847.2NN;231.3s.
All paired CIs against live fixed1000 cross0. No established dominance.

Issue: cached64-root vs live256-root fixed1000 expert shift−.002144 exceeds .001 threshold. One checking move(index3463) contributes−1.096738CE/512, almost entire shift. Raw root logits equal; descendant values diverge. Sources/calibration reconstruction correct. Diagnostic: search/diagnose_deep_drift.py, output drift-diagnostic.json in live study. CPU deterministic staged128→mixed budgets audit exact; not proof of GPU invariance.

101 now runs same256-root fixed1000 twice plus a fixed shuffle, and state2000 repeat plus shuffle, all on the new allocation. Within-allocation contrasts isolate ordering; comparisons with the old live study include hardware migration. Frozen coefficients, no retuning. Logs every descendant oracle result for diagnostic position3463 in both fixed repeats. Output aug-deep-order-audit-v1. Inspect source and results, distinguish raw kernel drift from search amplification. If aggregate drift>.001 persists, investigate before any golden promotion. No new golden evaluation is queued. Do not edit the audit once its plan freezes/runs.

102-conditioned-search-smoke is queued after101; plan frozen at aug-conditioned-search-v1/plan.json. Fourarms:actual,common+200,common+400,equal2400. Actual root policy/quotas fixed; changed header affects full continuation priors/critics. Fresh hypothetical-header prefill prevents inconsistent KV; actual+counterfactual prefills both charged. Shift uses same bounded delta for both players to preserve rating gap. Smoke actual must match the new-allocation audit control exactly. Do not queue full103 until102 passes. No golden until audit resolved.

Next scientific direction must retain no external memory and report small ablations as variant evidence rather than dismissing entire families. Current GPU wait can be used for math/review and cached CPU studies.

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
