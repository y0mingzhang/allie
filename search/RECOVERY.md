# Search worktree recovery

Codex owns this worktree (`codex/search-v1`), GOAL.md, search/ and private
results/search-v1/. Claude owns main and training/data/model research. Read
GOAL.md and /data/group_data/dei-group/yimingz3/allie/controller/STOP first.
Never submit training, stop the tunnel/controller, or change Claude's jobs.
The UI goal is stale/paused; do not reset it. Our authorized goal is in GOAL.md.

## Current state: 2026-09-19 ~06:40 UTC

Active goal: BOTH >=10x golden macro and expert-macro training-equivalent CM,
plus improved mean search-node / CM frontier. NOT achieved. Fixed checkpoint
and law snapshots remain unchanged. User wants sustained creative research,
fast iterations, a table after each experiment, and images only when requested.
Latest follow-up: AFTER this goal succeeds, reproduce the frozen winner on
Claude's B_2 model; direct transfer first, later tuning separately. Current
checkpoint/targets stay fixed until then.

Resource: one general/normal L40S job10498670 on babel-t5-32, running from
04:52UTC with8h limit. Tmux socket search-v1-10498670, session oracle; resident
engine PID1097060. Keep only one GPU, within Claude's shared8-normal cap.
No Slurm submissions this turn. Independent workbench survives controller restart.
Accounting observer2682849 on controller, singletonflock; inspect before recovery.
All previous usage preserved in status.json/job-receipts; main ledger untouched.

Runtime archive and node-local STAGED.json are complete:
/scratch/yimingz3/allie/search-runtime/sglang-0.5.9-torch2.9.1-cu128.
Use its bin/python on GPU node for CPU helpers too; add chessmix-overlay to
PYTHONPATH only for Parquet builders. Do not restart a ready resident engine.
The original cold-NFS startup cost is charged, not erased by local staging.

## Immediate task

LATEST: August023 and both CPU analyses are DONE. Worker289.63s; main calibration
39.67s, distributional13.32s. Fit-CV selects MCTS1000 reverse/Elo for macro and
softtau.1/1000 forward/Elo for expert; combined Q+softtau.5/Elo in the second
family. Confirmation CE: legal1.52313/1.50320;4ply1.48301/1.40864;
MCTS1000reverse/Elo1.47013/1.37818;soft1000/Elo1.46508/1.37369;
combo1.46834/1.38287. Draw/outcome-consistency not selected. All reported.
Most promising cost point: soft256/Elo1.47530/1.39715 at250.5nodes vs4ply835.7.

CPU unvisited-action fallback experiment DONE (34.34s), unvisited-results.json.
No clear both-metric win: ordinaryMCTS1.47006/1.37773 vsrootcriticpseudo1
1.46846/1.38251. Reported. Its MCTS-root prior has a tiny BF16 batching shift
relative to the shared32-root prior used by the initial August analysis:
root-batch-audit.json mean raw CEdelta -0.00009694, KL0.00002925. Not algorithm gain.

NEXT/RUNNING: frozen July study golden-augcal-v1, 11methods selected solely by
August fitCV, budgets16/64/256/1000 on one shared tree. Queue024 failed reading
a briefly empty NON-ATOMIC request BEFORE any scoring. Error retained. Retry
025-golden-august.request.json published atomically, identical spec/frozen plan.
ALWAYS publish queue requests by temp+rename, never apply_patch directly.
Watch025 and golden-augcal-v1/results.json. Routine watcher watch_search_queue
has the task, polls60s. The worker automatically runs analyze_augcal after scoring.
Raw compact root/Q/soft/visits/WDL caches retained this time for future CPU-only
calibration tests. Golden methods and coefficients are frozen in plan.json;
do not edit its hashed inference/analysis sources while running.
New scorer uses vocabulary-order tie breaking to match the canonical evaluator;
also regresses old raw/legal/MCTS reverse scores against021 on every block.

After025: report CE/CM/node table and paired CIs vs4ply, updateREPORT/FRONTIER,
commit current own code. Do not declare 10x success without actually meeting it.
August raw outcome experiments are model-training-overlap development only.

User chose AUGUST tuning, even if training-seen, instead of insisting on another
held-out development set. July golden remains unchanged. Existing dev/dev_expert
are blitz-only, and Elo calibration trained there failed golden transfer.

August build DONE: results/search-v1/aug-tune-v1/{manifest,strat,clocks,feats,games}.
Source August2026, same cell/mask/tokenization rules, ~25–29K moves/cell.
May overlap model training. Do NOT call it model-held-out or convert its CE with
July's law. sample.json has4096 positions (128/cell/fold),2978 games, game-disjoint
parameter-fit and confirmation folds. IDs/test documents excluded, golden exact
token duplicates excluded, provenance hashed. No labels/future enter inference.

Queue023-august runs search.engine.balanced_dev_pilot, writes aug-search-v1:
- fixed continuation depths1..4, widths4/2/2; per-root node counters added to
  tree.py; independent parity test against original passed at all depths/mate.
- identical repaired MCTS tree snapshots16/64/256/1000; alternative expectation,
  soft/minimax backups on the SAME tree, per-root node costs.
- full categorical WDL visit means reconstructed in new isolated native module
  distributional.cpp. Exact original scalar Q/tree, probability mass, terminal
  and node counts checked. Existing binaries untouched. No extra model nodes.

Tmux window august-research runs search.engine.august_chain, log
results/search-v1/logs/august-chain.log. It waits for023, then runs analyze_august
and distributional_policy on CPU and report. Singletonlock; inspect before restart.
The analysis source hashes were frozen in the request before new scoring.
Do not edit those three analysis files mid-experiment.

analyze_august: forward/reverse policy calibration, global/Elo/format groups.
Only fitfold0 supplies parameters and3-way gameCV selections. All arms reported
on fold1; paired whole-game CIs for named controls/CV selections.
fit_policy.py uses tested analytic implicit gradients for reverse-KL normalization.
distributional_policy.py tests Q+draw/risk/root-outcome-consistency and combined
backup signals, SAME node budgets. Coherent root/children leave Bayesian ratio
unchanged; normalization tested. No training or external chess evaluator.
After023: inspect errors/results, report concise table with CM pending, update
REPORT.md. Choose subsequent golden policies from fitCV, not golden scores.

The unused strat-dev-v1 dataset also exists: exact July golden reconstruction
passed, disjoint unused July for15cells plus unseen2024-04..07 expert-classical.
July expert-classical golden fraction is1.0, so no disjoint July games exist.
This temporal fallback was validated but user then chose August. It has NOT been
used for parameter selection/scoring. Preserve its provenance; do not mix samples.

## Fixed checkpoint/evaluation

Checkpoint r2-3e16-control-t20-w20-pf052h-s42, last.pt resolves to
checkpoints/step-00002274-9a5e6eb1/model.pt under durable main results/pretrain.
SHA3739d0d90f2ea826874ddb29a0ec8f99e28a0ef843ab8f0e1e6d9f4f7f9b5cc5.
Frozen source data-v1-round2/source-ours. Move/strength headers unchanged;
no clock/Elo/continuous side-inputs, but time and WDL output heads present.
Model ~34Mtotal/25Mnonembedding,8x512. Model training stores only data-v1.

Golden is unchanged strat-eval-v1, SHA94530acf164812ba1123256fb7ba975e5270dcde9444cded497fcc3ca2ad8d2f.
Exact16cell macro, expert4cell macro;1,553,058moves/338248expert/26278games.
Official raw macro1.524007755681445, expert1.476428895813994.
Training CM laws in training-cm-laws.json are immutable vertically anchored
isoflop shapes, ND3e16. CM is conditional training equivalence, not speedup.
10x targets macro<=1.3832044316765544, expert<=1.2992929937638005;
see goal-targets-10x-both.json. Law uncertainty not covered by game bootstrap.

Golden sample golden-balanced-v1/sample.json:8192=512/cell,6752games,
seed1926731, SHA6e3bfafbdceaf473eb070dd1a426aefadf0e11dc314d2bac2aeb07728b6f59ff.
Estimator exact canonical-raw cellCE + paired sample(method-canonical), then
16/4cell mean. Whole-game bootstrap. Full raw/port/legal/search chain reported.
All policies frozen before golden.021 reuses this already reported sample;
never call it newly untouched. Final test/test_expert scoring remains unopened.

## Completed results

012 golden baseline suite (results golden-balanced-v1/results.json):
- legal1.514130/1.464733,CM1.1279/1.1200,0nodes.
-2ply1.495308/1.431265,CM1.4361/1.5833,147.1nodes.
-4ply1.477565/1.401889,CM1.8331/2.2127,844.1nodes.
-releasedadaptive50 CM1.1299/1.1110; repairedfixed50 CM1.1836/1.1864.
Warm suite283.5s,4ply173.6s; CPUanalysis8.8s.

021 frozen MCTS1000 golden:
-reverse1.480187/1.395653,CM1.7662/2.3859;970.926nodes;311ssearch/6.43sanalysis.
-forwardCM1.2615/1.6917.
-Elo-calibrated output failed transfer:CM1.3719/1.2658.
No statistically supported domination of4ply. FRONTIER.md/json has nodes/CM;
point dominance is not significance. Expert subset costs unavailable in older
cache block totals; new instrumentation records them per root.

Old blitz-only development experiments (CM not applicable):
014 MCTS1000 reverseCE1.430671/expert1.396336 (~85ssearch).
015 sixply1.434642/1.400756;231.7s,~8.17Mnodes.
016–018 adaptiveexpectation1000 not better thanMCTS1000.
019 MCTS4000 forward1.425956/1.376713;589s;smaller batches harmedthroughput.
020 predicted-time budget retry: matchedtotal2,048,000sims; predicted allocation
1.430423/1.395646 vsfixed1.430671/1.396336 vsSHUFFLED1.428628/1.393605.
No evidence true predicted time helps relative toshuffle. LiteralGrilllambda
hurtsCE; calibratedlambda needed. Allie formulation review in ALLIE_REVIEW.md.

022 shared-tree backup ladder DONE/REPORTED:
94.36s for2048roots, all16/64/256/1000snapshots and6backups. At1000/commoncal:
native1.430671/1.396336;expectation1.454060/1.421198;softtau.1 1.444358/1.412361;
minimax1.462296/1.446491. No1000win with shared calibration. Exact old tree/cache
identity passed. Small-budget gain is confounded by1000-budget calibration.

CPU residual-policy-dev: mixingMCTS+sixply improvesblitz but costsboth trees.
Same-node distributions/backup mixtures are the next attempt to keep gains cheap.
SEARCH_MATH.md contains coherent-policy/projection ideas and reverseKLtailbounds.

## Claude reviews

Round3 data review DONE at209bbc0; metadata/pinning nits fixed1c33cfa.
Tier1 model focusedreview DONE; poolingB2 SHA/fresh provenance/boardMAC fixes.
Tier2 focusedreview DONE after fixing MoE eval capacitydrop/batchdependence and
binding identityproof to control sources/evaluator/B2/numerics/freshness and every
25-step log through300 (post split/resume). Real pilot/unit/per-variant checks
remain Claude's launch prerequisites. Latest gate-FP32/stats delta no objection.
No shared files changed by us. Historical FP8 failure/code paths sent to perfagent.
Phone via tools.mcp__phone_a_friend__phone for sharedstate or costly review questions.
