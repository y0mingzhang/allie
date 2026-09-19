# Search worktree recovery

Codex owns this worktree (`codex/search-v1`), GOAL.md, search/ and private
results/search-v1/. Claude owns main and training/data/model research. Read
GOAL.md and /data/group_data/dei-group/yimingz3/allie/controller/STOP first.
Never submit training, stop the tunnel/controller, or change Claude's jobs.
The UI goal is stale/paused; do not reset it. Our authorized goal is in GOAL.md.

## Current state: 2026-09-19 ~07:25 UTC

Active goal: BOTH >=10x golden macro and expert-macro training-equivalent CM,
plus improved mean search-node / CM frontier. NOT achieved. Fixed checkpoint
and law snapshots remain unchanged. User wants sustained creative research,
fast iterations, a table after each experiment, and images only when requested.
Latest follow-up: AFTER this goal succeeds, reproduce the frozen winner on
Claude's B_2 model; direct transfer first, later tuning separately. Current
checkpoint/targets stay fixed until then.

Resource: one general/normal L40S job10498670 on babel-t5-32, running from
04:52UTC with8h limit. Tmux socket search-v1-10498670, session oracle; resident
engine PID1232195 (verify ready.json). Keep only one GPU, within Claude's shared8-normal cap.
No Slurm submissions this turn. Independent workbench survives controller restart.
Accounting observer2682849 on controller, singletonflock; inspect before recovery.
All previous usage preserved in status.json/job-receipts; main ledger untouched.

Runtime archive and node-local STAGED.json are complete:
/scratch/yimingz3/allie/search-runtime/sglang-0.5.9-torch2.9.1-cu128.
Use its bin/python on GPU node for CPU helpers too; add chessmix-overlay to
PYTHONPATH only for Parquet builders. Do not restart a ready resident engine.
The original cold-NFS startup cost is charged, not erased by local staging.

## Immediate task

Current: 2026-09-19 ~07:25 UTC. Goal 10x/10x still active and unmet.
Keep the original r2 checkpoint until success; then reproduce on Claude B_2.
Latest confirmed golden remains soft1000/Elo CE 1.4665905 / 1.3768662,
CM 2.1510x / 3.0249x at970.926 nodes. Lower-cost soft256 is about250 nodes
and CM1.8706x/2.2849x; strict statistical domination of4ply is not established.

GPU: existing job10498670,1 general L40S,babel-t5-32,ends~12:52UTC.
Tmux socket search-v1-10498670/session oracle,window0. Current enginePID1232195,
capacity786432,mem_fraction_static.5; verify engine-queue/ready.json. Starts from
node-local archived runtime in22.8s total (engine20.1s). Do not start a second
engine. All prior charges including failed026/startups/idle time remain recorded.
Accounting observer2682849 owns status.json; do not race its writes.

RUNNING:033-selection-pilot, search.engine.selection_pilot. Four first-play
urgency/exploration variants: bootstrap, running mean, bootstrap minus.2sqrt
visited-prior mass, and original zero-FPU with cpuct2.5. Same4096 August roots,
512 roots/batch,1000 simulations with64/256/1000 snapshots. One GPU, no newSlurm.
Outputs aug-selection-v1/<variant>/<block>.npz retain full compact trees too.
Unit default path and independent FPU-sign check passed; live first512-root
control must exactly match aug-compact-v1 before variants execute. Check
aug-selection-v1/identity.json and033 error/result. Routine watcher
/root/watch_search_queue polls60s up to12min; no substantive delegation.
After033 completes run CPU search.engine.analyze_selection on GPU node's local
Python with OMP_NUM_THREADS=1. It selects only from fit-gameCV and reports all
arms on August confirmation. Then update report and table to user. No golden
has been opened for these variants. Do not modify hashed running sources.

DONE031: aug-compact-v1 tree cache,129.438s for4096x1000. New compact.cpp and
compact_native.py reconstruct every64/256/1000 snapshot and arbitrary own/
opponent soft temperatures on CPU. Fake-oracle reference, checkmate signs,
unchanged visits/tree, and real-GPU symmetric backups pass to3e-12.
Asymmetric temperature scan0/.025/.1/.5/infinity x both roles,256/1000 budgets,
global/Elo output calibration selected EXISTING symmetric.1/1000/Elo for both
metrics. Confirmation1.465198/1.374123;256control1.475531/1.398580. No promotion.
Analysis22.709s incl11.781s reductions. Results aug-compact-v1/asymmetric-results.json.

DONE032: aug-conditioning-v1, eight root-only hypothetical-rating query arms.
Moved self/opponent Elo digits only; history/time-control identical. Full-prefix
query work charged;3.756s GPU worker,6.592s CPU calibration. MacroCV selected
unchanged soft1000. ExpertCV picked both ratings-200 guidance stacked on search:
CE1.465481/1.372414 vs1.465198/1.374123, pairedexpertCI[-.00779,+.00362].
No convincing improvement; not promoted or golden-scored. This repeats the
previous blitz prompt idea on balanced16cell data and permits stacking.

INFRA030 PASSED: fresh old262144 and new786432 cache at SAME128-root batching
are EXACT for roots/Q/visits.512-root batch improves blocktime38.16 ->26.98s
for~993K node evaluations,1.4146x; root logits exact, internalBF16 branch noise
is reported (meanQ gap.00116,4361 visit entries changed). Performance only.
026 initially failed historical parity on1/1024roots;027/028 proved a fresh
OLD-capacity process has identical rounding difference (meanrawCE5.3e-8nats,
KL3.64e-9). Therefore not a capacity bug. Historical19changedvisit entries
preserved; no numerical gain credited. Artifacts cache-root-diagnosis*.json,
cache-benchmark-{small,large}.json. Never silently delete failures.

August tuning data is aug-tune-v1,4096 positions/2978games,128 percell per
fit/confirmation fold. Both folds may overlap model training (user-approved).
Only unchanged July golden establishes held-out quality/CM. Old dev/dev_expert
are blitz-only. Earlier private July+2024 strat-dev-v1 was built/validated but
never used; no need rebuild. Never open test/test_expert labels for scoring.

025 golden-augcal-v1 remains frozen; do not rerun after changing direct.py.
It retains compact root/Q/visits/WDL caches for compatible CPU-only calibration.
All selected coefficients froze from August fitCV.024 failed non-atomic request
reading before scoring;025 identical retry succeeded. ALL new queue requests
must use temp+atomic rename, never direct apply_patch to *.request.json.

Useful next: inspect FPU results; if no gain, test learned budget allocation
with nested game CV, richer model-only search signals, or more simulations.
Current symmetric backup is a useful control; separate-tau and rating-guidance
experiments are recorded failures, not unreported wins. Report whole-game
uncertainty, node cost, full-prefix work and training-law caveats.

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
