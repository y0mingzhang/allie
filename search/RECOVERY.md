# Search worktree recovery

Codex owns this worktree (`codex/search-v1`), GOAL.md, search/ and private
results/search-v1/. Claude owns main and training/data/model research. Read
GOAL.md and /data/group_data/dei-group/yimingz3/allie/controller/STOP first.
Never submit training, stop the tunnel/controller, or change Claude's jobs.
The UI goal is stale/paused; do not reset it. Our authorized goal is in GOAL.md.

## Current state: 2026-09-19 ~07:55 UTC

Active: fixed-checkpoint10x/10x golden training-equivalent CM + improved node/CM
frontier. NOT achieved. Latest real golden quality: cpuct2.5 soft1000/Elo,
CE1.46103765/1.36915556, CM2.33908x/3.35045x at971.869 mean logical nodes.
Routed cached stopping:613.818 nodes, CE1.4635142/1.36915556,
CM2.2527x/3.35045x. Actual cpuct2.5 mixed execution still pending.
After current goal succeeds, reproduce frozen winner on Claude B_2; do not
swap models or target law now. User requests table after experiments; no images
unless asked. August tuning is user-approved potentially training-seen.

One general L40S job10498670 on babel-t5-32,04:52UTC through~12:52UTC.
Tmux socket search-v1-10498670,session oracle,GPU0. EnginePID1232195,
capacity786432,mem_fraction_static.5. Verify ready.json; do not restart a ready
engine or submit a duplicate GPU. Queue requests MUST use temp+atomic rename.
Runtime: /scratch/yimingz3/allie/search-runtime/sglang-0.5.9-torch2.9.1-cu128.
Observer2682849 owns status.json; all wall/idle/startup/failure time charged.
No Slurm or shared main mutations this turn; no outstanding peer reviews.

## Immediate work

039-deep-selection is running on SAMEGPU: aug-deep-v1, ordinarycp2.5 zeroFPU,
4000sim with64/256/1000/4000snapshots,160roots/batch. Conservative KVbound max
650012<786432 checked before queue. Driver repeats512-root1000 identity then
uses160-root actual batches. Own1000snapshot is matched-numerics control.
After completion: OMP_NUM_THREADS=1 <runtime>/bin/python -m
search.engine.analyze_deep_selection aug-deep-v1. This analyzer uses each
variant's own logits and reports root drift vsold512batch; no silent baseline
replacement. Current worker may take10min. Routine watcher assigned039.

038-soft-selection DONE; soft_cp25 and soft_cp5 output in aug-soft-selection-v1.
FitCV pickssoft_cp5 for both; confirmation1.456085/1.358828 at976.63nodes.
OrdinaryzeroFPUcp5 was1.456614/1.357294; difference tiny/mixed and FPUconfounded.
At matchedbootstrapFPUcp2.5, newsoft CE1.459466/1.365697 is WORSE than ordinary
bootstrapcp2.5 CE1.456975/1.359714. No established benefit; do not promote.
Incremental softbackup exact CPUreference and livebaselineidentity passed.
First unit-test failure was reference-test mass double count, fixed and rerun;
keep both logs. Current code committedf7eeae0.

040-coverage is QUEUED behind039 on SAMEGPU:
coverage.cpp/coverage_native.py/coverage_pilot.py. Root quotas maximize
prior**eta/(1+n), weights sqrt(p),p,sqrt(p*(1-p)), with ordinaryPUCTcp2.5 zeroFPU belowroot. Existing
nodecounter and outputbackups unchanged; no terminal skipping. CPUunit tests PASS: disabled-mode exactbaseline,FPU sign, independentrootquota
choice at every64step for all3modes. Logs/coverage-test-v2.log. Queue040 exists:
DO NOT duplicate. Expected~7min after039. Analyzer is standard
analyze_selection aug-coverage-v1. All3arms share samecp2.5/zeroFPU interior.


CPUinnovation check DONE38.9s including dataread. Puredeep-minus-oneplyfailed:
CE1.48037/1.42440 vsbaseline1.457688/1.359266. Learneddeep+one selected onCV
butconfirmation1.458048/1.362592 givesnoheld-outwin. Dropped. aug-selection-v1/
innovation.json; report all5arms. First CPUanalysis wasterminated before result
because its innerloop repeatedlydecompressednpzarrays; fixed toloadonce and
vectorize rootchildselection. Bothlogs retained, noGPUwork lost.

035-selection-wide DONE414.34s. FitCV selects zero_cp10_1000_elo for macro and
bootstrap_cp25_1000_elo for expert. Confirmation1.456151/1.359711 and
1.456975/1.359714 respectively. Gains vs originalcp1.25, but incremental CIs vs
cp2.5 overlap zero; paired-vs-cp25.json records this. No golden promotion yet.
036-golden-explore DONE227.45s scoring +0.821s analysis (total260.66s).
Frozen August cp2.5 winner, fixed1000 and Elo routing form. Quality above;
paired gain vs cp1.25 CI macro[-.00800,-.00311],expert[-.01507,-.00050].
All output params fit on August; no golden retune. Reused golden sample.

037-permutation-audit DONE. Frozen actual cap768 cp1.25 router, same512batch,
fixed permutation seed1938043. OriginalCE1.46792797/1.37614560,
permuted1.46817144/1.37701178; shifts+.00024346/+.00086618.
Paired game95% CIs[-.00000584,+.00051353] /[+.00010922,+.00168712].
Expert numerical effect detectable and near.001 threshold. Both orders still
significantly beat4ply. Do NOT pick better order. Maxpolicyabs.0793,meanKL9.1e-5,
maxKL.0934; raw-rootmeanKL1.69e-5,maxKL.00564. Final winner needs own order audit
or batch-invariant inference before fresh claim. Scoring150.69s; analysis78.74s
included NFS stall (processD/rpc_wait_bit_killable). No traceback or lost results.
Claude notified. Avoid calling this a variance estimate based on one permutation.

CPU latent-budget mixture DONE3.08s. Nested gameCV components0/16/64/256/1000,
Elo/time/state/shuffledtime gates; selected EXISTINGfixed1000 bothmetrics.
No promotion; aug-search-v1/latent-budget.json. Removed dead ifFalse expression
and verified exact result equivalence; latent-cleanup-equivalence.json.
CPU state-dependent calibration DONE4.09s on cp2.5 trees. Shared contextual
coefficients with Elo-specific basealpha/beta, fit/preprocess within gameCV.
MacroCV selects state, expertCV value_scale. Confirmation1.454969/1.357212 and
1.455569/1.358288, but paired CIs vsbaseline overlapzero. No promotion yet.
All optimizer fits converged. General state arm permits a few negative betas;
interpret as empirical logit correction, not universally positive rationality.

Earlier034 actualcap768 passed (613.7nodes,CM2.109/3.054); cached-vs-live
maxpolicyabs.2404 is preserved in original report.037 is the controlled same-
batch-size/order audit and does not replace that finding. Earlier033 FPU pilot
pickedcp2.5;031asymmetricbackup and032strength-guidance were unsuccessful.
Complete experiment tables: REPORT.md; node frontier: FRONTIER.md; all golden
methods including duplicate controls: GOLDEN_METHODS.md. Regenerate via
python -m search.report / search.frontier / search.golden_registry after results.
Redirect verbose generated tables to logs. Full details in results subfolders.

Next: analyze038, pursue real algorithm improvements with dev selection before
golden. Current10x thresholds remain far off. Final success requires frozen
winner, fresh disjoint-game confirmation and correct sampling-frame/raw anchor;
test/test_expert remain unopened. Never infer readiness from cached-only costs.

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
