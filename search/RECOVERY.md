# Search worktree recovery

Codex owns this worktree (`codex/search-v1`), GOAL.md, search/ and private
results/search-v1/. Claude owns main and training/data/model research. Read
GOAL.md and /data/group_data/dei-group/yimingz3/allie/controller/STOP first.
Never submit training, stop the tunnel/controller, or change Claude's jobs.
The UI goal is stale/paused; do not reset it. Our authorized goal is in GOAL.md.

## Current continuation: faster stack live; deeper golden running

Resident engine now PID1713252, same general L40S job10498670 on babel-t5-32.
Cache1500000 tokens, memory fraction0.8, 30s process startup. Log
results/search-v1/logs/engine-10498670-1500k.log. Tmux oracle has reserve keeper
and engine1500k windows; do not kill the server/session. Old PID1232195 exited.
No new Slurm job or extra GPU. All allocation time remains charged by observer.

046 forest benchmark: exact CPU scheduling identity, but only~1.2x deep speedup.
047 handle bridge:512x1000 13.10s->8.02s (1.63x),160x4000 18.69s->13.08s
(1.43x). Fixed-policy CE drift on timing subsets +.0000075/+ .0000127;
max policy abs .000611/.000157. No golden quality claim from timing.
048 parallel expansion:1/2/4 threads give BIT-EXACT queries and outputs.
2threads~7.14s/12.00s vs8.30s/13.16s;4 gives7.05s/12.03s. Choose2 for headroom.
049 larger cache:1024x1000 ~13.38s (4threads),320x4000~21.11s (2threads).
All timing source/config/results retained; not a same-position quality comparison.

050-fast-golden-coverage is running/queued, NOT duplicate:
golden-fast-coverage-v1, frozen golden_fast_coverage.py.
Actual4000sim,320root blocks,2 expansion threads, forced-move exact skip.
Logical prefix1000 reconstructed from birth indices, with its old output
coefficients from aug-coverage-v1;4000 coefficients from aug-coverage-deep-v1,
selected by both fit-CV metrics before golden. Own raw/legal port and prior
1000 gold controls included. About9min. Routine watcher watch_search_queue
will notify completion/error; report fulltable, uncertainty and cost.
This is reused golden, not fresh final confirmation; higher cost isn't dominance.
No further GPU request after050 yet.

CPU expected-outcome backup ablation completed, all4 temperatures worse than
unchanged regularized soft backup on August. Both fit-CV metrics selected
unchanged control. aug-coverage-v1/behavior.json and executed-source snapshot.
One check needed double-roundoff tolerance1e-9 (actual largest mismatch9e-12);
initial log preserved. Unit independent recursive/reference tests passed.

Next research idea: adaptive root quota proportional to the square root of
the CE sensitivity of the UPDATED policy, mixed with original-prior quota.
Static successful quota only knows the prior. Keep PUCT below root and tested
soft backup. Update weights at block boundaries so root subtrees can still run
in parallel; zero-mixture must reproduce static allocation under exact oracle.
Not implemented yet.

Potential later retrieval axis: Claude has no index; use June2026 or earlier
bank (NOT August, to avoid future-player/style retrieval). Exclude bank games
by full-game token hashes against all golden/dev/test and August tune/confirm
games; same-cell neighbors or header-aware keys; fit mixing on August only.
Disclose extra datastore, build/query memory/cost and retrieval-vs-search gains.
No bank built and no retrieval requests yet. User inference scope allows ideas,
but don't silently claim a datastore is a pure tree-search improvement.

## Previous continuation: parallel root-action scheduler

044 deep coverage completed: August own 1000 snapshot 1.452653/1.357165,
4000 snapshot 1.446924/1.344827 at 3886.63 mean nodes. Paired macro delta
-.005730 CI[-.010462,-.000650]; expert -.012339 CI[-.028694,+.004799].
No golden 4000 yet; higher cost alone is not Pareto dominance.

045 internal allocation completed. Against coverage 1.452666/1.357460:
prior 1.458130/1.367425; soft derivative 1.452756/1.358427;
75% derivative+25% prior 1.452481/1.357144. Selected-arm paired CIs overlap
zero. No promotion. Results and paired-vs-reference.json are in aug-influence-v1.
User received all three-arm table.

Format-specific calibration failed confirmation; aug-coverage-v1/
format-calibration.json. Final all fits converged after coordinate rescaling.
One intermediate rerun executed stale behavior even while recording a new end
source hash (likely NFS/bytecode caching); preserved stale-rerun and first
artifacts. Final ran exec(compile(exact_read_bytes)) with before/after hash
verification. Prefer immutable module names / exact byte execution when editing
producers. This CPU-only issue did not invalidate golden or NN results.

046-forest-benchmark queued/running on existing 10498670, no new Slurm job.
forest.cpp/native.py implement independent root-action subtrees with precomputed
root-quota logical visit times. Each subtree still ordinary cp2.5 PUCT; tree
births record ORIGINAL simulation time, so reductions reconstruct prefix budgets.
CPU deterministic-oracle test passed: exact node/boot/birth/structure, soft values,
mixed/zero budgets, forced actions, terminal and depth limits, and prefix NN counts.
Forest requests can contain many root branches; same cache size.
Benchmark: 512x1000 and160x4000, sequential/forest with and without exact forced-
action policy shortcut. First no-skip forest shows no speedup because forced
moves create a1000-round tail. Wait for full report. CPU identity does not imply
GPU BF16 identity; drift explicitly measured with fixed calibration. This is
timing, not golden quality selection.

New uncommitted files forest.cpp, forest_native.py, forest_benchmark.py,
format_calibration.py, paired_august.py, report/RECOVERY edits.
Engine caches imported modules: don't mutate imported implementation for another
request without a fresh module/library identity. No outstanding peer reviews.
Current best remains actual 043: CM2.60708x/3.99839x,615.40076nodes,147.46s.

## Previous continuation: live coverage verified

043 DONE: actual mixed coverage615.40076nodes, macro1.454045189/expert1.356337615,
CM2.60708x/3.99839x.147.462s scoring,.814s summarize. Vs4ply delta95%
macro[-.029540,-.017477],expert[-.061942,-.030034],at27%fewer nodes.
Cached->live macroshift-.00001747,expertEXACT;maxpolicyabs.00575,
meanKL1.30e-7,maxKL.0001724. MuchmorestablethanoldPUCTrouting.
Stillneedswinnerpermutationaudit/freshgameconfirmationforfinalclaim.

Goldenutilities DONE(CPU only): standard10 CE1.464542/1.373073,
CM2.21806x/3.17971x,worsemacrothan cp2.5;standard20 CE1.461550/1.368086,
CM2.32086x/3.39911x,incrementalCIs overlap0. Dropped; no stacking.
All resultsreported to user; rootcoverage remainswinner.

044-deep-coverage running onexistingGPU. 045-influence queued afterit:
rootquotaunchanged, belowroot allocate byprior alone,softBellman derivative,
or75%derivative+25%prior, each1000sim. Sourcesinfluence.cpp/native/pilot.
CPUtest PASS: disabledchange exactcoveragecontrol, independently recomputed
softvalues and ENTIREselected paths at every step for eachmode, terminals
included. GPUfixturefirst512 will requireidentitybefore3arms. Expected7min
after044. Analyzer standardanalyze_selection aug-influence-v1; must compare
toexistingcoverage_bernoulli1000Elo fromaug-coverage-v1, notjusthistorical
cp1.25 zero baseline. AllfitCV andconfirmationarmsreported,noCMuntilgolden.
MathematicalmotivationandheuristiccaveatsinSEARCH_MATH.md. No validation-
target or future moves inallocation. No newSlurmjobs.

Reports/frontier include043/utility paths; rerunreport/frontier/registry.
Watcherwatch_search_queue notified043finished;checkwhetherstillwatching044
beforeassigning another. Deep4000analysis samegenericpipeline as039.
No outstandingClaude review: tier2b done (auxsumfix), laterpropbias FYIs
notrequestsanddidnotchangeourfiles.

## Previous continuation: root coverage and auxiliary-loss review

Current best actually executed golden: 042 root coverage, CE1.452971275/1.356337615,
CM2.65168x/3.99839x,975.560 mean nodes. Selected sqrt(prior*(1-prior))
root quota by August fit-gameCV; interior cp2.5 zeroFPU. Same1000 sims.
Vs cp2.5 paired delta95% macro[-.011486,-.004449],expert[-.023330,-.001908].
Scoring223.72s, CM analysis.761s. Cached Elo stopping615.4005 nodes,
CE1.45406266/1.35633761,CM2.60636x/3.99839x. NOT goal10x/10x; reusedgolden.
Actual mixed execution is043-dynamic-coverage, queued/running; output
golden-dynamic-coverage-v1. Its live results are authoritative, not cached ones.
044-deep-coverage queued after043, Aug4096 roots ×4000sim,160batch,
same root quota; own1000 snapshot controls batching. Expected~11min. Only
q/logit snapshots saved (no duplicate full-tree gigabyte). Analyze with
python -m search.engine.analyze_deep_selection aug-coverage-deep-v1.
Routine watcher watch_search_queue follows043/044 at60s intervals,20min max.
No new Slurm jobs; existing10498670, same L40S. STOP absent, ledger observer live.

040 root quotas DONE420.55s. FitCV selected Bernoulli quota both metrics:
Aug1.452666/1.357460,978.20nodes. sqrt prior1.451727/1.355295.
Incremental Aug CIs vscp2.5 overlap0 (paired-vs-cp25.json); golden042 then
showed improvement as above. All three quotas and all budget arms reported.
041 history averaging DONE1.41s inference+1.94s analysis.88.65%coverage,
mean3.18variants; bothCVmetrics select unchangedsearch. Arithmetic stack
1.458212/1.358519,geometric1.458022/1.358474 vs1.457688/1.359266.
CIs overlap0; notpromoted. Individual variant CE and percellcoverage saved.
Generator seesprefixonly, exactFEN/legalcommutation, unchangedirreversible
suffix; samepositionIDs. No clocks on r2; B2wouldneedclockconsistent variant.
Sent results to Claude.

CPU utilities DONE2.46s on cp2.5 trees. FitCV macroselectsstandardized20
(confirmation1.455133/1.354195); expertselectsstandardized10
(1.458242/1.353592). Incremental AugCIs overlap0. Eightarms incllinear,
logodds,tanh,cubic,rank,std05/10/20 allreported. BothCVselectedarms frozen
for CPU-onlygolden check in golden-utilities-v1; scriptgolden_utilities.py,
loglogs/golden-utilities.log, maystillrunning (tool session82409).
No new NNnodes; baselinecachedtrees unchanged.
CPU recursivecriticregularization DONE: lambda0,.25,.5,.75 allworse,
lambda1 (unchangedsearch) selectedboth. Values in aug-selection-v1/discount.json.
Independentnative recursion/terminal/missingmass/budget checks pass;
lambda1 bit-exact nativebaseline, productionq agrees3e-12.

Claude tier2b review DONE after fixing normalization blocker. Original MoE
seq balancing usedmean_rows while taskCE issummed, causing2x auxgradient
change when2048tokenssplitinto2x1024. ActualCPUreproconfirmed. Claude fixed
to1024*sum_rows (perinputtoken, inclunscored), explicitworld/8. AddLoss
downstreamgradient-scalecontract documented. Warned dropped-route-adjusted
FLOPs mean usefulassignedFLOPs, notphysicalpaddedBMMexecutedFLOPs.
I sent explicitdone after readingfixandtest. Subsequentproportionalbias/stats
FYI not reviewed; no outstanding requested review on that delta.

All workprivate; no main/training/datasets/ledger changes. CPU preflight
golden_coverage syntax issue was fixed before submission. Deepcoverage first
queue attempt used systempython withoutscipy andfailedbeforequeue; queued
properly044later. No duplicated GPU work. Observerchargesallreservedtime.
Next: read043 andgoldenutilities; reporttables; analyze044; pursuefurther
gains, notfinalize. Beforefinalclaimfreshdisjointgames/frameanchor and
winnerbatch-orderaudit required. Currentgoalnotcomplete.

## Historical state: 2026-09-19 ~07:55 UTC

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

039-deep-selection DONE683.32s, analysis7.91s. Own160batch1000 snapshot:
CE1.457260/1.358568 at973.87nodes;4000:1.454943/1.361713 at3870.72nodes.
FitCV picks4000 both, but confirmationexpertworse. No clear cost/quality win;
notpromotedgolden. Own1000snapshot keepsbatchnumerics matched. Rootlogitdrift
vsold512batchmax.25,meanabs.000422 reported. All64/256/1000/4000scoresretained.


038-soft-selection DONE; soft_cp25 and soft_cp5 output in aug-soft-selection-v1.
FitCV pickssoft_cp5 for both; confirmation1.456085/1.358828 at976.63nodes.
OrdinaryzeroFPUcp5 was1.456614/1.357294; difference tiny/mixed and FPUconfounded.
At matchedbootstrapFPUcp2.5, newsoft CE1.459466/1.365697 is WORSE than ordinary
bootstrapcp2.5 CE1.456975/1.359714. No established benefit; do not promote.
Incremental softbackup exact CPUreference and livebaselineidentity passed.
First unit-test failure was reference-test mass double count, fixed and rerun;
keep both logs. Current code committedf7eeae0.

039b-child-features DONE10.46s, CPUanalysis4.87s. FitCV selected unchanged
baseline bothmetrics. Opponenttime1.459508/1.359330;time+entropy1.462279/1.361811
vsbaseline1.457688/1.359266. No win; notpromoted. All7arms retained.
Former description: Collect root and all
legalchild time/WDL/policy heads from frozenmodel; test predictedopponent
thinking time and replyentropy as actionfeatures. No futureobservedtime/reply.
CPUuniform-head/mate/sign/illegalmass checks passed (child-features-unit.json).
Outputsaug-child-features-v1. Aftercompletion runCPU
python -m search.engine.analyze_child_features (OMP_NUM_THREADS=1,localruntime).
All7arms, honestgameCV, sameexistingcp2.5 root/Q caches, extraNNqueries charged.
Cheaponeply controls are also reported. No goldenpromotion/CM yet.

040-coverage is RUNNING on SAMEGPU:
coverage.cpp/coverage_native.py/coverage_pilot.py. Root quotas maximize
prior**eta/(1+n), weights sqrt(p),p,sqrt(p*(1-p)), with ordinaryPUCTcp2.5 zeroFPU belowroot. Existing
nodecounter and outputbackups unchanged; no terminal skipping. CPUunit tests PASS: disabled-mode exactbaseline,FPU sign, independentrootquota
choice at every64step for all3modes. Logs/coverage-test-v2.log. Queue040 exists:
DO NOT duplicate. Expected~7min after039. Analyzer is standard
analyze_selection aug-coverage-v1. All3arms share samecp2.5/zeroFPU interior.


New history-transposition idea: transpose_history.py generates up to4 legal
3/4ply commuting-window variants withidenticalendpointFEN, followedbyunchanged
pawn/capture move beforecurrentroot. Entirealteredsuffix legality and automatic
terminationchecked. No changes afterroot; generator seesprefixonly. r2hasnoclock
input; B2future wouldneed consistent clockreconstruction (Claude warned).
Unit test passed; CPUgenerationunderway/completed, seeaug-transpositions-v1/
variants.json andlogs/transposition-build.log. NOT yetGPUqueued. Must report
originalvsindividualvariants, coveragebycell andfixedweightoriginalvsvariantmean,
includingallfallbacks. Sameabsoluteposition0..len-1foreachSGLangsequence. Model
canhavehistorydependent humanpreferences; don'tclaiminvarianceofhumanchoices.

Posthoccoverage-diagnostics.json:cp1.25rootpriorcoverage97.86%,cp5 99.30%;
expert97.08%->99.09%;meancreatednodedepth5.46->5.35. Descriptiveassociation,
notcausalproof. Motivates explicitrootquota040.

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
