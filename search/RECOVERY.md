## Update: 2026-09-19 11:12 UTC

071 is DONE,95.51s for4 value-conditioning variants ×16384positions,
1,930,336 extra NNchildren. Analysiscompleted: noresolvedgain. Plus400
(CVmacroselected) confirm1.445677/1.316248, equal1800+actual(CVexpertselected)
1.445350/1.314913 vsparent1.445115/1.315885; all pairedCIs cross0.
No deep counterfactualrating search justified. Bothrootprefixcalls andchild
calls recorded. Actual model-forward16.02s, summedqueryblocks20.56s; most
remainingtime is CPUoneplyboardlayout/I/O outsideblocktimers. Optimization
canwaitunless thisprobe becomesrelevant; it didnotwin.

CPU aug-depth-mixture-v1 DONE: current improved policies at128/256/512
mixedwith1000 usingstatic/Elo/state+clock sigmoidgates. Bothfit-CV metrics
select unchangedparent. Gateweightsnearlyzero, no newgoldenmethod.
This wasdistinctfromcostrouting (allmixturespay1000nodes), and a retry
ofoldfailedlatentbudgethypothesisusinglargerdata/currentbesttrees.
Allscoredarmsreported.

GPUqueueallrequests through071done;engineidlebutresident. Sameallocation
10498670/engine1713252 ends12:52UTC. No new submissions/sharingchanges.
Next hypothesis NOT implementedyet: uncertainty-dependent internal value
geometry (temperature scaled by1-bootstrap^2, or transform critic to logodds
before Bellman backup). Root-only nonlinear utilities alreadyfailed; any new
testmustisolateINTERNALchange, useexactcontrolandindependentreference.
Could inspectcriticvaluehistogrambeforechoosing. Goalunchanged10x/10x.
---
## Update: 2026-09-19 11:06 UTC

070 is DONE. Live state0.1: CE1.445629/1.340769, CM2.98444x/5.01942x,
500.524 mean nodes,59.01s/8192positions. Elo0.01:1.447947/1.338970,
2.87387x/5.15814x,462.855nodes,52.60s. Uniform512:1.451049/1.356525,
2.73399x/3.98781x,497.973nodes,53.58s. Analysis1.07s. Both routers'
paired CIs vsuniform512 favor them onbothmetrics. Frozen plan/results in
golden-value-of-compute-v1, receipt070. Updated123-entry registry/frontier.
LATEST.md is the concise current artifact. Goal10x/10x remains unmet.

CPU probes completed:
- aug-value-gap-v1: smooth per-action Q-gap correction, no resolved incremental
  win (expertbest -0.00097 butCI[-.00393,+.00252]). Not promotedgolden.
- aug-reverse-bellman-v1: reverse-KL internal Bellman regularization.
  Independent constrainedsolver/bounds/rootfinding/tinypriors/negamax testsPASS.
  Forwardcontrol reproducedexactly. Reverse0.1/0.2/0.4 allworse; visited-only
  reverse0.2 expertCE1.31365vs1.31588 butCIwide[-.01431,+.00818].
  MacroCVselectsforward, expertCVvisited-only. No goldenpromotion.
  Claude verifiedvalueformula; unseenmassdiagnostics saved.45.2sCPU.

071-critic-conditions is RUNNING/finishing on sameGPU; do not duplicate.
Producercritic_conditions.py, resultsaug-critic-conditions-v1. 16384August
positions ×4 hypotheticalrating contexts(actual/equal1800/equal2400/both+400).
Alllegalcandidatechild WDL; actualhumanpolicy staysunchanged, roots+childNN
costscharged. Terminalsign/root-moverWDL correct; hypothesesuseONLYprefix.
Afterworker.json appears, runCPU analyze_critic_conditions.py (written, syntax
checked, not yet launched) in node-localruntime OMP_NUM_THREADS=1.
This tests counterfactual VALUE queries, distinct from prior failedroot-logit
counterfactualguidance. Ifnoresolvedgain, noexpensivedeepconditionedsearch.

Current code through routercommitted ea810ce. Reverse/valuegap/criticprobe
and latestdocs changes still needcommit. Allprivate, main untouched.
Latestnodebabel-t5-32, job10498670 ends12:52UTC; sameengine1713252. No new
Slurmjobs, additionalGPUs, sharedledgermutations. Observerretainsallcharges.
---
## Current state: 2026-09-19 10:58 UTC

Own inference goal remains ACTIVE, 10x macro AND expert CM, not achieved.
Latest frozen golden output is golden-temperature-stack-v1:
CE1.4445987842/1.3388551263, CM3.03536x/5.16714x,969.8158 nodes.
Increment beyond the conditional-mixture parent is uncertain (paired CIs cross0).
All reports/registry updated through this result (117 method/control entries).
Current checkpoint/laws/split unchanged; main read-only. No outstanding peer review.

GPU queue070-value-of-compute JUST QUEUED, same L40S allocation10498670,
babel-t5-32, engine1713252, cache1.5Mtokens; ends12:52UTC. Do not duplicate.
Producer search.engine.golden_value_of_compute; frozen plan
golden-value-of-compute-v1/plan.json. Compares uniform512, August CV macro
winner state0.1 and expert winner elo0.01. New growforest module continues
same tree/KV from128 to chosen budget; deterministic-oracle tests match
one-shot trees/values/counts exactly. No repeated NN calls. Root clocks
aligned at target column-1, hash verified. Final actual live result authoritative.
Every arm scored on same reused golden; fresh confirmation still reserved.

New CPU research:
- aug-budget-surface-v2:128/256/512/1000 simulations →125/250/499/972 nodes.
  Selected policy (conditional strength + residual temperature) at each budget,
  calibration re-fit within same3 game folds. August confirm:
  1.46523/1.35627;1.45548/1.33943;1.45025/1.32894;1.44511/1.31588.
- Numerical audit: legacy adaptive reducer retained tiny missing prior mass
  after full expansion. Corrected diff reducer zeroes that tail. Largest
  Q drift2.45e-5 on one action; refitted CE drift<4e-11. Failedv1 source,
  log and audit preserved, including NFS stale-source v2 first attempt.
- aug-value-of-compute-v2: ridge router predicts paired CE differences from
  Elo/format, root entropy/time/clock and128-Q spread/policy KL only.
  Regularization chosen on fit-gameCV; lambda budget price fitted on fit
  inputs for mean<=512 nominal simulations. Macro winner state0.1 confirms
  CE1.445912/1.315980 at502.84 nodes; expert winner elo0.01 confirms
  1.448392/1.315885 at463.91. No golden CM until070. State gains over
  uniform512 have both paired CIs below0 in August.
  v1 is INVALID: FP32 zero padding with1e-300 caused NaN KL. Archived source
  and INVALID.json. v2 castsfloat64 and asserts finite matrices; unit regression
  test and known-budget-allocation test pass. No GPU/golden used by invalidv1.

Retrieval069 completed: June63283games/3.80387Mpositions,3.93GBfp16.
Local checksummed mmap cache in /scratch...search-retrieval/c3bae257... .
All vs ANYsharedparticipant excluded, top512 sameplayer share0.113%.
Large retrieval confirmation incremental CIs cross0; NOT promotedgolden.
Whole069 warm invocation7.30s; query1.61s/16384positions. Attributions/storage
notes sentClaude. No active retrieval jobs or reasons to rebuild bank.

Next: inspect070 receipt/log and results, report CE/CM/node table, update
frontier/report/registry, then keep researching towards10x/10x. No final success
claim on reused golden. Need frozen winner fresh-game confirmation and
batch/order invariance check before finalclaim. Archive failures, retain costs.
Own new code only; no shared main/training/data/ledger changes.
---
## Latest continuation: expanded stack and larger retrieval (2026-09-19 10:35 UTC)

Goal10x macro AND10x expert remains ACTIVE and NOT achieved. Same oneGPU10498670,
babel-t5-32, general/normal, ends12:52UTC. Resident engine1713252 unchanged.
Both STOP files absent; no Slurm/shared changes. Allocation observer preserves all charges.
Claude round4 data@506e9fb mechanical review DONE; recency screen only pinned-month share,
not all-history exposure emulation. Peer agreed month-exact1e17 check before promotion.

065 expanded August16384trees DONE. Fourfold larger SAME August inventory/game folds.
Expanded backup differentiable fitting now CV-selects joint10scalar; condition214.
expanded_stack.py combines constant/subtree/joint with single/sigma.5/sigma1/learned.
CV selects subtree_sigma10 macro and joint_sigma05 expert. All fold parameters retained.
Joint fold parameters recovered from hashed previous execution log, full fits verified,
all reused control scores reproduce exactly. No neural training or new golden tuning.

066 golden attempt failed only overstrict reducer float64 parity (one value1.105e-12
against2e-13 tolerance) before writing scored blocks. Original executed source archived
under golden-expanded-stack-v1/executed-source, failure receipt and plan retained.
067 golden_expanded_stack_v2 DONE,1e-10 tolerance + measured differences, same frozen fits:
- constant_single CE1.45203346/1.35433497, CM2.691431/4.113747.
- subtree_sigma10 CE1.44865908/1.34580384, CM2.840959/4.655957.
- joint_sigma05 CE1.44868707/1.34565881, CM2.839675/4.665945.
All969.8158nodes; scoring107.27s, analysis7.13s. Old4000 still best absolute quality
1.44739937/1.34116928 at3854.934nodes. Reused golden108registry entries, no final claim.
User got table. REPORT/FRONTIER/registry regenerated. No goal reset or new checkpoint.

Expanded consideration-set rank-only choice FAILED:1.47944/1.37276 vs1.45426/1.33062.
Future-legality idea already failed early blitz pilot; cached diagnostic median illegal
mass.002, p99.088, but human-chosen children no cleaner; fitted coefficient0. Dropped,
no new GPU probe. Claude informed. No material/Stockfish heuristic implemented.

Next/current: larger temporally clean June datastore (separate retrieval attribution).
retrieval_bank_large.py built retrieval-bank-large-v1:63283games/3803870positions,
256K/cell target, actual expert-classical availability smaller. Same old exclusions:
golden/dev/test/Aug games by site and full/clipped moves; test moves only exclusion hashes.
068 retrieval_features_large DONE:3.93GB fp16keys,5.09Mprefilltokens,4.70sforward,
8.95s summed block work,159.22s total incl loading/hashing/storage.
Durable under own results/search-v1. Free group space checked~3TB.

CPU retrieval_players.py currently running (SSH exec session10133): scans site/white/black
from June bank source shards and exact August inventory prefixes, writes
retrieval-players-large-v1/players.npz +manifest. Names hashed, IDs0 unknown; ablation excludes
ANY shared participant. Log logs/retrieval-players-large.log. No shared store mutation.
CPU local cache builder running (session85843), log retrieval-cache-large.log:
retrieval_cache.py verifies each durable shard once and consolidates node-local mmap
under /scratch/yimingz3/allie/search-retrieval/<report_sha>. Cold/warm timings recorded
in retrieval-features-large-v1/local-cache.json. Prevent repeated NFS metadata/read costs.

NOT yet queued: retrieval_neighbors_large.py, intended069. Wait for player sidecar and
local cache completion, inspect errors. Existing feature bank complete. Collector returns
512same-cell legal neighbors both with/without shared players; kernelsk32/128/512,temp.03/.1.
After worker finishes run python -m search.engine.analyze_retrieval_large (OMP_NUM_THREADS=1).
Analyzer fitsglobal lambda inside3gamefolds; baselinesdirectElotemp and fixedsubtree_sigma10
(using exact stored fold params), scores all16cells and paired participant exclusion.
No golden retrieval until CV/confirmation support a win. Adds data memory; reportbank/build/
query/memory and extra rootprefill, not pure search. Claude no objection with exclusions.

Runtime /scratch/yimingz3/allie/search-runtime/sglang-0.5.9-torch2.9.1-cu128/bin/python.
Pyarrow CPU scripts need PYTHONPATH=/data/group_data/dei-group/yimingz3/allie/envs/chessmix-overlay.
All own operations via SSH babel-t5-32 to avoid controllerNFS. Never kill tmux keeper.

## Latest continuation: mixtures and larger development sample (2026-09-19)

All requests through064 completed; engine idle, no Slurm changes. 10498670 still
running on babel-t5-32 until12:52UTC. Goal10x/10x NOTachieved.
064 golden-player-search-v1: fixed search-strength mixture sigma.5:
CE1.45055442/1.351203637,CM2.755694/4.302838 at969.8158nodes.
sigma1:1.4502528/1.35121634,CM2.769040/4.302048 samecost.
History-adapted.5:1.45041094/1.35103752,CM2.762032/4.313178,1183.8427nodes.
History-adapted1:1.45017454/1.35281716,CM2.772517/4.204040,sameextra cost.
Mostly static-mixture benefit. All4 preregistered arms plusmatchedconstant
reported. Reusedgolden; no corrected significance/finalconfirmationclaim.
Current-root1000trees arecached060; newhistoricalqueries areactual. Totalstandalone
runtime NOTequalhistory-onlyruntime. Charge historical roots/children/prefill
perquery evenifphysicallyshared. AllparametersfitAugustonly.
Results include pairedCIs andCM. Best4000point stillCM2.899512/4.989236.

062 aug-adaptive-deep-v1 complete andanalyzed: adaptive4000 macroregresses,
expertnearlyflat; no promotion. Sameconstant4000 Aug1.446684/1.343797 vs
subtree4000 1.449047/1.343005. Allbudgets256/1000/4000 inresults.json.

063 aug-player-search-v1 complete, inference17.51s for27,412 pastpositions,
819,973 childqueries,4096currentqueries. CVselectedadapt_s1.0_e1.0 macro
andadapt_s0.5_e0.25 expert. Control+staticpriorsalsoenteredgolden064.

CPU aug-diff-backup-v2 complete: differentiationofoutputbackup only; node
selectionneverusesfittedtau. tau=exp(a)*(1+(desc-1)/16)^b.
Independentfloat64torch/autograd+finite-differences+zero-tailtestsPASS.
BothfitCVselectconstantcontrol, despiteconfirmationfixed2scalar
1.449715/1.346246 andjoint10scalar1.449203/1.344298 vscontrol1.452666/1.357460.
JointHessiancondition461 inoptimizercoordinates. No goldenpromotion.
v1 hadlatent0*exp(overflow)zero-unvisited-mass edgecase; originalsourcespreserved
inaug-diff-backup-v1/executed-source. Fixedguard andfullyexpandedmass exactly0;
v2 fullrerun changesCE<3e-12. Failededitscript alsoaccidentallyreranv1 CPU,
recordedSUPERSEDED.json/logs; no GPUsubmission/reset.
Sourcefilesdiff_backup.cpp/native.py/analyze_diff_backup.py nowv2.

Next: expandAugust sample4x (512positions/cell/fold=16384), SAME existing
aug-tune-v1 inventory andsamegamefolds, nested original128+384uniformremaining.
No newcorpus/goldendata. Fit/CV rankingsunstable at2048fitmoves; improveprecision
beforemoregolden. Not yetbuilt/queued atthisnote. Use newoutputdir andhashes.
Then collect1000-sim compacttrees withmatchedconstant/subtree snapshots at256/1000,
andcomparestaticmixtures/calibratedbackup onexpandedfit-CV/confirmation.
No needrepeat expensivehistoryadaptation yet; tinygain+22%nodecost.

All newresultsareinREPORT/FRONTIER/GOLDEN_METHODS afterregeneration.
No outstanding peerreview. Claude agrees differentiatedbackups sound, requested
identifiabilitycheck andindependentautograd; bothprovided. Clarifiedtopeer
that output-onlytau doesn't affecttreeexpansion.

## Current continuation: adaptive backup and within-game adaptation (2026-09-19 ~10:00 UTC)

Goal remains active: fixed-checkpoint 10x macro AND10x expert CM plus node frontier.
Not achieved; do not reset UI goal or main files. Own Slurm10498670 on babel-t5-32
runs through12:52UTC. Engine PID1713252 (1.5M cache), socketsearch-v1-10498670.
Both STOP files absent. No new allocation or shared mutation. Use SSH node for I/O.

060 golden-adaptive-temperature-v1 finished: constant vs subtree on identical
actual1000 trees,969.8158 NNnodes/position. Constant macro1.4529637225/expert1.3565202547,
CM2.651997/3.988074. Subtree1.4520683107/1.3530486294,CM2.689940/4.190107.
Expert deltaCI[-.006803,-.000359],macro[-.002036,+.000235]. Treat as candidate:
many golden tests, no multiple-testing-corrected win or fresh confirmation yet.
Best expensive point remains static4000 CM2.899512/4.989236,3854.934nodes.

CPU aug-adaptive-temperature-v1: both fitCV choose subtree tau=.2sqrt(16/(16+desc-1));
August1.450230/1.347842 vsconstant.1 1.452666/1.357460. Count is exploration
proxy, not independent samples. Other depth/action-scale/constant temps reported.
CPU aug-moment-tail-v1: all moment-constrained unvisited fallback arms fail; no promotion.

061 aug-history-residual-v1 finished inference8.26s and CPUanalysis:
same-player past-move onehot-minus-own-model residual, no weightupdates.
Searchcontrol1.4527775/1.3583490; last8hidden1.4515642/1.3584818.
MacroCVselected unchangedzero-strength,expertCVlast8; no expert confirmation gain.
No golden promotion. All variants inresults.json; user received table.

062-adaptive-deep is currently running on residentengine:
module search.engine.adaptive_deep, outputaug-adaptive-deep-v1.
4000sim,320rootbatch,2threads, forcedskip. Save compacttrees plus constant/subtree
backup at256/1000/4000 logicalprefix budgets. Afterworker done run:
  OMP_NUM_THREADS=1 <runtime>/python -m search.engine.analyze_adaptive_deep
It fits separate Elo alpha/beta inside fitgameCV and reports all6arms, pairedCIs.
No golden promotion until results/freeze.

063-player-search is queued after062:
module search.engine.player_search, outputaug-player-search-v1.
For each Augustquery evaluate oneply alllegalactions at up to8previous OWN moves,
strictprefix only. Heads preservecondition; exactterminalvalues. Pastprefix+pasttarget
is key (sameprefix canhave different actual historicalchoices).
Codehistory tests passed; no querytarget supplied to histories().
Afterworker done run:
  OMP_NUM_THREADS=1 <runtime>/python -m search.engine.analyze_player_search
CPUanalyzer written and math tests passed. Bayesianlatent searchstrength factors
[.25,.5,1,2,4], priorlogwidth .25/.5/1, evidencepower .25/1.
Fitpast oneply alpha/beta on fitqueryhistories only, refit insidegameCV; currentroot
1000q fromexistingstaticcache. No currenttarget entersposterior. Compare cheaper
staticpriormixtures; chargeALL historical roots/children/prefill perquery even if
shared physically. Reportcostinflation. No golden until fittedselection.

Reports/frontier source updated toinclude060+CPUadaptive/moment/history; regenerate.
Uncommitted newfiles adaptive_temperature*, analyze_adaptive_temperature,
moment_tail*,analyze_moment_tail,golden_adaptive_temperature,history_residual,
analyze_history_residual,adaptive_deep,analyze_adaptive_deep,player_search,
analyze_player_search. Commit afterreports/resultsreview.
Lastcommit8a36d0a. No peerreviewpending. Claude suggestedactualclock (failedAug)
andlikelihood of searched vsraw onpastmoves (063). Peer warnedmultipletesting;
agreedcandidate-only. Modelweights/testsplitunscored unchanged.


# Search worktree recovery

Codex owns this worktree (`codex/search-v1`), GOAL.md, search/ and private
results/search-v1/. Claude owns main and training/data/model research. Read
GOAL.md and /data/group_data/dei-group/yimingz3/allie/controller/STOP first.
Never submit training, stop the tunnel/controller, or change Claude's jobs.
The UI goal is stale/paused; do not reset it. Our authorized goal is in GOAL.md.

## Current continuation: completed rollout and cheap calibration screens

059 human_rollout DONE, no GPU request after it. OneGPU10498670 unchanged,
engine1713252 remains ready (idle time charged). Do not duplicate or reset.
4096Augustroots,4 trajectories/legal root move; preserved Elo/TC and independent
counter RNG, exact terminals/forced skip, full cache handles. CPU independent
legality/categorical RNG/perspective/terminal/variance/batch independence tests
passed. Initial test had a replica-ID lookup bug (shared first node); corrected
test uses root/action/replica. Implementation unchanged. Logs retained.
Scoring95.59s, total128.44s,7.125Mnonroot NNnodes.
Elo-calibrated depth1 CE1.517948/1.492103(29nodes),
depth4 1.488556/1.430232(379), depth8 1.497611/1.451025(840),
depth16 1.515411/1.486457(1739); static1.452777/1.358349(971).
BothfitCV choose static. Globalcalibration also reported, no promotion.
Rollout+search and noise-shrunk rollout stacks also bothCV choose static.
Noise diagnostics in aug-rollout-v1/stack.json. High horizon variance supports
only a negative result for four samples, not a claim all rollout budgets fail.

All private CPU screens done:
- aug-retrieval-residual-v2: bothCV select search_log_ratio_k128 but confirmation
1.453222/1.357116 vs1.452777/1.358349; bothpairedCIs overlap0. No promotion.
- aug-boardbook-v1: macroCV selects count smoothing,1.452462/1.358064;
expertCV unchanged; CIs overlap0. Exact-board bank covers~9%queries. No promotion.
- aug-clock-calibration-v1: ONLYbefore-move feat[row,target_column-1,0], hashes
verified and target/prefix/label alignment asserted. Clock-only alpha and
search alpha/beta variants. Selected trouble_both and seconds_format_beta
REGRESS onconfirmation; no golden. Adds legal info to r2noclock, explicitly
separate from algorithm-only gain. First result write failed local Path variable
shadowing; corrected, plan-first/executed-first preserved.
- aug-outcome-consistency-v1: 1/4/16 Bayesian projections of child WDL to root
WDL, scalarfit strength. MacroCV unchanged; expertCV iter4 worsensconfirmation.
No future outcomes supplied. Small changes only, no golden.
All usertables sent except outcome-consistency (send next).

Failed retrieval first-producer code moved out of search/ to private
results/search-v1/retrieval-failed-sources; original bank builder reconstructed
and verified against original manifestsha. Correct producers remain*_v2.py;
do not try rerunning failed queue modules: their error receipts prevent repeats.
Current reports include new screens; rerunreport after any addedresults.

Next hypothesis to consider: adaptive Bellman temperature based on actual
subtree expansion (uncertain shallow nodes closer to policy expectation,
well-expanded nodes more decisive), using cached coverage trees for cheap
CPU selection. Not implemented or preregistered yet. Could instead pursue
another justified model-only method. Currentgoal10x/10x stillnotachieved.
No peer review outstanding, no sharedfiles or Slurm mutation.

## Previous continuation: retrieval pilot and faster golden complete

One unchanged GPU10498670/babel-t5-32, engine1713252, cache1.5M. No extra GPU.
050 DONE: actual4000 gives golden CE1.44739937/1.34116928,
CM2.89951/4.98924,3854.934mean nodes,555.79s scoring+.816s analysis.
Own logical1000 control CE1.45296415/1.35638745 (CM2.65198/3.99557).
Paired4000-1000 macroCI[-.007688,-.003369],expert[-.022537,-.008573].
Higher cost alone not dominance; reused golden, not final fresh confirmation.
Reports/frontier/registry updated (79 entries including duplicate controls).

051 DONE: adaptive root curvature, own static control, all1000 sims.
August static CE1.4527775/1.3583490; half-adaptive1.4526109/1.3579575;
fullyadaptive1.4501743/1.3504992. BothfitCV choose half-adaptive; pairedCIs
include0, so no golden promotion. Full arm not selected from confirmation.

052 feature capture PASS: root logits identical normal/LAST/FULL, last states
match, arbitrary changed future does not change preceding captured features.
June-only retrieval bank:503281 targets/9057games after fixing game-end mask.
Original retrieval-bank-v1 incorrectly labelled6920 terminal2346/2347 tokens;
053 feature extraction STOPPED on target-range assertion before any result.
Corrected retrieval-bank-v2 only masks those targets, keeps games/tokens unchanged,
parent manifest hash retained. Original failed queue/artifacts retained.
Builder now correct and points v3 to prevent overwrite; do not rebuild v3 blindly.
055 retrieval-features-v2 DONE:733780inputtokens,~520MB,3.11s invocation.
056 aug-retrieval-v1 neighbors DONE:4096roots,.286s prefill+.158s query,
1.97s whole invocation. Same-cell, legal-move-filtered cosine neighbors.
All9kernels×2baselines on August: direct retrieval fails; macroCV selects
search+k32/temp.1 CE1.4509585/1.3556495 vssearch1.4527775/1.3583490,
but pairedmacroCI[-.006415,+.001367]/expert[-.012944,+.002023] overlap0.
ExpertCV picks unchangedsearch. No golden promotion.

057 residual-neighbor GPU cache failed missing no_grad, before cache output.
058 retrieval_residual_v2 fixes it, writes aug-retrieval-residual-v2.
CPU retrieval_residual_analysis currently running (log retrieval-residual-analysis-v2.log).
Tests residual zero-strength/zero-effect/support/fallback passed.
Residual compares empirical neighbor actions against frozen-head expected moves,
linear additive or smoothed log ratio; kernels32/128,temp.1, globalfit strength.
No model weight updates, no query target enters GPU neighbor task.
CPU boardbook.py running (log boardbook.log), exact board/cell empirical counts
from the SAMEJune bank, constantmix/Dirichlet smoothing, matched static/direct
controls. Output aug-boardbook-v1. No new data/GPU and no golden yet.
Read results before any resubmission; service caches modules, use fresh names.
Failed sources/requests stay as evidence, not quality results.

Bank safeguards: preJuly Juneonly; dev/test/golden/August game IDs and full/clipped
move hashes excluded (test moves read ONLY for exclusion fingerprints, unscored).
All retrieval is additional training-corpus memory, separately labelled from
pure search CM, with build/query/memory cost. Claude informed. No sharedchanges.
After CPU results, show tables, updateREPORT/RECOVERY, commitprivatecode.
Goal10x/10x still NOT achieved. No outstanding peer review.

## Previous continuation: faster stack live; deeper golden running

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
