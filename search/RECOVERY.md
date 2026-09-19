# Search recovery — 2026-09-19 17:51 UTC

## Authority and goal
Latest user constraint: NO EXTERNAL MEMORY. No human-game retrieval/books/kNN,
cross-game player profiles or episodic datastore. Fixed model + current game
context + rules + query-local tree/KV only. Historical reports stay for audit;
no retrieval work running or to restart. Carefully ablate ideas, prioritize
promising directions. Next equal-budget tactical-vs-normal search check.
Yiming explicitly resumed this goal after the usage reset:10x macro AND10x expert
training-equivalent CM, fixed checkpoint/laws, improved NN-node/quality frontier.
Do not reset/complete the goal. Goal UI returnednull at18:02UTC; restoredthe
sameexplicitlyresumedgoal withcreate_goal,no tokenbudget or changedtargets.
GOAL.md in thisworktree recordsstatus. MainGOAL isanothertrack.
Read both controller/STOP and results/search-v1/STOP before work. Neither existed
at last check. Never cancel tunnel/controller/Claude jobs or mutate main.

## Running
ONE preempt GPU10503413, RTX6000Ada on babel-x9-24,4CPUs/48G.
Backfill ends18:03:51UTC (54min, not requested8h). General reserved for Claude.
Engine3807432, tmux socket search-v1-10503413/sessionoracle. Ready/requests/results
under results/search-v1/engine-queue. Runtime on GPU node:
/scratch/yimingz3/allie/search-runtime/sglang-0.5.9-torch2.9.1-cu128/bin/python.
Use this runtime for scipy; controller torch210 runtime lacks scipy.
Observer search.advance --watch accounts all reserved/idle GPU hours. Previous
cumulative10.7547222222GPUh preserved. Before any replacement inspect jobs/ledger/
checkpoints/data/STOP and phone Claude. Keep only one GPU.

083-golden-time-allocation-v2 DONE:3 preregistered live arms fixed512,
predicted_time,shuffled_time on current tree/output recipe. No neural training.
Common root-time prepass separately charged to each arm. All3 results required;
no selecting gold winner. Queued module golden_time_allocation_v2.py.
082 critic-transpositions DONE: averaging same-board legal history WDL predictions
failed August CV (selected zero correction). No golden promotion.
081 activation-ridge-v2 DONE; CVmacro selects ridge.1 but expert selects plain.
080 failed CPU test before inference, archivedfailedsources;do not rerunoldmodule.
079 golden activation DONE:step1 CE1.44387179/1.33839146 CM3.07195/5.20376;
step16 CE1.44379446/1.33803906 CM3.07588/5.23182;oldcosinecontrol CE1.44375333/
1.33801255 CM3.07797/5.23394. Expert gain CIs cross0;all3reported,nogoldselection.

## Current hypothesis
aug-time-allocation-v2 DONE CPU1.99s. On modern cached tree+policy, predicted time
normalized by16-cell August mean wins gameCV for both metrics. Confirmation CE
1.44914054/1.32620857 vs fixed5121.45025256/1.32893835 and shuffled1.45393713/
1.33735081. Gain vs shuffle significant; gain vs fixed uncertain. Cached nodes
503.75/499.69/501.28;nominal time/shuffle histograms equal. CM pendinggolden083.
This tests Allie-like time allocation; NOT faithful released traversal or reverseKL.
Revisit reverseKL output independently after this; quiescence idea not implemented.

## Fixed inputs and reporting
Checkpoint durable main results/pretrain/r2-3e16-control-t20-w20-pf052h-s42/
checkpoints/step-00002274-9a5e6eb1/model.pt (SHA3739d0d90f2ea826874ddb29a0ec8f99e28a0ef843ab8f0e1e6d9f4f7f9b5cc5).
Frozen source results/recipe10x/data-v1-round2/source-ours.8x512,no clock inputs.
Golden strat-eval-v1 unchanged;8192sample=512/cell,6752games in golden-balanced-v1.
Full raw anchor1.524007755681445/1.476428895813994;10x targets1.3832044316765544/
1.2992929937638005. Laws training-cm-laws.json immutable. Final success requires
fresh disjoint-game confirmation;test/test_expert unopened. Current golden reused.
August expanded16384 tuning/confirmationgame disjoint,possiblytraining-seen,
userapproved. Never usegoldenlaw toconvertAugustCE. Neverfuturehumanmoveinputs.
After each completedexperiment showallarmsCE/CM(cost); regenerate search.report,
search.frontier,search.golden_registry. No images unlessasked. Tables/artifacts
LATEST.md,REPORT.md,FRONTIER.md,GOLDEN_METHODS.md underresults/search-v1.

## Parallel work and peers
User requested Astra explainer agent search_explainer. Scope corrected by user:
main algorithmFAMILIES inhuman-chessprediction,notonlyAlliecomparison. Agent owns
results/search-v1/explainer/ only. Must reportartifactwhenready;do not duplicate.
ClaudeMoE31249f5 reviewDONE:independentCPU8path/90MACchecks passed. Numericaledge
sqrt(softplus(-110)) givesNaNgrad;eval384expert densepadding64xactivecost flagged.
Reviewreportreviews/moe-31249f5.md;peerfixesfollowups mayarrive. Allolderreviewsclosed.

Earliernotes archivedresults/search-v1/recovery-history/before-20260919-1751.md.
Do not read wholearchive unlessneeded;itcontains staleallocations/pausedstates.

Update17:55UTC:084 reverse_surface DONE CPU50.6s. Reverse-KL modern output losesall4budgets.083 livegolden time beatswithin-cellshuffle,notresolvedvsfixed; newtablesLATEST/REPORT. No quiescence implemented. Replacement10503933 queuedpreempt afterany10503413,8h/max,min30min,oneGPU. Previousobserverhadexited; sacctreconciliation restoredcurrentcharges11.496944GPUh,setsidobserverrestarted logobserver-10503933.log. Checkitislive. Agentfinishedexplainer/index.html,validatedHTML/JS/links,notbrowserrender. UsercorrectedAugustnaming: reuseddevelopmentcheck,notnew/freshconfirmation.

Update17:59UTC:085-tactical queued/running after independentCPUtestPASSED.
Code tactical.cpp/tactical_native.py/tactical_pilot.py. Rootalllegal; thenonly
captures/promotions(top2neuralprior) orallevasionsincheck,maximum256nonterminal
NNnodes and6plies/root. Mean/soft.2/soft.05/maxbackup value-minusoneply residual
onfrozenparent,global/Elo scalarfit. All arms reportedAugustonly first. Source
frozenplan andatomicper256rootchunks; incompleteblocksrecompute afterpreemption.
No NNweightchanges/externalengine. Inspect085result/error beforeanyretry.

18:02UTC:085DONE92.5scollection+12.0sanalysis. Meanextra200.68 NN/rootinclprefill.
AugustCVmacroselectstactical_.05:CE1.44443944/1.31318394 vsparent1.44511482/
1.31588462;expertpairedCI[-.004728,-.000692],macroCIcross0. CVexpertselects
max_elo:1.44394055/1.31245455,itsincrementalCIs cross0. No goldenpromoted:
additional20%nodes doesn't establishfrontier dominance. Next CPUcheckusecached
quiescencefeatures withlowerbasebudgets, so tacticalworkcanreplaceordinarynodes.
Currentjobabouttotimeout;replacementdependencyshouldstartautomatically.
