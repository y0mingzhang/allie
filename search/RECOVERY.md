# Search recovery — 2026-09-19 18:15 UTC

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
ONE preempt GPU 10503933, RTX6000Ada on babel-x9-24, 4 CPUs / 48G.
Backfill ends 19:28:51 UTC (84 minutes, not the requested 8 hours).
10503413 timed out. No further replacement has been submitted. General remains
reserved for Claude. New resident engine PID 3859509 is ready for job 10503933;
startup 22.5 s. Requests/results: results/search-v1/engine-queue.
Runtime on GPU node:
/scratch/yimingz3/allie/search-runtime/sglang-0.5.9-torch2.9.1-cu128/bin/python
Observer search.advance --watch reconciles sacct to own status.json, including idle
time and every old charge. Before any replacement inspect jobs, ledger, data,
checkpoints and STOP; phone Claude. Never overlap two owned GPUs.
Tasks through087 are DONE. 088 failed a2.3e-12 calibration audit tolerance;
its source/error are retained. 089 is DONE (symmetric selected by both CV metrics).
090 is DONE (one64-root deep-scaling smoke). 091 is RUNNING, then092 analyzes it. No incomplete retrieval tasks. Do not run any
retrieval or external-memory experiment, even if stale notes propose one.

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

## Current evidence and next decision
083 time allocation: normalized predicted time improves over within-cell shuffled
signals, but not conclusively over fixed512. Existing broader router is better.
084 reverse-KL output with modern mixture/calibration loses across all4 budgets.
085 additive tactical residual gives small expert gains but adds ~201 queries.
086/087 finish the cost-substitution check. Each lower-budget parent is refitted
inside the same game folds. Exact expected-cost controls independently randomize
between adjacent ordinary budgets; they do NOT execute both or ensemble policies.
At 450.87 NN nodes, ordinary expected CE 1.451263/1.330967 vs tactical max/Elo
1.452959/1.332198. All6 tactical variants lose on point estimates; some CIs cross0.
At699.64 nodes the analogous gap is +.000541/+.000807, also uncertain.
Reports: aug-tactical-budget-v1/{REPORT.md,results.json,cost-comparison.json}.
Source modules tactical_budget.py and tactical_cost_compare.py; cached analysis
10.13s + .35s, no new NN queries, no golden eval, CM pending.

Do not promote this additive tactical residual or claim tactical search broadly
fails. Any retry should reuse existing tree evaluations and target unstable
frontier nodes, rather than repeatedly score all root children in a separate tree.
Choose the next strongest hypothesis from actual evidence; avoid another broad
parameter grid solely because cached evaluation is cheap. Current score plateau
is far from10x macro, so marginal calibration wins alone are unlikely sufficient.
Older asymmetric own/opponent backup grid (aug-compact-v1) selected symmetric
.1/.1; it predates current root-coverage/count-decay/output stack. That is a
possible controlled recheck, not a pending task or justification for a large scan.

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
User-requested Astra algorithm-family explainer is DONE and delivered, and the
user praised it. Artifact results/search-v1/explainer/index.html. No extra images
requested. It marks retrieval excluded by the current goal.
ClaudeMoE31249f5 reviewDONE:independentCPU8path/90MACchecks passed. Numericaledge
sqrt(softplus(-110)) givesNaNgrad;eval384expert densepadding64xactivecost flagged.
Reviewreportreviews/moe-31249f5.md;peerfixesfollowups mayarrive. Allolderreviewsclosed.

Earliernotes archivedresults/search-v1/recovery-history/before-20260919-1751.md.
Do not read wholearchive unlessneeded;itcontains staleallocations/pausedstates.

Current worktree goal remains active. No new Slurm submission needed now.

## Live deep-scaling study (18:31 UTC)
091-deep-scale-half calls search.engine.deep_scale, max_blocks32: first2048
positions from a score-independent interleaving of128/cell/fold (4096 total).
092-deep-scale-analyze-half follows automatically and fits each output recipe
inside the original game folds. Do not duplicate requests or submit another GPU.
The first block took44s including serialization; estimate~23min for2048.
Studies and atomic chunks: results/search-v1/aug-deep-scale-v1.
Progress: progress.json; results-prefix2048.json after analysis. Full4096 not
requested yet; extending later uses the same collection module withoutmax_blocks.
Growing trees1000->4000->16000 shareall evaluations. Compare constanttau.1,
currenttau.2/sqrt(1+n/16), and normalizedtau.2/sqrt(1+n/(16*B/1000)). Current and
normalized are exactlyequal at1000; count-scale recursion/identity tests passed.
Purpose: separate more computation from increasingly optimizing continuation
behavior. Ordinary search/root coverage unchanged. No golden access.

Default controller Python (/home/yimingz3/miniconda3/bin/python3.11) DOES have
numpy/scipy/torch; torch210 runtime lacks scipy. Default lacks pybind11/chess.
No environment installation made. A cursory primary-paper search found Gumbel
AlphaZero (https://openreview.net/pdf?id=bERaNdoegnO): its guarantee is playing
policy improvement with correct values, not human likelihood; do not substitute
best-action identification for the distribution-estimation objective blindly.

## Update 18:57 UTC
091/092 completed2048,093ratingprecision completed.094 now extends deep-scale to full4096;095 analyzes it. Do not duplicate. The current allocation10503933 expires19:28:51UTC.
New CPU modules lapse_screen.py, diagnose_fit.py, stability_diagnostic.py, value_uncertainty.py completed. Rating precision module completed on resident GPU (10.6s). Lapse/rating/uncertainty: no resolved development gains and no golden promotion. report.py includes their full tables.
Fit-only diagnostics: calibrated confidence53.7% versus accuracy53.9%; first20plies see only~.0014CE search benefit, versus~.1 later. Early-to-late root value instability Spearman.778, but held-out CE-gain prediction R2 only.0025overall/.0046expert. Do not mistake value instability for human-prediction improvement.
Next: inspect aug-deep-scale-v1/results.json after095. Costs and every arm required; no goldenCM conversion on this August cohort. No external memory or retrieval.

A causal root-quota screen is now queued as096 smoke after095: quota_replay.py/.cpp. It reconstructs independent root-action traces from16K trees, observes Q128/Q256 only, and redistributes the remaining744simulations using action instability. Two sigma rules (plain and sqrt-visit scaled), clipped allocation multipliers. Every birth must match the reconstructed original schedule; baseline1000 Q/nodes must be bit-identical. Cache exhaustion is a hard error. Only run full after smoke passes. Counterfactual results remain conditional on cached NN predictions; any promotion needs actual live search due batch numerics. No newGPUallocation.
