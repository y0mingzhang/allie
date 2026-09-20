USER PARKED2026-09-19~21:15EDT. Do not execute the continuation instructions below without new user direction. Inference stops here; next work is MoE kernels.

# Recovery

Owner: Codex, branch `codex/time-router-v1`, durable worktree
`/data/group_data/dei-group/yimingz3/allie/worktrees/time-router-v1`.

Read its GOAL and the latest user instructions first. The old search goal remains
paused/unmet. This finite follow-up is separately authorized. Never remove the
old results/search-v1/STOP. This study uses its own time-router-v1/STOP.

Initial GPU: job10506788, one general L40S on babel-n5-24, 2h ceiling. The queue
contains four August grids and the frozen coverage/Allie July grids. Outputs are
immutable and resumable by block. `finish_grid` observes completion, writes the
study STOP to release the GPU, and runs CPU evaluation. Its log is finish-grid.log.
The Allie development fit observer logs allie-fit.log. Frozen selection is
frozen.json, plus the preregistered think-time addendum. Do not re-fit on July.

Next phase: after the first job fully exits and cached-results.json plus
gold-allocations.json exist, remove only the study STOP and submit at most one
replacement general L40S. New service.py recognizes action=live. Queue exactly
nine requests (seven frozen routers plus two matched-per-cell diagnostics),
using gold-allocations.json unchanged. Start finish_live to stop GPU promptly
when all nine complete, then analyze. Job receipts and the cumulative status
accountant preserve all charges. Do not duplicate submissions.

Python CPU: /home/yimingz3/src/allie/.venv/bin/python, OPENBLAS_NUM_THREADS=2.
GPU runtime staged by workbench.sbatch. Full canonical anchors and tests remain
untouched. Final write-up should use live-results.json as authoritative, report
cached/live noise and node drift, and include all declared methods.

Update2026-09-20T01:00Z: general10506788 COMPLETED1288s, released for MoE perf. Replacement10506918 on preempt (1L40S,90min), receipt live-job.json. Its service supports live requests. finish_grid --live (finish-resumed.log) resumes the remaining grid, analyzes, queues nine live arms, stops GPU and scores. Interrupted queue006 STOP error retained as .interrupted-general-release.json; no blocks discarded. Accounting observer restarted with stdlib-only imports.
