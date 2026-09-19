#!/bin/bash
# Retain the allocation for debugging if the oracle exits. No uncontrolled retry loop.
cd /data/group_data/dei-group/yimingz3/allie/worktrees/search-v1
"$SEARCH_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 search/cache.py --serve 2>&1 | tee -a results/search-v1/logs/oracle.log
printf '\nOracle exited. Repair via this tmux session; allocation remains available.\n'
exec bash
