#!/bin/bash
#SBATCH --job-name=r-ext-build
#SBATCH --account=dippolit
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/data/group_data/dei-group/yimingz3/allie/ext-v1/logs/%x-%j.log
# Parse raw files as they land (downloads may still run elsewhere) until all are fetched or
# UNTIL (epoch seconds) passes, then parse the rest, finalize and test. Every step is restartable.
set -uo pipefail
cd /home/yimingz3/src/allie
PY=.venv/bin/python
until $PY scripts/extdata.py complete || [ "$(date +%s)" -ge "${UNTIL:-0}" ]; do
    $PY scripts/extdata.py parse
    sleep 60
done
$PY scripts/extdata.py parse &&
    $PY scripts/extdata.py finalize &&
    $PY /data/group_data/dei-group/yimingz3/allie/ext-v1/tests/test_ext.py
