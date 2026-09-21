#!/bin/bash
#SBATCH --job-name=bigrun-v2
#SBATCH --account=dippolit
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=babel-q9-32,babel-x9-32
#SBATCH --output=/home/yimingz3/src/allie/results/recipe10x/bigrun-v2/%x-%j.out

# One chunk of the run planned in bigrun-v2/NAME. Submit with `bigrun.py submit NAME
# [sbatch options]`, which adds --job-name, --output=NAME/logs and --signal=B:USR1@<margin>
# from NAME/plan.json; the frozen copy NAME/bigrun.py supervises, requeues and scores.
exec /home/yimingz3/src/allie/.venv/bin/python \
    /home/yimingz3/src/allie/results/recipe10x/bigrun-v2/"${1:?usage: bigrun.sbatch NAME}"/bigrun.py task "$1"
