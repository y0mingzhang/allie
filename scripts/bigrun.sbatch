#!/bin/bash
#SBATCH --job-name=bigrun-v1
#SBATCH --account=dippolit
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --gres=gpu:L40S:8
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=babel-o5-20,babel-o5-24,babel-n5-32,babel-q5-32,babel-x9-32
#SBATCH --output=/home/yimingz3/src/allie/results/recipe10x/bigrun-v1/logs/%x-%j.out

# Usage: sbatch --time=01:00:00 scripts/bigrun.sbatch bench
#        sbatch scripts/bigrun.sbatch train <steps>
exec /home/yimingz3/src/allie/.venv/bin/python /home/yimingz3/src/allie/results/recipe10x/bigrun-v1/bigrun.py task "$@"
