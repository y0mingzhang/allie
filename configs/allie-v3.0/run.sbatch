#!/bin/bash
#SBATCH --job-name=bigfix
#SBATCH --account=dippolit
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:8
#SBATCH --exclude=babel-q9-32,babel-x9-32,babel-t5-28
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-0%1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yimingz3/src/allie/results/recipe10x/bigfix-24x1536d75m4shipv2nf-c8s200f0v4/logs/%x-%A_%a.out
exec /home/yimingz3/src/allie/.venv/bin/python /home/yimingz3/src/allie/results/recipe10x/bigfix-24x1536d75m4shipv2nf-c8s200f0v4/modelexp.py task /home/yimingz3/src/allie/results/recipe10x/bigfix-24x1536d75m4shipv2nf-c8s200f0v4/round.py bigfix-24x1536d75m4shipv2nf
