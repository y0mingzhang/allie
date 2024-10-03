#!/bin/bash

#SBATCH --job-name="long-tunnel"
#SBATCH --partition="long"
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100GB
#SBATCH --signal=B:SIGUSR1@600
#SBATCH --mail-user=yimingz3@cs.cmu.edu

function sig_handler_USR1()
{
    pkill -f "lichess-bot" -2
    echo "   Signal trapped -  `date`"
    echo "   Requeueing job id" $SLURM_JOB_ID
    scontrol requeue $SLURM_JOB_ID
}

trap 'sig_handler_USR1' SIGUSR1
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

mkdir -p server-logs
redis-server redis-dump/redis.conf >server-logs/redis.out 2>server-logs/redis.err &

for system in $(python src/server/utils.py)
do
    python -u src/server/main.py $system >server-logs/$system.out 2>server-logs/$system.err &
done

sleep 120
cd lichess-bot/
python -u lichess-bot.py >../server-logs/bot.out 2>../server-logs/bot.err &

echo "all launched"
wait
