#!/bin/bash
#SBATCH --job-name=allie-codex
#SBATCH --account=dippolit
#SBATCH --partition=dei-group
#SBATCH --qos=dei_group_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --chdir=/home/yimingz3/src/allie
#SBATCH --output=/data/group_data/dei-group/yimingz3/allie/controller/slurm-%j.log

set -Eeuo pipefail
umask 077
codex=/home/yimingz3/.local/bin/codex
claude=/home/yimingz3/.local/bin/claude
project=/home/yimingz3/src/allie
state=/data/group_data/dei-group/yimingz3/allie/controller
socket="allie-codex-${SLURM_JOB_ID:?Run this script with sbatch}"
thread=01a089c1-c24b-7940-84b8-0af47d9c2be4
session=98a183f3-e82d-433b-8159-e6980d399a19
mkdir -p "$state"
printf '%s\n' "$SLURM_JOB_ID" > "$state/job.id"
hostname > "$state/host"
prompt=$(cat "$project/scripts/codex-kickstart.txt")

cleanup() { tmux -L "$socket" kill-server 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 0' TERM INT
requeue() {
    trap '' USR1
    echo "$(date -Is) Requeueing $SLURM_JOB_ID before timeout."
    cleanup
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap requeue USR1

echo "$(date -Is) job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} host=$(hostname)"
echo "Attach here: tmux -L $socket attach -t codex (or -t claude)"
while true; do
    # Enable RC at boot and recover it if the daemon or connection fails.
    if timeout 45 "$codex" remote-control start --json > "$state/rc-status.log" 2>&1; then
        if ! tmux -L "$socket" has-session -t codex 2>/dev/null; then
            # RC runs in the background; launch the local TUI with full access.
            tmux -L "$socket" -f /dev/null new-session -d -s codex -x 160 -y 48 \
                "$codex" resume --dangerously-bypass-approvals-and-sandbox \
                --cd "$project" "$thread" "$prompt"
        fi
    else
        echo "$(date -Is) RC unavailable; retrying in 60 seconds. See $state/rc-status.log"
    fi
    if ! tmux -L "$socket" has-session -t claude 2>/dev/null; then
        # Read at each launch so prompt edits need no resubmission.
        kick=()
        [[ -s $project/scripts/claude-kickstart.txt ]] &&
            kick=("$(cat "$project/scripts/claude-kickstart.txt")")
        tmux -L "$socket" -f /dev/null new-session -d -s claude -x 160 -y 48 -c "$project" \
            -e PHONE_A_FRIEND_THREAD="$thread" \
            "$claude" --resume "$session" --dangerously-skip-permissions \
            --dangerously-load-development-channels server:phone-a-friend \
            --remote-control allie "${kick[@]}"
        # Development channels always ask for confirmation.
        for _ in {1..30}; do
            tmux -L "$socket" capture-pane -p -t claude 2>/dev/null |
                grep -q 'I am using this for local development' &&
                { tmux -L "$socket" send-keys -t claude Enter; break; }
            sleep 1
        done
    fi
    # Background wait lets the shell handle Slurm signals immediately.
    sleep 60 & wait $! || true
done
