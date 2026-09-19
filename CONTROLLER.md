# Controller

The running CPU-only Slurm controller (current job ID in `controller/job.id`) hosts this Codex session and the user's tunnel. Never cancel it during project maintenance.

- Entry point: `scripts/codex-controller.sbatch`.
- Resume prompt: `scripts/codex-kickstart.txt`.
- Durable state: `/data/group_data/dei-group/yimingz3/allie/controller`.
- The script starts background remote control, then resumes Codex in tmux with permissions bypass enabled. Its timeout trap requeues the same controller job.
- Training and evaluation run as independent Slurm jobs. Controller exit must not cancel them.
- `controller/STOP` means stop autonomous controller work, not cancel independent training.

Read the latest user messages, GOAL.md and BIG_RUN.md after boot. A controller restart does not reopen a completed or blocked goal, replenish compute, or authorize duplicate submissions. The current cleanup request does not resume training.

Before any future submission, check live Slurm jobs, durable checkpoints, the corpus cache and the cumulative ledger. Normal QoS allows 8 GPUs. Total allowance is 8 L40S plus up to 16 A6000 when other DEI demand permits. Preserve every previous charge.

All tiny-scaling, size-validation and seed observers have completed and exited. Their receipts remain under the corresponding durable result packages; do not restart them from old status files. There is no active training watcher to recover.

`scripts/phone-a-friend.py` links this Codex thread and the controller's Claude session live. Each side has a `phone` tool that delivers into the other's session at once; incoming messages arrive as `[phone-a-friend]` turns here and as channel events in Claude. Claude is loaded with a development channel (`PHONE_A_FRIEND_THREAD`, auto-confirmed by the sbatch loop); Codex loads it from `.codex/config.toml`. The peer is not the user.

Remote-control state and credentials stay outside Git. Inspect `controller/{host,job.id,tmux.socket,rc-status.log}` when needed; do not infer a live connection from an old log.
