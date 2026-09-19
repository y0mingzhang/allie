# Search research plan

Starting checkpoint: /data/group_data/dei-group/yimingz3/allie/results/pretrain/r2-3e16-control-t20-w20-pf052h-s42/last.pt
Frozen source: /data/group_data/dei-group/yimingz3/allie/results/recipe10x/data-v1-round2/source-ours

## Hypotheses

1. Temperature/legality are cheap controls; separate their effect from added inference.
2. One-ply model WDL can identify policy mistakes. Blend a bounded value correction
   with the policy rather than replacing its human distribution with greedy winning play.
3. Predicted think time, policy entropy and root/child value disagreement may identify
   where extra computation helps. These are predictions from the prefix, not future labels.
4. If shallow values help, test expectation over human-policy replies and deeper adaptive
   expansion. Preserve probability on unexpanded moves; do not score truncated top-k as
   a full distribution.

A coherent exact policy's sampled continuations marginalize back to that same policy.
Sampling more continuations alone is not a new information source. Improvements require
useful auxiliary predictions, constraints, calibration or a demonstrably useful correction.
Value heads trained on human outcomes can have off-policy errors on unusual branches.

## First pilot

Use a deterministic 1,024-position subset of each existing prepared dev/dev_expert
split for the first quick screen, preserving their target and legal-move definitions.
Game-hash split: fit vs development confirmation. These are mostly blitz; development
numbers are not the 16-cell golden metric. Final evaluation must use the exact existing
strat-eval-v1 arrays, all scored moves and unchanged macro aggregation.
Store full model move logits plus top-eight child WDL and root time predictions;
CPU sweeps then reuse these GPU predictions. No golden outcome scores in selection.
The attempted custom development-set build was dropped at the user's request; its
CPU attempt produced no dataset and no GPU work. Existing evaluation remains authoritative.

Use the user-requested persistent one-GPU workbench on preempt, eight hours, fixed frozen checkpoint and source. Checkpoint
prediction batches atomically and resume by input/model/config digest. Keep the model resident across CPU method selection, per the user's explicit preference; record idle allocation time too. Record cold compilation separately from
warm throughput, GPU type, inference calls/tokens and wall time.

Before scaling: verify packed/prefix equivalence and causal prefix truncation, legal
candidate coverage, value perspective, terminal positions and full-policy normalization.
Freeze a method and substantially enlarge independent confirmation before final golden CE.

## Research references

- Zhang et al., Human-aligned Chess with a Bit of Search:
  https://arxiv.org/abs/2410.03893. Time prediction drives adaptive simulation counts;
  expert prediction and playing-strength calibration are distinct evaluation questions.
- Schultz et al., Mastering Board Games by External and Internal Planning with Language Models:
  https://arxiv.org/abs/2412.12119. Model-guided external search without an external engine
  is feasible; its playing-strength improvements do not establish human-move CE gains.

## Status

Bug hunt finished in main worktree: results/codex-bug-report.md. No training files modified.
Worktree and goal created; building the evaluation/candidate cache harness.

The first CPU sweep also includes expert-only application of each correction. This
uses the known mover Elo, leaves all nonexpert predictions exactly unchanged, and
can reduce final search cost by scoring only the four expert cells. Any such restriction
is reported explicitly; it cannot be sold as improving every rating group. The same
conditional option is available to the cheap calibration control.
