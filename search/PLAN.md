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

## Serving equivalence check

The frozen model stores BF16 rotary tables at absolute flattened coordinates. Moving
32 development prefixes from offset zero into a packed buffer changed CE by +0.00505
on this small probe (mean policy KL 0.000507); this is not a global bias estimate.
The initial oracle therefore failed its check before producing experiment predictions.

The inference adapter now copies the saved rotary coordinates relative to each document
and aligns prefixes to 128-token boundaries. The same 32-position check then had zero
logit difference, zero CE difference and zero policy KL. This is an explicit numerical
inference variant, not a checkpoint/training edit. Final evaluation must report the
unchanged official baseline, canonical-coordinate direct policy, cheap calibration,
and search separately. Do not attribute a numerical baseline change to search.

Persistent allocation 10497511 is live on one RTX6000Ada. The initial 2,048-position
root/top-eight-child cache completed; CPU fitting uses fold0 and reports on fold1.

## First pilot result and expanded confirmation

On the fixed development confirmation fold (1,050 sampled positions), canonical
raw expert CE was 1.505234; legal masking gave 1.495676; the fit-selected one-ply
correction (temperature1, beta2, top8, no time gate) gave 1.492570. Paired game-bootstrap
95% interval for its incremental expert CE change vs legal calibration was
[-0.007996,+0.001695], so this screen does not establish an incremental search win.
Expert top1 accuracy was slightly worse. The settings are frozen in selected.json.

Expanded confirmation: confirm.py applies those unchanged settings to all prepared
fold1 development positions. It does no parameter search. Per-batch metrics, the
fixed plan and resumable prediction caches live under results/search-v1/confirmation.
These are development results, not the exact golden macro metrics.

Transport: durable filesystem polling introduced several seconds per request due to
shared-storage metadata visibility. rpc.py now serves authenticated HTTP directly
from the resident process. Tokens are kept in an untracked mode0600 file and never
printed. A transport-only restart verified exact equality with all 128 first-batch
cached root predictions. It preserves those original artifact identities and writes
the new service identity separately. Expanded confirmation now advances roughly
2,560 positions per 7–13 seconds after startup; use recorded timings, not this
informal rate, in final inference-cost reporting.
