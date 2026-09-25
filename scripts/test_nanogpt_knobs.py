"""arch adam_every and wd_scale, and modelexp's schedule overrides, on one GPU.

off: a small compiled MoE trains 6 steps (MTP phase and the embed split inside) with the default arch;
its losses and a digest of every parameter go to --dump, to compare with --ref, the same dump from the
commit before the switches (SCRIPTS=that checkout's scripts; its arch has no adam_every / wd_scale).
wd_scale 0.5: every optimizer group's weight decay is half the default's, and training is finite.
adam_every: the Adam and scalar groups run at half the default lr, twice its weight decay and
square-rooted betas, and step on every step (their Adam step counts equal the training steps, and
Adam-trained weights move on even steps too), where the default leaves them untouched on even steps.
schedule: modelexp's sched overrides final_lr / plateau, and mtp 0 gives mtp_steps 0 (weights [1]).
tc_header False: ids 10..377 (base time and increment) map to the unknown ones and occur only at header
positions 1 / 2 of real rows; the model's forward equals the default forward on rows so replaced.

    torchrun --standalone --nproc_per_node=1 scripts/test_nanogpt_knobs.py [--dump F] [--ref F]
"""

import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(
    0, os.environ.get("SCRIPTS", os.path.dirname(os.path.abspath(__file__)))
)
import modded_arch
from lm_data import Packed
from modded_medium import Config, TrainingManager, create_model, make_context
from modded_medium import move_losses
from modded_wsd import Schedule

DATA = "/data/group_data/dei-group/yimingz3/allie/lichess_tokens_v2"
SCHEDULE = Schedule(warmup_steps=2, mtp_steps=4, split_step=5, batch_rows=2)
NEW = "adam_every" in modded_arch.DEFAULTS
ADAM = ("lm_head", "embed", "embed2", "value_embed", "board", "router", "scalars")


def digest(m):
    h = hashlib.sha256()
    for n, p in sorted(m.named_parameters()):
        h.update(n.encode() + p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def train(val, **kw):
    torch.manual_seed(701)
    arch = dict(moe=[16, 2], moe_seq=0.001) | kw
    cfg = Config(
        width=128, head_dim=64, layers=8, max_tokens=1024, scheduled_steps=8, arch=arch
    )
    m = create_model(cfg)
    mgr = TrainingManager(m, cfg, SCHEDULE)
    net = torch.compile(m, dynamic=False, fullgraph=True)
    adam = [p for p in m.parameters() if getattr(p, "label", "") in ADAM]
    losses, moved = [], []
    for step in range(6):
        mgr.advance_schedule(step)
        before = [p.detach().float().clone() for p in adam]
        for micro in range(2):
            if micro == 1:
                mgr.activate_hooks(step)
            rows = torch.as_tensor(val.rows(np.array([100 * step + 10 * micro])))
            x, y = rows[:, :-1].cuda(), rows[:, 1:].cuda()
            ctx = make_context(x, mgr.ws_short * 128, mgr.ws_long * 128)
            logits = net(x.flatten(), y.flatten(), ctx, mgr.get_forward_args())
            loss, move, count = move_losses(logits, x, y, ctx, mgr.mtp_weights)
            (loss / 8).backward()
            losses.append(float(move.detach() / count))
        mgr.step_optimizers(step)
        moved.append(any(not torch.equal(b, p.float()) for b, p in zip(before, adam)))
    assert all(np.isfinite(losses)), losses
    return dict(losses=losses, digest=digest(m), moved=moved), mgr


def groups(mgr, key):
    return [g[key] for o in mgr.optimizers for g in o.param_groups]


def schedule():
    import modelexp as mx

    r = dict(sched=dict(final_lr=0.004, plateau=2.0))
    s, decay = mx.schedule(r, 4690)
    assert s["final_lr"] == 0.004 and s["plateau"] == 2.0, s
    assert (s["warmup_steps"], s["mtp_steps"], s["split_step"], decay) == (
        66,
        132,
        135,
        66,
    )
    assert mx.schedule({}, 4690)[0] == mx.SCHEDULE | dict(
        warmup_steps=66, mtp_steps=132, split_step=135
    )
    s, _ = mx.schedule(dict(sched=dict(mtp=0)), 4690)
    assert s["mtp_steps"] == 0 and s["split_step"] == 135, s
    assert isinstance(
        mx.schedule(dict(sched=dict(mtp=1e-9)), 4690)[0]["mtp_steps"], int
    )
    w = Schedule(**s)
    w.validate()
    assert all(w.mtp(i) == [1.0] for i in range(10))


def tc_header(val):
    """mask_tc_header on the whole vocabulary, TC ids only at header positions 1 / 2 of real rows,
    and a tc_header False forward == the default forward on rows with those tokens replaced."""
    from modded_medium import core

    v = torch.arange(2350, device="cuda")
    want = torch.where(v < 10, v, torch.where(v <= 191, 191, torch.where(v <= 377, 377, v)))
    assert torch.equal(core.mask_tc_header(v), want)
    rows = np.asarray(val.rows(np.arange(0, 400, 50)))
    tc = (rows >= 10) & (rows <= 377)
    bos = rows == 2348
    head = np.zeros_like(tc)
    head[:, 1:] |= bos[:, :-1]
    head[:, 2:] |= bos[:, :-2]
    head[:, :2] = True  # a row may start inside a cut header
    assert tc.any() and np.array_equal(tc, tc & head), "a TC id off a header position"
    torch.manual_seed(701)
    cfg = Config(width=128, head_dim=64, layers=8, max_tokens=1024, scheduled_steps=8, arch=dict(moe=[16, 2]))
    m = create_model(cfg)
    m.eval()
    net = torch.compile(m, dynamic=False, fullgraph=True)
    mgr = TrainingManager(m, cfg, SCHEDULE)
    mgr.advance_schedule(0)
    x = torch.as_tensor(rows[:1, :-1], device="cuda")
    ctx = make_context(x, mgr.ws_short * 128, mgr.ws_long * 128)
    with torch.inference_mode():
        ref = net(core.mask_tc_header(x.flatten()), x.flatten(), ctx, mgr.get_forward_args())
        m.tc_header = False
        got = net(x.flatten(), x.flatten(), ctx, mgr.get_forward_args())
        m.tc_header = True
        raw = net(x.flatten(), x.flatten(), ctx, mgr.get_forward_args())
    assert torch.equal(ref, got) and not torch.equal(ref, raw)
    print(json.dumps(dict(tc_positions=int(tc.sum()))), flush=True)


def main():
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    val = Packed(DATA, "val")
    off, base = train(val)
    print(json.dumps(dict(off=off)), flush=True)
    assert off["moved"] == [False, True] * 3, off["moved"]  # Adam on odd steps only
    if "--dump" in sys.argv:
        with open(sys.argv[sys.argv.index("--dump") + 1], "w") as f:
            json.dump(off, f)
    if "--ref" in sys.argv:
        with open(sys.argv[sys.argv.index("--ref") + 1]) as f:
            ref = json.load(f)
        assert ref["losses"] == off["losses"], (ref["losses"], off["losses"])
        assert ref["digest"] == off["digest"], "parameters differ from the reference"
        print("off == reference: losses and parameters bitwise", flush=True)
    if NEW:
        schedule()
        tc_header(val)
        wd, mgr = train(val, wd_scale=0.5)
        assert groups(mgr, "weight_decay") == [
            0.5 * w for w in groups(base, "weight_decay")
        ]
        assert groups(mgr, "lr") == groups(base, "lr")
        every, mgr = train(val, adam_every=True)
        assert every["moved"] == [True] * 6, every["moved"]
        halves = [
            g["initial_lr"] * (1 if o is mgr.muon_opt else 2)
            for o in mgr.optimizers
            for g in o.param_groups
        ]
        assert halves == groups(base, "initial_lr")
        adam = [g for o in (mgr.adam_opt, mgr.scalar_opt) for g in o.param_groups]
        ref = [g for o in (base.adam_opt, base.scalar_opt) for g in o.param_groups]
        for g, r in zip(adam, ref, strict=True):
            assert g["weight_decay"] == 2 * r["weight_decay"]
            assert g["betas"] == tuple(b**0.5 for b in r["betas"])
        steps = [  # (adam_every, default) Adam step counts per parameter; unused tables stay 0
            (o.state[p]["step"], b.state[q]["step"])
            for o, b in ((mgr.adam_opt, base.adam_opt), (mgr.scalar_opt, base.scalar_opt))
            for g, h in zip(o.param_groups, b.param_groups, strict=True)
            for p, q in zip(g["params"], h["params"], strict=True)
        ]
        assert all(e == 2 * d for e, d in steps) and max(steps) == (6, 3), steps
        print(json.dumps(dict(wd_scale=wd["losses"], adam_every=every["losses"])))
    dist.destroy_process_group()
    print("PASS nanogpt knobs" if NEW else "reference dumped", flush=True)


if __name__ == "__main__":
    main()
