"""arch adam_every and modelexp's schedule overrides, on one GPU.

off: a small compiled MoE trains 6 steps (MTP phase and the embed split inside) with the default arch;
its losses and a digest of every parameter go to --dump, to compare with --ref, the same dump from
another commit (made by that commit's own copy of this check; a86c10a4 or later: earlier copies train
prop balancing, the default MoE update before Quantile Balancing was the only one).
adam_every: the Adam and scalar groups run at half the default lr, twice its weight decay and
square-rooted betas, and step on every step (their Adam step counts equal the training steps, and
Adam-trained weights move on even steps too), where the default leaves them untouched on even steps.
schedule: modelexp's sched overrides final_lr / plateau, and mtp 0 gives mtp_steps 0 (weights [1]).

    torchrun --standalone --nproc_per_node=1 tests/checks/nanogpt_knobs.py [--dump F] [--ref F]
"""

import hashlib
import json
import sys

import numpy as np
import torch
import torch.distributed as dist

from allie import paths
from allie.model import arch as model_arch
from allie.data.packed import Packed
from allie.model.network import Config, TrainingManager, create_model, make_context
from allie.model.network import move_losses
from allie.train.schedule import Schedule

DATA = str(paths.DATA / "lichess_tokens_v2")
SCHEDULE = Schedule(warmup_steps=2, mtp_steps=4, split_step=5, batch_rows=2)
NEW = "adam_every" in model_arch.DEFAULTS
ADAM = ("lm_head", "embed", "embed2", "value_embed", "board", "router", "scalars")


def digest(m):
    h = hashlib.sha256()
    for n, p in sorted(m.named_parameters()):
        h.update(n.encode() + p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def train(val, **kw):
    torch.manual_seed(701)
    arch = dict(moe=[16, 2], moe_seq=0.001, moe_update="quantile") | kw
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
            logits = net(x.flatten(), ctx, mgr.get_forward_args())
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
    from allie.experiments import modelexp as mx

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
            for o, b in (
                (mgr.adam_opt, base.adam_opt),
                (mgr.scalar_opt, base.scalar_opt),
            )
            for g, h in zip(o.param_groups, b.param_groups, strict=True)
            for p, q in zip(g["params"], h["params"], strict=True)
        ]
        assert all(e == 2 * d for e, d in steps) and max(steps) == (6, 3), steps
        print(json.dumps(dict(adam_every=every["losses"])))
    dist.destroy_process_group()
    print("PASS nanogpt knobs" if NEW else "reference dumped", flush=True)


if __name__ == "__main__":
    main()
