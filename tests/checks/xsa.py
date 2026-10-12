"""arch xsa on one GPU: a small compiled MoE trains 6 steps with and without it. XSA starts at zero, so the
first step's losses equal the plain run's bitwise; its strengths move and training stays finite.

    torchrun --standalone --nproc_per_node=1 tests/checks/xsa.py
"""

import hashlib

import numpy as np
import torch
import torch.distributed as dist

from allie import paths
from allie.data.packed import Packed
from allie.model.network import Config, TrainingManager, create_model, make_context
from allie.model.network import move_losses
from allie.train.schedule import Schedule

DATA = str(paths.DATA / "lichess_tokens_v2")


def digest(m):
    h = hashlib.sha256()
    for n, p in sorted(m.named_parameters()):
        h.update(n.encode() + p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def train(val, **kw):
    torch.manual_seed(701)
    arch = dict(moe=[16, 2], moe_seq=0.001, moe_update="quantile", fp32_small_masters=True) | kw
    cfg = Config(
        width=128, head_dim=64, layers=12, max_tokens=1024, scheduled_steps=8, arch=arch
    )
    m = create_model(cfg)
    sched = Schedule(
        warmup_steps=2, mtp_steps=0, split_step=5, batch_rows=2, momentum_warmup=True
    )
    mgr = TrainingManager(m, cfg, sched, decay_start=2, end_step=8)
    net = torch.compile(m, dynamic=False, fullgraph=True)
    losses = []
    for step in range(6):
        mgr.advance_schedule(step)
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
    assert all(np.isfinite(losses)), losses
    return dict(losses=losses, digest=digest(m)), m


def main():
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    val = Packed(DATA, "val")
    off, _ = train(val)
    on, m = train(val, xsa=True)
    assert on["losses"][:2] == off["losses"][:2], (on["losses"], off["losses"])
    assert on["digest"] != off["digest"] and m.xsa_alphas.detach().abs().sum() > 0
    dist.destroy_process_group()
    print("PASS xsa", flush=True)


if __name__ == "__main__":
    main()
