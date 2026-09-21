"""Two-GPU compiled-block recovery proof, including ZeRO-2 and embedding split.

Synthetic, deterministic input rows isolate checkpoint recovery from the sampler.
Compare a freshly constructed/resumed model to its uninterrupted continuation,
including every model/optimizer tensor. This does not prove world-size resharding.
"""
import argparse
import gc
import io
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from modded_medium import Config, TrainingManager, create_model, core, make_context, move_losses, cpu_copy
from modded_wsd import Schedule, install


def build():
    torch.manual_seed(701)
    cfg = Config(width=128, head_dim=64, layers=8, max_tokens=1024,
                 scheduled_steps=8, extension_steps=0, initial_batch_rows=8,
                 bf16_weights=True, zero2=True, ckpt='eager',
                 arch=dict(mlp='swiglu', moe=[16, 2], moe_seq=.001, moe_kernel='scatter'))
    m = create_model(cfg, device=torch.device('cuda', int(os.environ['LOCAL_RANK'])))
    mgr = TrainingManager(m, cfg)
    return m, mgr, torch.compile(m, dynamic=False, fullgraph=False)


def advance(m, mgr, net, step):
    mgr.advance_schedule(step)
    core.grad_accum_steps = 2
    primary = []
    for micro in range(2):
        if micro == 1:
            mgr.activate_hooks(step)
        g = torch.Generator(device='cuda').manual_seed(410 + 100 * step + 10 * micro + dist.get_rank())
        rows = torch.randint(378, 2346, (1, 1025), generator=g, device='cuda')
        x, y = rows[:, :-1], rows[:, 1:]
        ctx = make_context(x, mgr.ws_short * 128, mgr.ws_long * 128)
        logits = net(x.flatten(), y.flatten(), ctx, mgr.get_forward_args())
        loss, move, count = move_losses(logits, x, y, ctx, mgr.mtp_weights)
        (loss * (dist.get_world_size() / 8)).backward()
        primary.append(float(move.detach() / count))
    mgr.step_optimizers(step)
    return primary


def equal(a, b, path='root'):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and torch.equal(a, b), path
        return 1
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        return sum(equal(a[k], b[k], f'{path}.{k}') for k in a)
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        return sum(equal(x, y, f'{path}[{i}]') for i, (x, y) in enumerate(zip(a, b)))
    assert a == b, (path, a, b)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', device_id=torch.device('cuda', int(os.environ['LOCAL_RANK'])))
    assert dist.get_world_size() == 2
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    install(core, Schedule(warmup_steps=2, mtp_steps=0, split_step=5, batch_rows=8), -1, 8)
    m, mgr, net = build()
    for s in range(3):
        advance(m, mgr, net, s)
    # Even-index optimizer boundary retains inactive-Adam gradients. Rebuild before split5.
    buf = io.BytesIO()
    torch.save(dict(model=cpu_copy(m.state_dict()), manager=mgr.rank_state_dict()), buf)
    losses = [advance(m, mgr, net, s) for s in range(3, 8)]
    expected = dict(model=cpu_copy(m.state_dict()), manager=mgr.rank_state_dict())
    assert m.split_embed
    del m, mgr, net
    gc.collect()
    torch.cuda.empty_cache()
    m, mgr, net = build()
    buf.seek(0)
    saved = torch.load(buf, weights_only=False)
    m.load_state_dict(saved['model'])
    mgr.load_rank_state_dict(saved['manager'])
    resumed = [advance(m, mgr, net, s) for s in range(3, 8)]
    assert losses == resumed, (losses, resumed)
    count = equal(expected, dict(model=cpu_copy(m.state_dict()), manager=mgr.rank_state_dict()))
    report = dict(equal=True, tensors=count, losses=losses, resumed=resumed, split_embed=m.split_embed)
    (out / f'proof-rank{dist.get_rank()}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(rank=dist.get_rank(), equal=True, tensors=count)), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
