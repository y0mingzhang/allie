"""Two-GPU scatter-gather equivalence and recovery proof, with actual ZeRO-2 hooks.

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

import sys
sys.path.insert(0, os.environ.get("SRC", str(Path(__file__).resolve().parent)))
import torch
import torch.distributed as dist
from modded_medium import Config, TrainingManager, create_model, core, make_context, move_losses, cpu_copy
from modded_wsd import Schedule, install


CANDIDATE = os.environ.get('CANDIDATE', 'scatter-gather')


def build(kernel):
    torch.manual_seed(701)
    cfg = Config(width=128, head_dim=64, layers=8, max_tokens=1024,
                 scheduled_steps=8, extension_steps=0, initial_batch_rows=8,
                 bf16_weights=True, zero2=True, ckpt='eager',
                 arch=dict(mlp='swiglu', moe=[16, 2], moe_seq=.001, moe_kernel=kernel))
    m = create_model(cfg, device=torch.device('cuda', int(os.environ['LOCAL_RANK'])))
    with torch.no_grad():
        for layer in m.modules():
            if hasattr(layer, "experts") and hasattr(layer, "down"):
                layer.down.normal_(0, .02)
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
    assert not core._inflight and not core._pending
    gradients = {n: cpu_copy(dict(main=getattr(p, 'main_grad', None), grad=p.grad))
                 for n, p in m.named_parameters()}
    mgr.step_optimizers(step)
    state = mgr.rank_state_dict()
    state['config']['arch']['moe_kernel'] = 'compared'
    return dict(loss=primary, gradients=gradients, model=cpu_copy(m.state_dict()), manager=state)


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
    m, mgr, net = build('scatter-tuned')
    expected = [advance(m, mgr, net, step) for step in range(8)]
    del m, mgr, net
    gc.collect(); torch.cuda.empty_cache()
    m, mgr, net = build(CANDIDATE)
    counts = []
    saved = None
    for step in range(8):
        actual = advance(m, mgr, net, step)
        counts.append(equal(expected[step], actual, f'step{step}'))
        if step == 2:
            saved = io.BytesIO()
            torch.save(dict(model=cpu_copy(m.state_dict()), manager=mgr.rank_state_dict()), saved)
    assert m.split_embed
    del m, mgr, net
    gc.collect(); torch.cuda.empty_cache()
    m, mgr, net = build(CANDIDATE)
    saved.seek(0)
    state = torch.load(saved, weights_only=False)
    m.load_state_dict(state['model']); mgr.load_rank_state_dict(state['manager'])
    resumed = []
    for step in range(3, 8):
        resumed.append(equal(expected[step], advance(m, mgr, net, step), f'resume{step}'))
    report = dict(equal=True, steps=list(range(8)), loss=[x['loss'] for x in expected],
                  tensors_per_step=counts, resume_tensors=resumed, split_embed=m.split_embed,
                  checks='CE, every FP32 owner gradient, pending grads, model, optimizer masters/momenta; resume through split5')
    (out / f'proof-rank{dist.get_rank()}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(rank=dist.get_rank(), equal=True, tensors=counts)), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
