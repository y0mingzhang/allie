"""Frozen 1000-simulation MCTS confirmation on the existing balanced sample."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import os
import numpy as np
from scipy.special import softmax
from .balanced_eval import atomic, digest, metrics, ROOT, GLOBAL_STOP

OUT = ROOT/'golden-mcts1000-v1'


def freeze():
    OUT.mkdir(exist_ok=True)
    calibration = json.loads((ROOT/'mcts1000-pilot/results.json').read_text())['calibration']
    post = json.loads((ROOT/'mcts1000-postprocess/results.json').read_text())
    assert post['selected_by_fit_cv_macro'] == post['selected_by_fit_cv_expert'] == 'elo_q'
    params = post['results']['elo_q']['coefficients']
    files = [Path(__file__), *[Path(__file__).with_name(n) for n in (
        'analyze_golden_mcts.py', 'balanced_eval.py', 'analyze_balanced.py', 'direct.py',
        'startup_env.py', 'model.py', 'adaptive_pilot.py', 'adaptive_policy.py', 'policy.py',
        'native_board.py', 'mcts_native.hpp', 'board.cpp')],
        Path(__file__).parent/'sglang_models/allie.py']
    plan = dict(sample_sha256=digest(ROOT/'golden-balanced-v1/sample.json'),
                serving_weights_sha256=digest(ROOT/'serving-export/model.safetensors'),
                methods=dict(reverse=calibration['reverse'], forward=calibration['forward'],
                             elo=dict(alpha=params[0], beta_low=params[1], beta_high=params[2], knots=[1000, 2600])),
                direct_alpha=json.loads((ROOT/'golden-balanced-v1/direct-control.json').read_text())['alpha'],
                calibration_sha256={name:digest(ROOT/name) for name in (
                    'mcts1000-pilot/results.json', 'mcts1000-postprocess/results.json')},
                source_sha256={str(p.relative_to(Path(__file__).parent)):digest(p) for p in files},
                native_sha256={p.name:digest(p) for p in (ROOT/'runtime/native').glob('*.so')},
                n_sims=1000, roots_per_block=128, repairs=True,
                note='Methods/parameters selected on dev only, frozen before new scoring. Reuses the previously reported golden sample; this is not a new untouched dataset. Report all three methods.')
    path = OUT/'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan, 'Frozen confirmation changed'
    else:
        atomic(path, plan)
    return plan


def run(oracle, spec):
    from .adaptive_pilot import traverse
    from .adaptive_policy import output
    import torch
    plan = freeze()
    proof = json.loads((ROOT/'engine-queue/011-l40s-recovery.result.json').read_text())
    for key in ('port_vs_reference', 'cached_vs_fresh'):
        assert abs(proof[key]['legal_ce_delta']) < .005 and proof[key]['mean_legal_policy_kl'] < .005
    runtime = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
    if (OUT/'runtime.json').exists():
        assert json.loads((OUT/'runtime.json').read_text()) == runtime
    else:
        atomic(OUT/'runtime.json', runtime)
    rows = json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions']
    bs = plan['roots_per_block']
    start = time.monotonic()
    costs = []
    for lo in range(0, len(rows), bs):
        if (ROOT/'STOP').exists() or GLOBAL_STOP.exists():
            raise RuntimeError('STOP requested')
        part = rows[lo:lo+bs]
        dest = OUT/f'{lo:06d}.npz'
        if not dest.exists():
            oracle.reset()
            begin = time.monotonic()
            z = oracle([r['prefix'] for r in part])
            tree, stats = traverse(part, z, oracle, np.full(len(part), plan['n_sims']),
                                   np.full(len(part), 1.25), repairs=True)
            legal = np.zeros_like(tree['values'], bool)
            elo = []
            for i, row in enumerate(part):
                legal[i, np.array(row['legal'])-378] = True
                offset = 3 if (len(row['prefix'])-11) % 2 == 0 else 7
                elo.append(sum(row['prefix'][offset+j]*10**(3-j) for j in range(4)))
            x = z[:, 378:2346].astype(float)
            policies = dict(port_raw=softmax(x, axis=1), legal=output(x, tree['values'], legal, beta=0),
                            calibrated_direct=output(x, tree['values'], legal, plan['direct_alpha'], beta=0))
            for direction in ('forward', 'reverse'):
                p = plan['methods'][direction]
                policies['mcts_'+direction] = output(x, tree['values'], legal, p['alpha'], p['beta'], direction)
            p = plan['methods']['elo']
            fraction = np.clip((np.array(elo)-p['knots'][0])/(p['knots'][1]-p['knots'][0]), 0, 1)
            beta = p['beta_low']*(1-fraction)+p['beta_high']*fraction
            policies['mcts_elo'] = output(x, tree['values']*beta[:, None], legal, p['alpha'], 1., 'forward')
            payload = {name:np.stack(metrics(p, part), axis=1) for name, p in policies.items()}
            stats.update(seconds=time.monotonic()-begin, new_tokens=oracle.new_tokens, forward_seconds=oracle.forward_seconds)
            tmp = dest.with_suffix('.partial')
            with tmp.open('wb') as f:
                np.savez_compressed(f, **payload, stats=json.dumps(stats), game=np.array([r['game'] for r in part]),
                                    ply=np.array([r['ply'] for r in part]))
            tmp.replace(dest)
        with np.load(dest) as f:
            costs.append(json.loads(str(f['stats'])))
        print('Golden MCTS1000', lo+len(part), '/', len(rows), flush=True)
    subprocess.run([sys.executable, '-B', '-m', 'search.engine.analyze_golden_mcts'], check=True,
                   env=dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2'))
    return dict(stage='Frozen balanced MCTS confirmation complete; all methods reported', positions=len(rows),
                elapsed_seconds=time.monotonic()-start, scoring_seconds=sum(c['seconds'] for c in costs),
                evaluated_leaves=sum(c['evaluated_leaves'] for c in costs))


if __name__ == '__main__':
    freeze()
