"""Evaluate the recorded scaling laws without cluster access or GPU dependencies."""
import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = ('move', 'expert2400', 'expert2600')


def loss(p, n, d):
    return p['E'] + p['A'] * (n / 1e7) ** -p['alpha'] + p['B'] * (d / 1e8) ** -p['beta']


def optimum(p, nd):
    a, b = p['alpha'], p['beta']
    log_n = (math.log(a * p['A'] / (b * p['B'])) + b * math.log(nd / 1e15)) / (a + b)
    n = 1e7 * math.exp(log_n)
    d = nd / n
    return dict(parameters=n, tokens=d, ND=nd, nominal_flops=6 * nd, ce=loss(p, n, d))


def match(p, target):
    if target <= p['E']:
        raise ValueError(f'Target {target} is at or below fitted floor {p["E"]}')
    base = optimum(p, 1e18)
    gamma = p['alpha'] * p['beta'] / (p['alpha'] + p['beta'])
    nd = 1e18 * ((base['ce'] - p['E']) / (target - p['E'])) ** (1 / gamma)
    return optimum(p, nd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric', choices=METRICS, default='move')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--nd', type=float, help='Our nominal N*D budget; default 1e18')
    mode.add_argument('--node-days', type=float, help='Our budget in equivalent 8-L40S node-days')
    mode.add_argument('--target-original', action='store_true', help='Match original Qwen measured CE')
    parser.add_argument('--mfu', type=float, default=.35, help='Assumed dense BF16 MFU for time conversion')
    args = parser.parse_args()
    if not 0 < args.mfu <= 1:
        parser.error('--mfu must be in (0, 1]')
    data = json.loads((HERE / 'scaling_fit.json').read_text())
    reference = json.loads((HERE / 'original_qwen.json').read_text())
    params = {r: data['fits'][r][args.metric]['fit']['parameters'] for r in ('ours', 'qwen_recipe')}
    flops_per_day = 362.05e12 * 8 * 86400 * args.mfu
    try:
        if args.target_original:
            target = reference['ce'][args.metric]
            ours = match(params['ours'], target)
        else:
            nd = args.nd if args.nd is not None else 1e18
            if args.node_days is not None:
                nd = args.node_days * flops_per_day / 6
            if not math.isfinite(nd) or nd <= 0:
                parser.error('Compute budget must be finite and positive')
            ours = optimum(params['ours'], nd)
            target = ours['ce']
        qwen = match(params['qwen_recipe'], target)
    except ValueError as exc:
        parser.error(str(exc))
    for recipe, allocation in (('ours', ours), ('qwen_recipe', qwen)):
        allocation['equivalent_L40S_node_days'] = allocation['nominal_flops'] / flops_per_day
        rows = [r for r in data['rows'] if r['recipe'] == recipe]
        allocation['extrapolates_N_or_D'] = not (
            min(r['parameters'] for r in rows) <= allocation['parameters'] <= max(r['parameters'] for r in rows)
            and min(r['tokens'] for r in rows) <= allocation['tokens'] <= max(r['tokens'] for r in rows))
    result = dict(metric=args.metric, target_ce=target, ours=ours, optimal_qwen_matching=qwen,
                  CM_vs_optimal_qwen=qwen['ND'] / ours['ND'],
                  assumption=f'Fitted laws are correct; node-days use 8 L40S, 362.05 TFLOP/s dense BF16, {args.mfu:.0%} MFU.')
    if args.target_original:
        result['original_qwen_nominal_flops'] = reference['nominal_flops']
        result['original_qwen_equivalent_node_days'] = reference['nominal_flops'] / flops_per_day
        result['CM_vs_original_run'] = reference['nominal_flops'] / ours['nominal_flops']
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
