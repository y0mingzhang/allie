"""Read existing frozen golden reports; attach measured search work, no tuning."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'results/search-v1'


def main():
    first = ROOT / 'golden-balanced-v1'
    report = json.loads((first / 'results.json').read_text())
    n = report['positions']
    costs = {}
    depths = np.zeros(4)
    for f in sorted(first.glob('tree-*.npz')):
        with np.load(f) as z:
            stats = json.loads(str(z['stats']))
        depths += stats['leaves_by_depth']
    costs['calibrated_two_ply'] = dict(nodes=float(depths[:2].sum()) / n, seconds=None)
    costs['calibrated_four_ply'] = dict(nodes=float(depths.sum()) / n,
        seconds=report['timing']['summed_four_ply_block_seconds'])
    for name, key in [('released_adaptive', 'released'), ('repaired_fixed', 'repaired')]:
        nodes = seconds = simulations = 0
        for f in sorted(first.glob('mcts-*.npz')):
            with np.load(f) as z:
                stats = json.loads(str(z['stats']))[key]
            nodes += stats['evaluated_leaves']
            seconds += stats['seconds']
            simulations += stats['simulations']
        costs[name] = dict(nodes=nodes/n, seconds=seconds, simulations=simulations/n)
    records = []

    def append(name, values, cost):
        records.append(dict(method=name, average_search_nodes=cost['nodes'],
                            average_simulations=cost.get('simulations'),
                            warm_seconds=cost['seconds'],
                            macro_ce=values['macro'], expert_ce=values['expert_macro'],
                            macro_cm=values['macro_training_eq_cm'],
                            expert_cm=values['expert_macro_training_eq_cm']))

    for name in ['port_raw', 'legal', 'calibrated_direct', *costs]:
        append(name, report['methods'][name], costs.get(name, dict(nodes=0, seconds=None)))
    later = ROOT / 'golden-mcts1000-v1/results.json'
    if later.exists():
        d = json.loads(later.read_text())
        for name in ['mcts_forward', 'mcts_reverse', 'mcts_elo']:
            append(name, d['methods'][name], dict(nodes=d['evaluated_leaves']/d['positions'],
                seconds=d['scoring_seconds'], simulations=1000))
    for r in records:
        r['point_dominated_by'] = [s['method'] for s in records if s is not r
            and s['average_search_nodes'] <= r['average_search_nodes']
            and s['macro_ce'] <= r['macro_ce'] and s['expert_ce'] <= r['expert_ce']
            and (s['average_search_nodes'] < r['average_search_nodes']
                 or s['macro_ce'] < r['macro_ce'] or s['expert_ce'] < r['expert_ce'])]
    result = dict(rows=records, golden_positions=n,
        node_definition='New non-root positions evaluated by the model; terminal backups do not count as model evaluations. Root prefill is additional for every method.',
        population='512 positions in each of 16 golden cells: the sample average equals the equal-cell macro average. A population estimate, not all 1.55M moves.',
        caveats=['Dominance flags compare point estimates; they do not establish statistically supported dominance.',
                 'MCTS50 seconds measure traversal; MCTS1000/four-ply include root evaluation. Two-ply standalone golden wall time was not measured.',
                 'Original blocks lack per-root node counts, so expert-only node cost cannot yet be recovered.',
                 'CM intervals and law-shape sensitivity remain in the source reports.'])
    path = ROOT / 'frontier.json'
    tmp = path.with_suffix('.partial'); tmp.write_text(json.dumps(result, indent=2)+'\n'); tmp.replace(path)
    lines = ['# Golden search cost and quality', '', result['node_definition'], '', result['population'], '',
             '| Method | Avg search nodes | Macro CE | Expert CE | Macro CM | Expert CM | Point dominated by |',
             '|---|---:|---:|---:|---:|---:|---|']
    for r in sorted(records, key=lambda r:r['average_search_nodes']):
        lines.append(f"| {r['method']} | {r['average_search_nodes']:.1f} | {r['macro_ce']:.4f} | {r['expert_ce']:.4f} | {r['macro_cm']:.2f}× | {r['expert_cm']:.2f}× | {', '.join(r['point_dominated_by']) or '—'} |")
    lines += ['', *result['caveats']]
    (ROOT / 'FRONTIER.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
