"""Medium-family analytical prefill costs on development histories only."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
reference = json.loads((ROOT/'results/inference-flops.json').read_text())['maia3_79m']
report = dict(reference=reference, splits={},
    convention='Model matmuls at actual development prefix lengths, final full-history windows. All prefix positions include the output head, matching current forward. Dense and useful causal attention conventions both reported; elementwise/optimizer/CPU work excluded.',
    limitations='Analytical FLOPs, not latency. Cold128-padding is a hypothetical one-prefix-per-target workload. The actual evaluator scores many moves from each full game in one padded forward; its amortized protocol is reported separately. No KV cache implementation or measured cached-inference claim. No final test access.')
for split in ('dev', 'dev_expert'):
    shards = [dict(np.load(ROOT/'results'/split/f'meta-{r}.npz')) for r in range(4)]
    game = np.concatenate([s['game'] for s in shards])
    ply = np.concatenate([s['ply'] for s in shards])
    elo = np.concatenate([s['elos'][:, 0] for s in shards])
    eval_tokens, eval_dense_pairs, eval_useful_pairs = 0, 0, 0
    for rank in range(4):
        prompts = json.loads((ROOT/'data'/f'prepared-{split}-4-{rank}.json').read_text())
        prompts.sort(key=lambda x:len(x[0]))
        for lo in range(0, len(prompts), 16):
            batch = prompts[lo:lo+16]
            length = ((max(len(x[0]) for x in batch)+127)//128)*128
            # Evaluator pads its final partial batch to16 isolated BOS rows.
            eval_tokens += 16*length
            eval_dense_pairs += 16*length**2
            eval_useful_pairs += (16-len(batch))*length
            for tokens, _, _ in batch:
                assert tokens.count(2348) == 1 and tokens[0] == 2348
                # Evaluator pads with BOS, each padding token its own document.
                eval_useful_pairs += len(tokens)*(len(tokens)+1)//2+length-len(tokens)
    groups = {}
    for label, mask in [('all', np.ones(len(game), bool)), ('expert2400', elo >= 2400), ('expert2600', elo >= 2600)]:
        length = ply[mask].astype(np.float64)+11
        padded = np.ceil(length/128)*128
        # All these dev histories fit even the final short attention window.
        assert length.max() <= 11*128 and padded.max() <= 11*128
        ids, inverse = np.unique(game[mask], return_inverse=True)
        longest = np.zeros(len(ids))
        np.maximum.at(longest, inverse, length)
        models = {}
        for width in (384, 448, 512, 576, 768, 1024, 1536):
            heads = width//64
            base = 24*16*width**2+2*width*2432+2*(26*heads*16+64)
            attention = 4*16*width
            parameters = 192*width**2+8*2432*width+26*16*heads+64+32+53
            cold_dense = base*length+attention*length**2
            useful = base*length+attention*length*(length+1)/2
            padded_dense = base*padded+attention*padded**2
            amortized = (base*longest+attention*longest*(longest+1)/2).sum()/len(length)
            models[f'w{width}'] = dict(parameters_world1=parameters,
                cold_dense_mean_flops=float(cold_dense.mean()), cold_useful_mean_flops=float(useful.mean()),
                cold_dense_p90_flops=float(np.quantile(cold_dense, .9)),
                cold_dense_max_flops=float(cold_dense.max()),
                cold_dense_mean_ratio_to_maia=float(cold_dense.mean()/reference['per_position_flops']),
                fraction_positions_cold_dense_le_maia=float((cold_dense <= reference['per_position_flops']).mean()),
                hypothetical_cold128_padded_dense_mean_flops=float(padded_dense.mean()),
                amortized_game_useful_flops_per_target=float(amortized))
            if label == 'all':
                models[f'w{width}']['current_eval_batch16_useful_flops_per_target'] = float((base*eval_tokens+attention*eval_useful_pairs)/len(game))
                models[f'w{width}']['current_eval_batch16_dense_reference_flops_per_target'] = float((base*eval_tokens+attention*eval_dense_pairs)/len(game))
        groups[label] = dict(positions=len(length), games=len(ids), mean_prefix_tokens=float(length.mean()),
                             mean_hypothetical_cold128_padded_tokens=float(padded.mean()), models=models)
    report['splits'][split] = groups
(ROOT/'results/modded-inference-workloads-dev.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({s: {w:g['all']['models'][w] for w in ('w512', 'w576', 'w1024')}
                  for s,g in report['splits'].items()}, indent=2))
