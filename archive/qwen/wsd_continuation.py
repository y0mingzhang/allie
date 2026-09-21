"""Strict, explicit stable-prefix horizon continuation; copied into frozen source."""
from pathlib import Path

# Filled by prepare_wsd_continuation.py from the verified WSD-v1 source manifest.
PARENT_SOURCE = None


def prepare_continuation(shared, local, args, config, source_hashes, runtime, output, pointer):
    assert PARENT_SOURCE is not None, 'Use the generated frozen continuation source'
    assert shared['source_sha256'] in (PARENT_SOURCE, source_hashes), 'Unapproved source migration'
    assert shared['runtime'] == runtime, 'Continuation requires the exact runtime'
    assert shared['args']['wsd_decay_start'] == -1, 'Continue only a stable prefix'
    assert shared['config'] == local['manager']['config'], 'Rank/model configuration mismatch'
    old = shared['config']
    assert set(old) == set(config)
    for key in old:
        if key == 'scheduled_steps':
            assert config[key] >= old[key], 'Cannot shorten the configured horizon'
        else:
            assert old[key] == config[key], f'Continuation changes {key}'
    for key in ('width', 'head_dim', 'extension_steps', 'initial_batch_rows', 'micro_batch',
                'lr_scale', 'seed', 'deterministic', 'wsd_schedule'):
        assert shared['args'][key] == args[key], f'Continuation changes {key}'
    assert old['scheduled_steps'] == shared['args']['steps']
    assert config['scheduled_steps'] == args['steps']
    assert args['extension_steps'] == 0
    assert 0 < shared['step'] < args['wsd_end_step'] <= args['steps']
    assert args['wsd_decay_start'] == -1 or args['wsd_decay_start'] >= shared['step'], 'Cannot change past LR updates'
    assert shared['tokens'] == local['data']['seen']*1024
    assert shared['step'] <= shared['args']['wsd_end_step']
    assert local['manager']['schedule_step'] == shared['step']-1
    assert Path(pointer).resolve().parent != Path(output).resolve()
    assert not (Path(output)/'last.pt').exists(), 'Continue into a fresh run; use resume thereafter'
    # Tensor states, moments, optimizer flags, gradients, RNG and loader state
    # are unchanged. Only the manager's allowed horizon identity is migrated.
    migrated = dict(local)
    migrated['manager'] = dict(local['manager'], config=dict(config))
    provenance = dict(parent_pointer=str(Path(pointer).resolve()), parent_step=shared['step'],
                      parent_tokens=shared['tokens'], parent_useful_training_flops=shared['useful_training_flops'],
                      parent_config=old, parent_source_sha256=shared['source_sha256'],
                      previous=shared.get('continuation_provenance'),
                      changed_configuration_fields=['scheduled_steps'] if old['scheduled_steps'] != config['scheduled_steps'] else [],
                      accounting='Inherited tokens/model FLOPs include the already-paid prefix; new allocation elapsed time starts at zero.')
    return migrated, provenance
