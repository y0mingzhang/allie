"""Strict, explicit stable-prefix horizon continuation; copied into frozen source."""
from pathlib import Path

# Filled by prepare_wsd_continuation.py from the verified WSD-v1 source manifest.
PARENT_SOURCE = {'modded_medium.py': 'bd05c09b29023d459f2e43754b9e383e121474c78cac38f0c4bd1b7c7ce30a34', 'modded_medium_core.py': '07612bbf8dea6c1d7d3813bb8329577ec93e15c3648936ea6363e65c7e95e8d5', 'lm_data.py': '1a97292c1823a58bde101f70264589c1ca1c0f9b59e68981f60f968f1c6a24ee', 'modded_checkpoints.py': '86ab047b47baf08175adbfa09521b49cd012388bde67d79c957b39ce0ab576fd', 'modded_train.py': 'aa8d82c8c1584102bedf7393401422a6326b222394f6abbc2cfcad7189fddc87', 'modded_runtime_stage.py': '7939f5807e705466a286d066d555a324db3d3373867a86bb698dd9222854dfc8', 'modded_wsd.py': '2cd14cea6751fa2335c6ea7a76dbf20858f49d418c543a3a5d5ca749fa6228db', 'modded_inference_workloads.py': 'a332add022e72ad25e1db11908ad7c26048afe3b4070c31300f59cf9dd22cd1c', 'lm_checkpoint.py': '1068224412046d4fa3a9fb2cfe08425ed0f89188ca2b8268ddc604bca50d325c', 'modded_runtime.py': '99b088671495a6164cdb8cb605219cf170785d26e249b5b000bd5117083a3c96'}


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
