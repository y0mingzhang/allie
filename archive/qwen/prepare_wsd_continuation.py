"""Build a separate continuation-capable source without editing WSD-v1."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'results/scaling/wsd-v1/source'
EXPECTED_TRAINER='aa8d82c8c1584102bedf7393401422a6326b222394f6abbc2cfcad7189fddc87'


def prepare(target):
    manifest=json.loads((PARENT/'source-sha256.json').read_text())
    assert manifest['modded_train.py']==EXPECTED_TRAINER
    for name,digest in manifest.items():
        assert hashlib.sha256((PARENT/name).read_bytes()).hexdigest()==digest
    target=Path(target);target.mkdir(parents=True,exist_ok=False)
    for name in manifest:shutil.copy2(PARENT/name,target/name)
    allowed={name:digest for name,digest in manifest.items() if name.endswith('.py')}
    helper=(ROOT/'scripts/wsd_continuation.py').read_text()
    assert helper.count('PARENT_SOURCE = None')==1
    helper=helper.replace('PARENT_SOURCE = None','PARENT_SOURCE = '+repr(allowed))
    (target/'modded_continuation.py').write_text(helper)
    trainer=(target/'modded_train.py').read_text()
    edits=[
        ("    p.add_argument('--wsd-fork-from')", "    p.add_argument('--wsd-fork-from')\n    p.add_argument('--wsd-continue-from')"),
        ('    assert not (a.resume and a.wsd_fork_from)', '    assert sum(bool(x) for x in (a.resume, a.wsd_fork_from, a.wsd_continue_from)) <= 1'),
        ('    load_path = a.resume or a.wsd_fork_from', '    continuation_provenance = None\n    load_path = a.resume or a.wsd_fork_from or a.wsd_continue_from'),
        ("        for key in ('width', 'head_dim', 'steps', 'extension_steps', 'initial_batch_rows', 'micro_batch', 'lr_scale', 'seed', 'deterministic', 'wsd_schedule'):",
         "        for key in ('width', 'head_dim', 'extension_steps', 'initial_batch_rows', 'micro_batch', 'lr_scale', 'seed', 'deterministic', 'wsd_schedule'):")
    ]
    for old,new in edits:
        assert trainer.count(old)==1,(old,trainer.count(old));trainer=trainer.replace(old,new)
    start=trainer.index('        if a.wsd_fork_from:\n')
    end=trainer.index("        model.load_state_dict(shared['model'])",start)
    original=trainer[start:end]
    guarded='''        if a.wsd_continue_from:
            from modded_continuation import prepare_continuation
            local, continuation_provenance = prepare_continuation(
                shared, local, vars(a), asdict(cfg), source_hashes, runtime, out, load_path)
        else:
            assert shared['args']['steps'] == a.steps, 'Resume changes steps'
            continuation_provenance = shared.get('continuation_provenance')
'''+''.join('    '+line+'\n' for line in original.splitlines())
    trainer=trainer[:start]+guarded+trainer[end:]
    old="        if a.wsd_fork_from:\n            # Endpoint model compute includes prefix; branch allocation time does not."
    assert trainer.count(old)==1
    trainer=trainer.replace(old,"        if a.wsd_fork_from or a.wsd_continue_from:\n            # Endpoint model compute includes prefix; branch allocation time does not.")
    old="                elapsed_seconds=elapsed_prior+time.monotonic()-start), directory/'model.pt')"
    assert trainer.count(old)==1
    trainer=trainer.replace(old,"                continuation_provenance=continuation_provenance,\n"+old)
    old="        job_id=os.environ.get('SLURM_JOB_ID'),"
    assert trainer.count(old)==1
    trainer=trainer.replace(old,"        continuation_provenance=continuation_provenance,\n"+old)
    compile(trainer,str(target/'modded_train.py'),'exec');compile(helper,str(target/'modded_continuation.py'),'exec')
    (target/'modded_train.py').write_text(trainer)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in target.iterdir() if p.is_file()}
    (target/'source-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(dict(source=str(target),sha256=hashes)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('target',type=Path)
    prepare(parser.parse_args().target)
