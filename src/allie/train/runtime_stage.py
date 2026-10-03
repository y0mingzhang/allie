"""Stage the pinned Python runtime as an immutable node-local cache."""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import subprocess
import sys
import tarfile
import tempfile

BASE = Path(os.environ.get("ALLIE_DATA", "/data/group_data/dei-group/yimingz3/allie")) / 'envs'
SOURCE = BASE/'modded-torch210'
MANIFEST = BASE/'modded-torch210-runtime.json'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def build():
    if MANIFEST.exists():
        info = json.loads(MANIFEST.read_text())
        assert digest(BASE/info['archive']) == info['sha256']
        return info
    record = SOURCE/'lib/python3.12/site-packages/torch-2.10.0+cu128.dist-info/RECORD'
    record_sha = digest(record)
    with tempfile.NamedTemporaryFile(dir=BASE, suffix='.tar.partial', delete=False) as f:
        temp = Path(f.name)
    try:
        subprocess.run(['tar', '--sort=name', '--exclude=__pycache__', '--exclude=*.pyc',
                        '-cf', str(temp), '-C', str(SOURCE), '.'], check=True)
        assert digest(record) == record_sha, 'Runtime changed while archiving'
        sha = digest(temp)
        archive = BASE/f'modded-torch210-{sha[:16]}.tar'
        temp.replace(archive)
    finally:
        temp.unlink(missing_ok=True)
    info = dict(archive=archive.name, sha256=sha, bytes=archive.stat().st_size,
                torch='2.10.0+cu128', torch_record_sha256=record_sha,
                note='Exact durable environment, excluding regenerable Python bytecode caches')
    MANIFEST.write_text(json.dumps(info, indent=2)+'\n')
    return info


def ensure():
    info = json.loads(MANIFEST.read_text())
    parent = Path('/scratch/yimingz3/allie/runtimes')
    parent.mkdir(parents=True, exist_ok=True)
    target = parent/f'torch210-{info["sha256"][:16]}'
    marker = target/'allie-runtime.json'
    if marker.exists():
        assert json.loads(marker.read_text()) == info
        return target
    temp = Path(tempfile.mkdtemp(dir=parent, prefix='torch210-stage-'))
    try:
        archive = temp/'runtime.tar'
        print('Staging pinned torch runtime to local cache', file=sys.stderr, flush=True)
        shutil.copyfile(BASE/info['archive'], archive)
        assert digest(archive) == info['sha256']
        unpacked = temp/'environment'
        unpacked.mkdir()
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            for member in members:
                path = PurePosixPath(member.name)
                assert not path.is_absolute() and '..' not in path.parts
                assert member.isfile() or member.isdir() or member.issym() or member.islnk()
                if member.issym():
                    if member.linkname.startswith('/'):
                        assert str(path) == 'bin/python' and member.linkname == '/usr/bin/python3.12'
                    else:
                        resolved = posixpath.normpath(posixpath.join(str(path.parent), member.linkname))
                        assert resolved != '..' and not resolved.startswith('../')
                if member.islnk():
                    link = PurePosixPath(member.linkname)
                    assert not link.is_absolute() and '..' not in link.parts
                member.mode &= 0o777
            # The only permitted external link is the existing system Python.
            tar.extractall(unpacked, members=members, filter='fully_trusted')
        record = unpacked/'lib/python3.12/site-packages/torch-2.10.0+cu128.dist-info/RECORD'
        assert digest(record) == info['torch_record_sha256']
        (unpacked/'allie-runtime.json').write_text(json.dumps(info, indent=2)+'\n')
        try:
            unpacked.rename(target)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            assert json.loads(marker.read_text()) == info
    finally:
        shutil.rmtree(temp)
    return target


def reexec_local():
    if not MANIFEST.exists():
        return  # Bundle preparation may not have finished on the controller.
    target = ensure()
    if Path(sys.prefix).resolve() != target.resolve():
        python = str(target/'bin/python')
        os.execv(python, [python, *sys.argv])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    if args.build:
        print(json.dumps(build()))
    else:
        print(ensure()/'bin/python')
