"""Parallel small-file reads feed a single tar stream; no GPU or dependency edits."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import stat
import sys
import tarfile


def paths(root):
    yield root
    for directory,dirs,files in os.walk(root):
        # Runtime imports use package code and metadata, not packaged test suites.
        dirs[:]=[d for d in dirs if d not in ('__pycache__','tests','test')]
        for name in dirs+files:yield Path(directory)/name
        # Thousands of unused architecture-specific cubins dominate metadata I/O.
        # Keep them available through the immutable durable source, on demand.
        if Path(directory).name=='flashinfer_cubin':dirs[:]=[d for d in dirs if d!='cubins']


def prepare(path):
    s=path.lstat();data=None
    if stat.S_ISREG(s.st_mode) and s.st_size<=1024*1024:data=path.read_bytes()
    return path,s,data


def main():
    import io
    root=Path(sys.argv[1]).resolve();iterator=iter(paths(root));pending=deque()
    with ThreadPoolExecutor(max_workers=16) as pool,tarfile.open(fileobj=sys.stdout.buffer,mode='w|',bufsize=1024*1024) as archive:
        archive.copybufsize=4*1024*1024
        for _ in range(64):
            try:pending.append(pool.submit(prepare,next(iterator)))
            except StopIteration:break
        while pending:
            path,s,data=pending.popleft().result();info=tarfile.TarInfo(str(path.relative_to(root)))
            info.mode=stat.S_IMODE(s.st_mode);info.mtime=s.st_mtime;info.uid=s.st_uid;info.gid=s.st_gid
            if path.name=='cubins' and path.parent.name=='flashinfer_cubin':
                info.type=tarfile.SYMTYPE;info.linkname=str(path);archive.addfile(info)
            elif stat.S_ISDIR(s.st_mode):info.type=tarfile.DIRTYPE;archive.addfile(info)
            elif stat.S_ISLNK(s.st_mode):info.type=tarfile.SYMTYPE;info.linkname=os.readlink(path);archive.addfile(info)
            elif stat.S_ISREG(s.st_mode):
                info.size=s.st_size
                if data is not None:archive.addfile(info,io.BytesIO(data))
                else:
                    with path.open('rb') as f:archive.addfile(info,f)
            else:raise RuntimeError(f'Unexpected runtime file type: {path}')
            try:pending.append(pool.submit(prepare,next(iterator)))
            except StopIteration:pass


if __name__=='__main__':main()
