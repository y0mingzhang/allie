"""Stage the private runtime from one checksummed stream onto node-local storage."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=ROOT/'results/search-v1/runtime/sglang-0.5.9-torch2.9.1-cu128.tar.zst'
TARGET=Path('/scratch/yimingz3/allie/search-runtime/sglang-0.5.9-torch2.9.1-cu128')


def main():
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    with (TARGET.parent/'archive-stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (TARGET/'STAGED.json').exists():return
        expected=Path(str(ARCHIVE)+'.sha256').read_text().split()[0]
        start=time.monotonic();local=TARGET.parent/(ARCHIVE.name+'.partial')
        print('Copying runtime archive to local storage',flush=True)
        h=hashlib.sha256()
        with ARCHIVE.open('rb') as src,local.open('wb') as dst:
            while block:=src.read(8*1024*1024):
                dst.write(block);h.update(block)
        assert h.hexdigest()==expected,'Runtime archive checksum mismatch'
        copied=time.monotonic();temporary=TARGET.with_name(TARGET.name+f'.archive-{os.getpid()}')
        temporary.mkdir(exist_ok=False)
        subprocess.run(['tar','--zstd','-xf',str(local),'-C',str(temporary)],check=True)
        manifest=dict(source=str(ARCHIVE),sha256=expected,copy_seconds=copied-start,
                      extraction_seconds=time.monotonic()-copied,total_seconds=time.monotonic()-start)
        (temporary/'STAGED.json').write_text(json.dumps(manifest,indent=2)+'\n')
        temporary.rename(TARGET);local.unlink()
        print('Runtime staged',manifest,flush=True)


if __name__=='__main__':main()
