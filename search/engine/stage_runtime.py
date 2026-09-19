"""Copy a pinned private runtime to local NVMe; durable source remains authoritative."""
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import shutil
import time

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results/search-v1/runtime/sglang-0.5.9'
TARGET=Path('/scratch/yimingz3/allie/search-runtime/sglang-0.5.9-torch2.9.1-cu128')


def main():
    target=TARGET.with_name(TARGET.name+'.partial');target.mkdir(parents=True,exist_ok=True)
    if (TARGET/'STAGED.json').exists():print(TARGET);return
    work=queue.Queue();work.put((SOURCE,target));errors=[];files=[];start=time.monotonic()
    def worker():
        while True:
            item=work.get()
            try:
                if item is None:return
                src,dst=item
                if src.is_symlink():
                    if not dst.is_symlink():dst.symlink_to(os.readlink(src))
                elif src.is_dir():
                    dst.mkdir(exist_ok=True)
                    for child in src.iterdir():
                        if child.name!='__pycache__':work.put((child,dst/child.name))
                else:
                    if not dst.exists() or dst.stat().st_size!=src.stat().st_size:shutil.copy2(src,dst)
                    files.append(dst.stat().st_size)
            except Exception as e:errors.append((str(item),repr(e)))
            finally:work.task_done()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        jobs=[pool.submit(worker) for _ in range(16)];work.join()
        for _ in jobs:work.put(None)
        for j in jobs:j.result()
    assert not errors,errors[:10]
    manifest=dict(source=str(SOURCE),files=len(files),bytes=sum(files),seconds=time.monotonic()-start)
    (target/'STAGED.json').write_text(json.dumps(manifest,indent=2)+'\n');target.rename(TARGET)
    print(TARGET,manifest,flush=True)


if __name__=='__main__':main()
