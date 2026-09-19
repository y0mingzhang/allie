"""Sequentially prefetch mapped shared-filesystem libraries into the node cache.

A cold mmap import can fault thousands of tiny network reads. This bounded helper
warms the same files in parallel, without modifying dependencies or model state.
"""
import argparse
import concurrent.futures
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('pid',type=int);a=p.parse_args()
    lines=Path(f'/proc/{a.pid}/maps').read_text().splitlines()
    paths={Path(l.split()[-1]) for l in lines if '/allie/worktrees/search-v1/results/search-v1/runtime/' in l}
    paths=[p for p in paths if p.is_file() and p.stat().st_size>50_000_000]
    start=time.monotonic()
    def warm(path):
        buf=bytearray(4*1024*1024);n=0
        with path.open('rb',buffering=0) as f:
            while size:=f.readinto(buf):n+=size
        return n
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:total=sum(pool.map(warm,paths))
    print(dict(files=len(paths),bytes=total,seconds=time.monotonic()-start),flush=True)


if __name__=='__main__':main()
