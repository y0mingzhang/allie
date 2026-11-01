"""Download one month of Lichess/standard-chess-games (Hugging Face) parquet files, verified.

Files are fetched in parallel with curl and each is checked against the size and sha256 that the
Hub records; exits nonzero unless every file verifies.
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from allie.data.pin import sha

REPO = "datasets/Lichess/standard-chess-games"
API = f"https://huggingface.co/api/{REPO}/tree/main/data/year={{}}/month={{}}"
URL = f"https://huggingface.co/{REPO}/resolve/main/{{}}"
TOKEN = Path.home() / ".cache/huggingface/token"  # authenticated requests get a higher rate limit
BEARER = f"Bearer {TOKEN.read_text().strip()}" if TOKEN.exists() else None


def fetch(entry, out):
    dst = out / Path(entry["path"]).name
    ok = lambda: dst.exists() and dst.stat().st_size == entry["size"] and sha(dst) == entry["lfs"]["oid"]
    for _ in range(5):
        if ok():
            return True
        tmp = dst.with_suffix(".tmp")
        subprocess.run(
            [
                "curl",
                "-K",
                "-",
                "-sSL",
                "--retry",
                "5",
                "--max-time",
                "3600",
                "-o",
                str(tmp),
                URL.format(entry["path"]),
            ],
            input=f'header = "Authorization: {BEARER}"\n' if BEARER else "",  # not in argv: ps shows it
            text=True,
        )
        if tmp.exists():
            tmp.rename(dst)
    return ok()


def main(month, out):
    year, mo = month.split("-")
    files = [
        f
        for f in json.load(
            urllib.request.urlopen(
                urllib.request.Request(API.format(year, mo), headers=dict(Authorization=BEARER) if BEARER else {})
            )
        )
        if f["path"].endswith(".parquet")
    ]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(lambda f: fetch(f, out), files))
    print(f"{month}: {sum(ok)}/{len(files)} files verified, {sum(f['size'] for f in files) / 1e9:.1f} GB")
    sys.exit(0 if files and all(ok) else 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("month", help="YYYY-MM")
    p.add_argument("out", help="output directory")
    main(**vars(p.parse_args()))
