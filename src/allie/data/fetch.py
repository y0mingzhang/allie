"""Download one month of Lichess/standard-chess-games (Hugging Face) parquet files, verified.

Usage: data.fetch YYYY-MM OUT_DIR. Files are fetched in parallel with curl and each is checked
against the size and sha256 that the Hub records; exits nonzero unless every file verifies.
"""

import hashlib
import json
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "datasets/Lichess/standard-chess-games"
API = f"https://huggingface.co/api/{REPO}/tree/main/data/year={{}}/month={{}}"
URL = f"https://huggingface.co/{REPO}/resolve/main/{{}}"
TOKEN = Path.home() / ".cache/huggingface/token"  # authenticated requests get a higher rate limit
AUTH = ["-H", f"Authorization: Bearer {TOKEN.read_text().strip()}"] if TOKEN.exists() else []


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch(entry, out):
    dst = out / Path(entry["path"]).name
    ok = lambda: (
        dst.exists()
        and dst.stat().st_size == entry["size"]
        and sha256(dst) == entry["lfs"]["oid"]
    )
    for _ in range(5):
        if ok():
            return True
        tmp = dst.with_suffix(".tmp")
        subprocess.run(
            [
                "curl",
                "-sSL",
                "--retry",
                "5",
                "--max-time",
                "3600",
                *AUTH,
                "-o",
                str(tmp),
                URL.format(entry["path"]),
            ]
        )
        if tmp.exists():
            tmp.rename(dst)
    return ok()


def main(month, out):
    year, mo = month.split("-")
    files = [
        f
        for f in json.load(urllib.request.urlopen(urllib.request.Request(
            API.format(year, mo), headers=dict([AUTH[1].split(": ", 1)]) if AUTH else {})))
        if f["path"].endswith(".parquet")
    ]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(lambda f: fetch(f, out), files))
    print(
        f"{month}: {sum(ok)}/{len(files)} files verified, {sum(f['size'] for f in files) / 1e9:.1f} GB"
    )
    sys.exit(0 if files and all(ok) else 1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
