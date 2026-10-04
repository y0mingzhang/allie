"""Chess rules and search trees (native/tree.cpp) and value backups (native/value.cpp), compiled on first use
into a cache keyed by their source, the chess-library header, Python and pybind11.

A built module loads without the cache's file lock, and a process's threads load it once: a contended flock
on an NFS cache waits out the server's retry backoff (30 s a round), which stalled live searches."""
import fcntl
import hashlib
import importlib.util
import os
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
import sysconfig
import threading

from allie.data.vocab import MOVE_ID, MOVES  # noqa: F401

SOURCE = Path(__file__).resolve().parent / "native"
_LOCK = threading.Lock()


def load(kind="tree"):
    with _LOCK:
        return _load(kind)


@lru_cache(maxsize=2)
def _load(kind):
    if kind not in ("tree", "value"):
        raise ValueError("native module must be tree or value")
    import pybind11
    source = SOURCE / (kind + ".cpp")
    include = Path(os.environ.get("ALLIE_CHESS_INCLUDE", str(SOURCE / "chess-library")))
    parts = [source.read_bytes(), sys.version.encode(), pybind11.__version__.encode()]
    if kind == "tree":
        parts.append((include / "chess.hpp").read_bytes())
    key = hashlib.sha256(b"\0".join(parts)).hexdigest()
    cache = Path(os.environ.get("ALLIE_SEARCH_CACHE", str(Path.home() / ".cache/allie/search"))) / key
    cache.mkdir(parents=True, exist_ok=True)
    name = "_allie_search_" + kind
    target = cache / (name + sysconfig.get_config_var("EXT_SUFFIX"))
    if not target.exists():
        build(source, include, cache, target)
    spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if kind == "tree":
        module.initialize(MOVES)
    return module


def build(source, include, cache, target):
    import pybind11

    with (cache / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            temp = cache / (target.name + f".{os.getpid()}.tmp")
            try:
                subprocess.run(["c++", "-O3", "-fopenmp", "-std=c++17", "-shared", "-fPIC", str(source),
                                "-I" + pybind11.get_include(), "-I" + sysconfig.get_path("include"),
                                "-I" + str(include), "-o", str(temp)], check=True)
                temp.replace(target)
            finally:
                temp.unlink(missing_ok=True)


def from_prefix(prefix):
    board = load().Position()
    for token in prefix[11:]:
        board.push(int(token))
    return board
