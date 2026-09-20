"""Bound future checkpoint storage while preserving published recovery pointers."""

import copy
import json
import re
import shutil
import signal
import sys
import threading
import time

import torch
import torch.distributed as dist

from lm_checkpoint import atomic_save


def prune(out, keep, current):
    if keep == 0:
        return []
    assert keep >= 2
    root = (out / "checkpoints").resolve()

    def checked(path):
        path = path.resolve()
        assert path.parent == root and re.fullmatch(
            r"step-\d{8}-[0-9a-f]{8}", path.name
        )
        return path

    protected = set()
    for name in ("last.pt", "best.pt", *[p.name for p in out.glob("fork-*.pt")]):
        if (out / name).exists():
            pointer = torch.load(out / name, map_location="cpu", weights_only=False)
            assert pointer["format"] == "allie-modded-medium-1"
            protected.add(checked(out / pointer["directory"]))
    known = protected | {checked(current)}
    log = out / "checkpoints.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            entry = json.loads(line)
            # Old/unpublished directories are not removed by this policy.
            if "directory" in entry:
                path = checked(out / entry["directory"])
                if path.is_dir():
                    known.add(path)
    latest = sorted(known, key=lambda p: (int(p.name.split("-")[1]), p.name))[-keep:]
    removed = []
    for path in sorted(known - protected - set(latest)):
        shutil.rmtree(path)
        removed.append(path.name)
    return removed


def publish(out, directory, step, world, names, save=atomic_save):
    """Point each of names (last.pt, fork-*.pt, best.pt) at a complete checkpoint directory."""
    pointer = {
        "format": "allie-modded-medium-1",
        "directory": str(directory.relative_to(out)),
        "step": step,
        "world_size": world,
    }
    for name in names:
        save(pointer, out / name)


def durable_save(state, path, tries=4):
    """atomic_save retried on OSError (fsync on soft-mounted NFS fails transiently, e.g. errno 512)."""
    for i in range(tries):
        try:
            return atomic_save(state, path)
        except OSError as e:
            if i + 1 == tries:
                raise
            print(
                f"{path}: {e!r}, retry {i + 1} in {2**i}s", file=sys.stderr, flush=True
            )
            time.sleep(2**i)


class AsyncSaver:
    """--async-checkpoint, at most one save in flight. Each rank snapshots its state into host
    buffers reused across saves and a thread writes it; commit (rank 0: pointers, prune) runs on
    the main thread once poll() finds every rank's files durable. The gloo group is used only by
    poll(), which every rank calls at the same step boundaries."""

    def __init__(self, group):
        self.group, self.buffers, self.files, self.roots = group, {}, [], 0
        self.thread = self.commit = None
        self.pending = self.written = False

    def snapshot(self, value):
        """cpu_copy into reused buffers; CUDA tensors are copied asynchronously into pinned
        memory, so the snapshot is complete after torch.cuda.synchronize()."""
        self.roots += 1
        return self._copy(value, (self.roots,))

    def _copy(self, value, path):
        if isinstance(value, torch.Tensor):
            meta = value.dtype, value.shape, value.stride(), value.device
            if path not in self.buffers or self.buffers[path][0] != meta:
                buffer = torch.empty_like(value, device="cpu", pin_memory=value.is_cuda)
                self.buffers[path] = meta, buffer
            return self.buffers[path][1].copy_(value.detach(), non_blocking=True)
        if isinstance(value, dict):
            return {k: self._copy(v, (*path, k)) for k, v in value.items()}
        if isinstance(value, list):
            return [self._copy(v, (*path, i)) for i, v in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(self._copy(v, (*path, i)) for i, v in enumerate(value))
        return copy.deepcopy(value)

    def add(self, state, path):
        self.files.append((state, path))

    def start(self, commit=None):
        assert not self.pending
        files, self.files, self.roots = self.files, [], 0

        def write():
            # process-directed signals go to the main thread, not into this thread's fsync
            signal.pthread_sigmask(
                signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGUSR1}
            )
            for state, path in files:
                durable_save(state, path)
            self.written = True

        self.pending, self.written, self.commit = True, False, commit
        self.thread = threading.Thread(target=write, daemon=True)
        self.thread.start()

    def poll(self, block=False):
        if not self.pending:
            return
        if block:
            self.thread.join()
        alive = self.thread.is_alive()  # before written: a finished thread has set it
        flag = torch.tensor(1 if alive else 0 if self.written else 2)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self.group)
        if flag == 2:
            raise RuntimeError(
                "checkpoint write failed; see the failing rank's writer traceback"
            )
        if flag == 0:
            commit, self.commit, self.pending = self.commit, None, False
            if commit:
                commit()

    def flush(self):
        """Wait for the in-flight save and commit it; returns the seconds waited."""
        t = time.monotonic()
        self.poll(block=True)
        return time.monotonic() - t
