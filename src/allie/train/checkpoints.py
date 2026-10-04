"""Checkpoint writing: durable atomic saves, the pointer files, pruning, the async saver and rank 0's
log writer."""

import copy
import json
import mmap
import os
import queue
import re
import shutil
import signal
import sys
import threading
import time
import weakref

import torch
import torch.distributed as dist

CHUNK = 16 << 20


class Paced:
    """Write-through file that fdatasyncs every CHUNK bytes and asks the kernel to drop them from the
    page cache. Buffered, a multi-GB save sits dirty on the node and uncommitted on the server, and
    the trainer's own small NFS writes wait behind it."""

    def __init__(self, f, chunk):
        assert chunk > 0
        self.f, self.chunk, self.dirty, self.error = f, chunk, 0, None

    def write(self, b):
        view = memoryview(b).cast("B")
        n = len(view)
        try:
            while view:
                k = self.f.write(view[: self.chunk - self.dirty])
                if not k:
                    raise OSError(f"write made no progress ({k!r})")
                view, self.dirty = view[k:], self.dirty + k
                if self.dirty == self.chunk:
                    self.sync(os.fdatasync)
        except OSError as e:  # torch.save masks it with its own RuntimeError
            self.error = e
            raise
        return n

    def sync(self, flush):
        flush(self.f.fileno())
        os.posix_fadvise(self.f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        self.dirty = 0

    def flush(self):
        pass


def atomic_save(state, path):
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb", buffering=0) as f:
        paced = Paced(f, CHUNK)
        try:
            torch.save(state, paced)
        finally:
            if paced.error:
                raise paced.error
        paced.sync(os.fsync)
    partial.replace(path)


def prune(out, keep, current):
    if keep == 0:
        return []
    assert keep >= 1  # the pointers move to a durable checkpoint before this runs
    root = (out / "checkpoints").resolve()

    def checked(path):
        path = path.resolve()
        assert path.parent == root and re.fullmatch(
            r"step-\d{8}-[0-9a-f]{8}", path.name
        )
        return path

    protected = set()
    names = ("last.pt", "best.pt", "final.pt", *[p.name for p in out.glob("fork-*.pt")])
    for name in names:
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


def publish(out, directory, step, world, names):
    """Point each of names (last.pt, fork-*.pt, best.pt) at a complete checkpoint directory."""
    pointer = {
        "format": "allie-modded-medium-1",
        "directory": str(directory.relative_to(out)),
        "step": step,
        "world_size": world,
    }
    for name in names:
        durable_save(pointer, out / name)


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


def pinned(like):
    """Page-locked host tensor laid out as torch.empty_like(like) with exactly its bytes, in its
    own pages. pin_memory=True would pin each block rounded up to a power of two, for the whole
    run."""
    meta = torch.empty_like(like, device="meta")
    n = meta.untyped_storage().nbytes()
    if n == 0:
        return torch.empty_like(like, device="cpu")
    pages = mmap.mmap(-1, n, mmap.MAP_PRIVATE)
    storage = torch.frombuffer(pages, dtype=torch.uint8).untyped_storage()
    buffer = torch.empty(0, dtype=like.dtype).set_(
        storage, 0, meta.shape, meta.stride()
    )
    cudart = torch.cuda.cudart()
    torch.cuda.check_error(cudart.cudaHostRegister(storage.data_ptr(), n, 0))
    weakref.finalize(buffer, cudart.cudaHostUnregister, storage.data_ptr())
    return buffer


QUIET = {
    signal.SIGINT,
    signal.SIGTERM,
    signal.SIGUSR1,
}  # left to the main thread's handlers


class RowLog:
    """Rank 0's jsonl rows, appended under out and echoed to stdout by a thread, in order. On NFS
    an append can wait out a checkpoint write in flight, and every rank waits for rank 0. A failed
    write is raised by the next put() or by close()."""

    def __init__(self, out):
        self.out, self.rows, self.error = out, queue.SimpleQueue(), None
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        signal.pthread_sigmask(signal.SIG_BLOCK, QUIET)
        try:
            while row := self.rows.get():
                file, line = row
                if file:
                    with (self.out / file).open("a") as f:
                        f.write(line)
                print(line, end="", flush=True)
        except BaseException as e:
            self.error = e

    def check(self):
        if self.error:
            raise RuntimeError("log writer failed") from self.error

    def put(self, file, line):
        """Append line to out/file (echo only if file is None)."""
        self.check()
        self.rows.put((file, line))

    def close(self):
        self.rows.put(None)
        self.thread.join()
        self.check()


class AsyncSaver:
    """Checkpoints written from a thread, at most one save in flight. Each rank snapshots its state into host
    buffers reused across saves and a thread writes it; commit (rank 0: pointers, prune) runs on
    the main thread once poll() finds every rank's files durable. The gloo group is used only by
    poll(), which every rank calls at the same step boundaries."""

    def __init__(self, group):
        self.group, self.buffers, self.files, self.roots = group, {}, [], 0
        self.thread = self.commit = None
        self.pending = self.written = False

    @property
    def nbytes(self):
        return sum(b.untyped_storage().nbytes() for _, b in self.buffers.values())

    def snapshot(self, value):
        """cpu_copy into reused buffers; CUDA tensors are copied asynchronously into pinned
        memory, so the snapshot is complete after torch.cuda.synchronize()."""
        self.roots += 1
        return self._copy(value, (self.roots,))

    def _copy(self, value, path):
        if isinstance(value, torch.Tensor):
            meta = value.dtype, value.shape, value.stride(), value.device
            if path not in self.buffers or self.buffers[path][0] != meta:
                buffer = pinned(value) if value.is_cuda else torch.empty_like(value)
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
            signal.pthread_sigmask(signal.SIG_BLOCK, QUIET)
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
