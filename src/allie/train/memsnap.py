"""Opt-in allocator history and failure-time dumps for MEMSNAP diagnostics.

Install before model/optimizer allocation. The allocator callback captures the
first OOM before Python unwinds; it can also be a caught/autotuner OOM, so the
exception fallback is separately labelled. No allocator settings are changed.
"""
import json
import os
from pathlib import Path
import sys

import torch

_active = None


class MemoryTrace:
    def __init__(self, path, rank):
        self.path, self.rank = Path(path), rank
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.oom_saved = False

    def filename(self, reason):
        return self.path.with_name(f'{self.path.stem}-{reason}-rank{self.rank}.pickle')

    def dump(self, reason, metadata):
        path = self.filename(reason)
        temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
        torch.cuda.memory._dump_snapshot(str(temporary))
        os.replace(temporary, path)
        report = dict(snapshot=str(path), rank=self.rank, reason=reason, **metadata)
        path.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'memory_snapshot': report}), file=sys.stderr, flush=True)

    def oom(self, device, alloc, device_capacity, device_free):
        if self.oom_saved:
            return
        self.oom_saved = True
        try:
            self.dump('oom', dict(device=device, requested_bytes=alloc,
                                  device_capacity_or_limit_bytes=device_capacity,
                                  device_free_bytes=device_free,
                                  note='first allocator OOM; may be caught by an autotuner'))
        except BaseException as exc:
            print(f'Failed to write allocator OOM snapshot: {exc!r}', file=sys.stderr, flush=True)


def start(path, rank):
    global _active
    if not path:
        return
    assert _active is None, 'only one memory recorder per process'
    _active = MemoryTrace(path, rank)
    torch.cuda.memory._record_memory_history(max_entries=500000)
    torch._C._cuda_attach_out_of_memory_observer(_active.oom)


def dump_on_error():
    if _active is None:
        return
    exc = sys.exc_info()[1]
    try:
        # Triton/driver failures can bypass the caching allocator observer.
        # History still captures the peak, even if some tensors were unwound.
        _active.dump('exception', dict(exception=repr(exc), allocator_oom_saved=_active.oom_saved))
    except BaseException as error:
        print(f'Failed to write exception snapshot: {error!r}', file=sys.stderr, flush=True)
