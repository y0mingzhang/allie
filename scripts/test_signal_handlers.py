"""CPU regression for Triton/LLVM replacing the trainer's native signal handlers.

Run in the pinned training environment: python scripts/test_signal_handlers.py.
Fresh subprocesses are essential: LLVM registers handlers once per process.
No CUDA context, training, data, or checkpoint is needed.
"""

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path


def pipeline():
    from triton._C.libtriton import ir

    ctx = ir.context()
    ir.load_dialects(ctx)
    ir.pass_manager(ctx).run(ir.builder(ctx).create_module(), "signal-regression")


def child(fixed):
    import random

    import numpy as np
    import torch
    import triton
    from modded_train import init_triton_signals

    signals = (signal.SIGUSR1, signal.SIGTERM)
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    assert not (set(signals) & mask), "Test child inherited blocked stop signals"
    assert not torch.cuda.is_initialized()
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    if fixed:
        init_triton_signals()
    assert random.getstate() == rng[0]
    now = np.random.get_state()
    assert now[0] == rng[1][0] and np.array_equal(now[1], rng[1][1])
    assert now[2:] == rng[1][2:]
    assert torch.equal(torch.get_rng_state(), rng[2])
    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == mask
    seen = []

    def stop_handler(signum, _frame):
        seen.append(signum)

    for sig in signals:
        signal.signal(sig, stop_handler)
    pipeline()
    # getsignal alone is insufficient: CPython still remembers the callback
    # when the underlying sigaction has been replaced by LLVM.
    assert all(signal.getsignal(sig) is stop_handler for sig in signals)
    os.kill(os.getpid(), signal.SIGUSR1)
    if not fixed:
        assert seen == [], f"Expected the pinned Triton 3.6 bug, got {seen}"
        print(
            json.dumps(dict(mode="unfixed", triton=triton.__version__, swallowed=True))
        )
        return
    assert seen == [signal.SIGUSR1], seen
    os.kill(os.getpid(), signal.SIGTERM)
    assert seen == list(signals), seen

    # Later compilation on the main thread and a worker thread must not undo
    # the repair. Process-directed signals still execute Python on main.
    for worker in (False, True, False, True):
        errors = []

        def run():
            try:
                pipeline()
            except BaseException as exc:
                errors.append(exc)

        if worker:
            thread = threading.Thread(target=run)
            thread.start()
            thread.join()
            assert not errors, errors
        else:
            pipeline()
        for sig in signals:
            os.kill(os.getpid(), sig)
    assert seen == list(signals) * 5, seen
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(mode="fixed", triton=triton.__version__, received=seen)))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2] == "fixed")
        return
    assert len(sys.argv) == 1, sys.argv
    for mode in ("unfixed", "fixed"):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", mode],
            text=True,
            capture_output=True,
            timeout=120,
        )
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        assert proc.returncode == 0, (mode, proc.returncode)
    print("PASS: LLVM prewarm preserves USR1/TERM through later MLIR pipelines")


if __name__ == "__main__":
    main()
