"""modelexp.run's watchdog survives a transient non-ENOENT stat error of its watch file
(NFS can return errno 512): the child is not reaped and run() does not raise. CPU only, ~8 s.

    .venv/bin/python tests/checks/modelexp_run.py
"""

import sys
import tempfile
from pathlib import Path

from allie.experiments import modelexp as mx


class Flaky(type(Path())):
    calls = 0

    def stat(self, **kw):
        Flaky.calls += 1
        if Flaky.calls in (3, 4):  # while the child runs
            raise OSError(512, "Unknown error 512")
        return super().stat(**kw)


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    watch, done = Flaky(tmp / "train.jsonl"), tmp / "done"
    watch.write_text("{}\n")
    child = f"import time; time.sleep(7); open({str(done)!r}, 'w')"
    mx.run([sys.executable, "-c", child], None, tmp / "log", watch=watch)
    assert Flaky.calls > 4 and done.exists(), (Flaky.calls, done.exists())
print("ok")
