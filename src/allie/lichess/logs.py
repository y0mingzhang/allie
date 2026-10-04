"""Logging that survives the shared filesystem failing under a running bot.

A writer thread takes the records off a queue, so no game ever waits on the disk. It appends to
the log file by path and reopens it after a failed write: a full quota fails writes (EDQUOT)
until space is freed, and a remounted filesystem leaves old handles failing while the path works
again. While the path fails, the lines go to a node-local file, and the log notes the gap once it
can be written again. Uncaught exceptions in any thread are logged, and also printed to stderr.
"""

import logging
import logging.handlers
import os
import queue
import sys
import tempfile
import threading

log = logging.getLogger(__name__)


def local(name):
    """A path for name on node-local disk that outlives the job (/scratch/$USER), else in the
    temporary directory."""
    for d in (
        os.path.join("/scratch", os.environ.get("USER", "allie")),
        tempfile.gettempdir(),
    ):
        if os.path.isdir(d) and os.access(d, os.W_OK):
            return os.path.join(d, name)
    return os.path.join(tempfile.gettempdir(), name)


class Resilient(logging.Handler):
    def __init__(self, path, fallback=None):
        super().__init__()
        self.path = path
        self.fallback = fallback or local(f"allie-bot-{os.getpid()}.log")
        self.stream, self.missed = None, 0

    def open(self, path):
        return open(path, "a", encoding="utf-8", errors="backslashreplace")

    def write(self, text):
        """Append text to the path, reopening it once if the open handle fails."""
        for _ in range(2):
            try:
                if self.stream is None:
                    self.stream = self.open(self.path)
                self.stream.write(text)
                self.stream.flush()
                return True
            except Exception:  # noqa: BLE001 - OSError from the disk, anything from a bad handle
                self.close_stream()
        return False

    def close_stream(self):
        try:
            if self.stream is not None:
                self.stream.close()
        except Exception:  # noqa: BLE001
            pass
        self.stream = None

    def emit(self, record):
        """Never raises: an exception here would end the writer thread, and the log with it."""
        try:
            text = self.format(record) + "\n"
            if self.missed and self.write(
                f"{self.formatter.formatTime(record)} log: {self.missed} lines could not be "
                f"written here; they are in {self.fallback} on {os.uname().nodename}\n"
            ):
                self.missed = 0
            if self.write(text):
                return
            self.missed += 1
            with self.open(self.fallback) as f:
                f.write(text)
        except Exception:  # noqa: BLE001
            self.handleError(record)

    def close(self):
        self.close_stream()
        super().close()


def setup(path=None, level=logging.INFO, fmt="%(asctime)s %(message)s"):
    """Log through a queue to path (or stderr), and log every thread's uncaught exception.
    Returns the queue listener (stop() flushes it)."""
    handler = Resilient(path) if path else logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))
    q = queue.SimpleQueue()
    listener = logging.handlers.QueueListener(q, handler)
    listener.start()
    root = logging.getLogger()
    root.handlers[:] = [logging.handlers.QueueHandler(q)]
    root.setLevel(level)

    def thread_hook(
        a,
    ):  # also on stderr: the log's own writer may be the thread that died
        if a.exc_type is not SystemExit:
            exc = (a.exc_type, a.exc_value, a.exc_traceback)
            log.error(
                "thread %s died", a.thread.name if a.thread else "?", exc_info=exc
            )
            threading.__excepthook__(a)

    def main_hook(*exc):  # also on stderr: on the way out the writer may have stopped
        log.error("uncaught exception", exc_info=exc)
        sys.__excepthook__(*exc)

    threading.excepthook = thread_hook
    sys.excepthook = main_hook
    return listener
