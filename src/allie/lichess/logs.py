"""Logging that survives the shared filesystem failing under a running bot.

A writer thread takes the records off a queue, so no game ever waits on the disk. It appends to
the log file by path and reopens it after a failed write: a remounted filesystem leaves old file
handles failing (with EDQUOT or ESTALE) while the path works again. While the path fails, the
lines go to a node-local file, and the log notes the gap once it can be written again. Uncaught
exceptions in any thread are logged.
"""

import logging
import logging.handlers
import os
import queue
import sys
import tempfile
import threading

log = logging.getLogger(__name__)


class Resilient(logging.Handler):
    def __init__(self, path, fallback=None):
        super().__init__()
        self.path = path
        name = f"allie-bot-{os.getpid()}.log"
        self.fallback = fallback or os.path.join(tempfile.gettempdir(), name)
        self.stream, self.missed = None, 0

    def write(self, text):
        """Append text to the path, reopening it once if the open handle fails."""
        for _ in range(2):
            try:
                if self.stream is None:
                    self.stream = open(self.path, "a")
                self.stream.write(text)
                self.stream.flush()
                return True
            except OSError:
                self.close_stream()
        return False

    def close_stream(self):
        try:
            if self.stream is not None:
                self.stream.close()
        except OSError:
            pass
        self.stream = None

    def emit(self, record):
        try:
            text = self.format(record) + "\n"
        except Exception:  # noqa: BLE001 - a bad record must not stop the writer
            self.handleError(record)
            return
        if self.missed and self.write(
            f"{self.formatter.formatTime(record)} log: {self.missed} lines could not be "
            f"written here; they are in {self.fallback} on {os.uname().nodename}\n"
        ):
            self.missed = 0
        if self.write(text):
            return
        self.missed += 1
        try:
            with open(self.fallback, "a") as f:
                f.write(text)
        except OSError:
            pass

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

    def thread_hook(a):
        if a.exc_type is not SystemExit:
            name = a.thread.name if a.thread else "?"
            log.error(
                "thread %s died",
                name,
                exc_info=(a.exc_type, a.exc_value, a.exc_traceback),
            )

    def main_hook(*exc):
        log.error("uncaught exception", exc_info=exc)

    threading.excepthook = thread_hook
    sys.excepthook = main_hook
    return listener
