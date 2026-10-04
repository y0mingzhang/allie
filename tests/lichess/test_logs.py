import builtins
import logging
import sys
import threading

import pytest

from allie.lichess import logs


@pytest.fixture
def restore(monkeypatch):
    """setup() replaces the root handlers and the exception hooks: put them back."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def test_log_survives_a_filesystem_outage(tmp_path, monkeypatch, restore):
    path, fallback = tmp_path / "bot.log", tmp_path / "local.log"
    h = logs.Resilient(str(path), str(fallback))
    h.setFormatter(logging.Formatter("%(message)s"))
    record = lambda msg: logging.LogRecord("x", logging.INFO, "f", 1, msg, None, None)
    h.handle(record("before"))

    class Stale:  # an open handle left behind by a remount: every write fails
        def write(self, text):
            raise OSError(122, "Disk quota exceeded")

        flush = close = lambda self: None

    h.stream = Stale()
    h.handle(record("reopened"))  # the path itself works: reopened, nothing lost
    real = builtins.open

    def down(file, *args, **kwargs):
        if str(file) == str(path):
            raise OSError(116, "Stale file handle")
        return real(file, *args, **kwargs)

    h.close_stream()
    monkeypatch.setattr(builtins, "open", down)
    h.handle(record("during 1"))
    h.handle(record("during 2"))
    monkeypatch.setattr(builtins, "open", real)
    h.handle(record("after"))
    h.close()
    lines = path.read_text().splitlines()
    assert lines[:2] == ["before", "reopened"] and lines[-1] == "after"
    assert "2 lines could not be written here" in lines[2] and str(fallback) in lines[2]
    assert fallback.read_text().splitlines() == ["during 1", "during 2"]


def test_uncaught_thread_errors_are_logged(tmp_path, restore):
    path = tmp_path / "bot.log"
    listener = logs.setup(str(path))
    t = threading.Thread(target=lambda: 1 / 0, name="game-x")
    t.start()
    t.join()
    logging.getLogger("allie").info("still logging")
    listener.stop()
    text = path.read_text()
    assert (
        "thread game-x died" in text
        and "ZeroDivisionError" in text
        and "still logging" in text
    )


def test_bad_records_do_not_stop_the_log(tmp_path, restore):
    path = tmp_path / "bot.log"
    listener = logs.setup(str(path))
    log = logging.getLogger("allie")
    log.info("chat from the opponent: %s", "\ud83d")  # a lone surrogate, as json.loads can give
    log.info("bad %d", "format")
    log.info("still logging")
    listener.stop()
    text = path.read_text()
    assert "\\ud83d" in text and "still logging" in text


def test_chat_reader_stops_when_abandoned():
    from allie.lichess.chat import END, split

    fed, abandon, gate = [], threading.Event(), threading.Event()

    class Chat:
        gid, done = "g", False

        def put(self, e):
            fed.append(e)

    def events():
        yield {"type": "gameState", "n": 1}
        gate.wait(5)
        yield {"type": "gameState", "n": 2}
        yield {"type": "gameState", "n": 3}

    stream = split(events(), Chat(), abandon)
    assert next(stream)["n"] == 1
    abandon.set()  # the game thread resumes on a new stream
    gate.set()
    for _ in range(500):
        if fed and fed[-1] is END:
            break
        threading.Event().wait(0.01)
    assert fed[-1] is END and [e["n"] for e in fed if e is not END] == [1]
