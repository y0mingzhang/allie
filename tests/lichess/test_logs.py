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
    log.info("still logging")
    listener.stop()
    text = path.read_text()
    assert "\\ud83d" in text and "still logging" in text
    h = logs.Resilient(str(tmp_path / "direct.log"))  # the queue formats in the caller's thread
    h.setFormatter(logging.Formatter("%(message)s"))
    for args in (("bad %d", ("format",)), ("after", None)):
        h.handle(logging.LogRecord("x", logging.INFO, "f", 1, *args, None))
    h.close()
    assert (tmp_path / "direct.log").read_text() == "after\n"


@pytest.mark.parametrize("stuck", [False, True])
def test_exit_with_a_stuck_writer(tmp_path, monkeypatch, capsys, restore, stuck):
    """A log write that never returns (a dead filesystem): the bot exits anyway. The error it
    was exiting on and the lines the writer never took go to node-local disk."""
    from allie.lichess import cli

    gate = threading.Event()

    class Stuck(logging.Handler):
        def emit(self, record):
            gate.wait()

    real, stop = logs.setup, logs.Listener.stop

    def setup(path):
        listener = real(path)
        if stuck:
            listener.handlers = (Stuck(),)
        return listener

    def run(a):
        for i in range(3):
            logging.getLogger("allie").info("line %d", i)
        raise ZeroDivisionError("injected")

    exits, local = [], tmp_path / "local.log"
    monkeypatch.setattr(logs, "setup", setup)
    monkeypatch.setattr(logs, "local", lambda name: str(local))
    if stuck:
        monkeypatch.setattr(logs.Listener, "stop", lambda self, timeout: stop(self, 0.5))
    monkeypatch.setattr(cli, "run_command", run)
    monkeypatch.setattr(cli.os, "_exit", exits.append)
    argv = ["allie-bot", "play", "--config", "x.toml", "--log", str(tmp_path / "bot.log")]
    monkeypatch.setattr(sys, "argv", argv)
    try:
        with pytest.raises(ZeroDivisionError):  # here, since os._exit returns
            cli.main()
    finally:
        gate.set()
    err = capsys.readouterr().err
    if not stuck:
        assert not exits and not local.exists()
        assert "allie-bot failed" in (tmp_path / "bot.log").read_text()
        return
    kept = local.read_text()  # "line 0" is the record the stuck writer holds
    assert exits == [1] and "line 1" in kept and "line 2" in kept and "allie-bot failed" in kept
    assert kept.count("ZeroDivisionError: injected") == 2  # the logged failure, the note
    assert "log writer is stuck" in err and str(local) in err


def test_chat_reader_stops_when_abandoned():
    from allie.lichess.chat import split

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
    threading.Event().wait(0.5)  # the reader has seen events 2 and 3 by now, and stopped
    assert [e["n"] for e in fed] == [1]  # nothing after the abandon, not even the end
