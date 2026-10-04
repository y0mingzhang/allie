import threading

from allie.search import native


def test_threads_load_a_built_module_once_without_the_file_lock(monkeypatch):
    """Games' first searches load the native modules at once. A built module must not take the
    cache's flock: on the NFS home a contended flock waits out the client's 30 s retry backoff (live
    load test 2026-10-04: first searches stalled 30, 60 and 90 s), and the threads share one load."""
    native.load(), native.load("value")
    locks = []
    monkeypatch.setattr(native.fcntl, "flock", lambda *a: locks.append(a))
    native._load.cache_clear()
    start, got = threading.Barrier(8), []

    def go(kind):
        start.wait()
        got.append((kind, native.load(kind)))

    threads = [threading.Thread(target=go, args=("tree" if i % 2 else "value",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert locks == []
    assert len(got) == 8 and len({(k, id(m)) for k, m in got}) == 2
