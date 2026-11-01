"""data.fetch hands curl the Hugging Face token on stdin, never in argv, which ps shows to every user."""

import hashlib
import http.server
import runpy
import subprocess
import sys
import threading

import pytest

from allie.data import fetch


@pytest.mark.parametrize("bearer", ["Bearer hf_dummy", None])
def test_token_reaches_the_server_not_argv(bearer, tmp_path, monkeypatch):
    seen, argv, run = [], [], subprocess.run

    class Server(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers["Authorization"])
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *_):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(fetch, "BEARER", bearer)
    monkeypatch.setattr(fetch, "URL", f"http://127.0.0.1:{server.server_port}/{{}}")
    monkeypatch.setattr(fetch.subprocess, "run", lambda cmd, **kw: (argv.extend(cmd), run(cmd, **kw))[1])
    entry = dict(path="data/a.parquet", size=2, lfs=dict(oid=hashlib.sha256(b"ok").hexdigest()))
    assert fetch.fetch(entry, tmp_path)
    server.shutdown()
    assert seen == [bearer] and argv and not any("hf_dummy" in a for a in argv)


def test_help(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fetch", "--help"])
    monkeypatch.delitem(sys.modules, "allie.data.fetch")
    with pytest.raises(SystemExit) as e:
        runpy.run_module("allie.data.fetch", run_name="__main__")
    assert e.value.code == 0 and "YYYY-MM" in capsys.readouterr().out
