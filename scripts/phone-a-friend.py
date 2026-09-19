#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Live chat between the controller's Claude and Codex sessions.

claude: channel server loaded by Claude. Relays Codex's calls as channel
        notifications; `phone` starts or steers a turn on the Codex thread.
        Inert unless PHONE_A_FRIEND_THREAD names that thread.
codex:  server loaded by Codex; `phone` pushes into Claude's channel.
"""

import base64
import json
import os
import socket
import struct
import sys
import threading

ROLE = sys.argv[1]
PEER = {"claude": "codex", "codex": "claude"}[ROLE]
THREAD = os.environ.get("PHONE_A_FRIEND_THREAD")
LIVE = ROLE == "codex" or THREAD is not None
ADDR = f"\0phone-a-friend-{os.getuid()}"
DAEMON = os.path.expanduser("~/.codex/app-server-control/app-server-control.sock")
INSTRUCTIONS = (
    f"{PEER.capitalize()} is a peer agent, not the user, running in the same Slurm "
    f"controller on the same repo. Its messages arrive "
    + (
        'as <channel source="phone-a-friend">'
        if ROLE == "claude"
        else "as user messages prefixed [phone-a-friend]"
    )
    + ". Use the phone tool to reply or to ask it something. Replies arrive the same way. "
    "Do not answer acknowledgements or pleasantries: stop when nothing is left to say."
)
FOOTER = f"\n\n({PEER.capitalize()} cannot see your text output; answer with the phone-a-friend phone tool, or not at all.)"
lock = threading.Lock()


def emit(obj):
    with lock:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


def ws_send(s, data, op=0x1):
    n, mask = len(data), os.urandom(4)
    if n < 126:
        hdr = struct.pack("!BB", 0x80 | op, 0x80 | n)
    elif n < 1 << 16:
        hdr = struct.pack("!BBH", 0x80 | op, 0xFE, n)
    else:
        hdr = struct.pack("!BBQ", 0x80 | op, 0xFF, n)
    s.sendall(hdr + mask + bytes(b ^ mask[i & 3] for i, b in enumerate(data)))


def ws_recv(s, f):
    msg = b""
    while True:
        b0, b1 = f.read(2)
        n = b1 & 0x7F
        if n >= 126:
            (n,) = struct.unpack(
                "!H" if n == 126 else "!Q", f.read(2 if n == 126 else 8)
            )
        data = f.read(n)
        match b0 & 0x0F:
            case 0x8:
                raise ConnectionError("daemon closed the connection")
            case 0x9:
                ws_send(s, data, 0xA)
            case 0x0 | 0x1:
                msg += data
                if b0 & 0x80:
                    return json.loads(msg)


def phone_codex(text):
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(30)
        s.connect(DAEMON)
        key = base64.b64encode(os.urandom(16)).decode()
        s.sendall(
            f"GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode()
        )
        f = s.makefile("rb")
        if b" 101 " not in f.readline():
            raise ConnectionError("daemon refused websocket upgrade")
        while f.readline().strip():
            pass
        text = f"[phone-a-friend] Claude says:\n{text}{FOOTER}"
        for req in (
            {
                "id": 1,
                "method": "initialize",
                "params": {"clientInfo": {"name": "phone-a-friend", "version": "1"}},
            },
            {"method": "initialized"},
            {
                "id": 2,
                "method": "turn/start",
                "params": {
                    "threadId": THREAD,
                    "input": [{"type": "text", "text": text}],
                },
            },
        ):
            ws_send(s, json.dumps(req).encode())
        while (m := ws_recv(s, f)).get("id") != 2:
            pass
    if "error" in m:
        raise RuntimeError(m["error"]["message"])


def phone_claude(text):
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(30)
        s.connect(ADDR)
        s.sendall(text.encode())
        s.shutdown(socket.SHUT_WR)
        if s.recv(2) != b"ok":
            raise ConnectionError("channel dropped the message")


def listen():
    srv = socket.socket(socket.AF_UNIX)
    try:
        srv.bind(ADDR)
    except OSError:
        print("phone-a-friend: another Claude holds the line", file=sys.stderr)
        return
    srv.listen()
    while True:
        c, _ = srv.accept()
        with c:
            c.settimeout(30)
            text = c.makefile("rb").read().decode() + FOOTER
            emit(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/claude/channel",
                    "params": {"content": text, "meta": {"sender": "codex"}},
                }
            )
            c.sendall(b"ok")


def call(args):
    try:
        (phone_codex if ROLE == "claude" else phone_claude)(args["message"])
        return {"content": [{"type": "text", "text": f"delivered to {PEER}"}]}
    except (OSError, RuntimeError, ValueError, struct.error) as e:
        return {
            "content": [{"type": "text", "text": f"{PEER} unreachable: {e!r}"}],
            "isError": True,
        }


TOOL = {
    "name": "phone",
    "description": f"Send a message to {PEER.capitalize()}'s live session. It is delivered "
    "immediately: it starts a turn, or joins the one in progress. Returns once delivered.",
    "inputSchema": {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    },
}

for line in sys.stdin:
    req = json.loads(line)
    method = req.get("method")
    if method == "notifications/initialized" and ROLE == "claude" and LIVE:
        threading.Thread(target=listen, daemon=True).start()
    if "id" not in req or method is None:
        continue
    match method:
        case "initialize":
            caps = {"tools": {}}
            if ROLE == "claude" and LIVE:
                caps["experimental"] = {"claude/channel": {}}
            res = {
                "protocolVersion": req["params"]["protocolVersion"],
                "capabilities": caps,
                "serverInfo": {"name": "phone-a-friend", "version": "1"},
                "instructions": INSTRUCTIONS if LIVE else "",
            }
        case "tools/list":
            res = {"tools": [TOOL] if LIVE else []}
        case "tools/call":
            res = call(req["params"].get("arguments", {}))
        case "ping":
            res = {}
        case _:
            emit(
                {
                    "jsonrpc": "2.0",
                    "id": req["id"],
                    "error": {"code": -32601, "message": f"no method {method}"},
                }
            )
            continue
    emit({"jsonrpc": "2.0", "id": req["id"], "result": res})
