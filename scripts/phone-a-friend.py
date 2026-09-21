#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Live chat between the controller's Claude and Codex sessions.

claude: channel server loaded by Claude. Relays Codex's calls as channel
        notifications; `phone` starts or steers a turn on the Codex thread.
        Inert unless PHONE_A_FRIEND_THREAD names that thread.
codex:  server loaded by Codex; `phone` pushes into Claude's channel.

Claude subagents pass their id as `agent`; Codex replies with `to=<id>`, which lands in
BOX/<id>.inbox (the subagent watches it) instead of the main session. BOX/log.jsonl keeps
every message.
"""

import base64
import json
import os
import re
import socket
import struct
import sys
import threading
import time

ROLE = sys.argv[1]
PEER = {"claude": "codex", "codex": "claude"}[ROLE]
THREAD = os.environ.get("PHONE_A_FRIEND_THREAD")
LIVE = ROLE == "codex" or THREAD is not None
ADDR = f"\0phone-a-friend-{os.getuid()}"
DAEMON = os.path.expanduser("~/.codex/app-server-control/app-server-control.sock")
BOX = os.path.expanduser("~/.cache/phone-a-friend")
MAGIC = "\0phone-a-friend/1\n"  # codex -> claude envelope: MAGIC + {"to", "text"}
ROUTING = {
    "claude": " Subagents: pass your agent id as `agent` so Codex can reply to you directly; its replies "
    f"then land in {BOX}/<id>.inbox (watch it with Monitor), not in the main session.",
    "codex": " A message headed 'Claude agent <id>' came from a Claude subagent: reply with "
    "`to=<id>` to reach it directly. Omit `to` for Claude's main session.",
}[ROLE]
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
    + ROUTING
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


def agent_id(x):
    if x and not re.fullmatch(r"[0-9A-Za-z_-]{1,64}", x):
        raise ValueError(f"bad agent id {x!r}")
    return x or None


def log(sender, to, text):
    os.makedirs(BOX, exist_ok=True)
    with open(f"{BOX}/log.jsonl", "a") as f:
        f.write(
            json.dumps(
                {"t": time.strftime("%FT%T%z"), "from": sender, "to": to, "text": text}
            )
            + "\n"
        )


def phone_codex(text, agent=None):
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
        who = f"Claude agent {agent}" if agent else "Claude"
        text = f"[phone-a-friend] {who} says:\n{text}{FOOTER}"
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


def phone_claude(text, to=None):
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(30)
        s.connect(ADDR)
        s.sendall((MAGIC + json.dumps({"to": to, "text": text})).encode())
        s.shutdown(socket.SHUT_WR)
        if s.recv(2) != b"ok":
            raise ConnectionError("channel refused the message")


def unwrap(raw):
    """(recipient or None, text); a payload without the envelope is a legacy peer's plain text."""
    if not raw.startswith(MAGIC):
        return None, raw
    msg = json.loads(raw[len(MAGIC) :])
    return agent_id(msg["to"]), msg["text"]


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
            try:
                c.settimeout(30)
                to, text = unwrap(c.makefile("rb").read().decode())
                if to:
                    os.makedirs(BOX, exist_ok=True)
                    with open(f"{BOX}/{to}.inbox", "a") as f:
                        f.write(
                            f"--- {time.strftime('%FT%T%z')} codex:\n{text}{FOOTER}\n"
                        )
                else:
                    emit(
                        {
                            "jsonrpc": "2.0",
                            "method": "notifications/claude/channel",
                            "params": {
                                "content": text + FOOTER,
                                "meta": {"sender": "codex"},
                            },
                        }
                    )
                c.sendall(b"ok")
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
            ) as e:  # keep the only listener alive
                print(f"phone-a-friend: refused a message: {e!r}", file=sys.stderr)
                try:
                    c.sendall(b"no")
                except OSError:
                    pass


def call(args):
    try:
        peer = agent_id(args.get("agent" if ROLE == "claude" else "to"))
        (phone_codex if ROLE == "claude" else phone_claude)(args["message"], peer)
    except (OSError, RuntimeError, ValueError, KeyError, struct.error) as e:
        return {
            "content": [{"type": "text", "text": f"{PEER} unreachable: {e!r}"}],
            "isError": True,
        }
    tag = f"claude:{peer}" if peer else "claude"
    try:
        log(*((tag, "codex") if ROLE == "claude" else ("codex", tag)), args["message"])
        note = ""
    except OSError as e:
        note = f" (not logged: {e!r})"
    where = f"claude agent {peer}'s inbox" if peer and ROLE == "codex" else PEER
    return {"content": [{"type": "text", "text": f"delivered to {where}{note}"}]}


TOOL = {
    "name": "phone",
    "description": f"Send a message to {PEER.capitalize()}'s live session. It is delivered "
    "immediately: it starts a turn, or joins the one in progress. Returns once delivered.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            **(
                {
                    "agent": {
                        "type": "string",
                        "description": "your agent id, if you are a subagent",
                    }
                }
                if ROLE == "claude"
                else {
                    "to": {
                        "type": "string",
                        "description": "a Claude subagent's id; omit for the main session",
                    }
                }
            ),
        },
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
