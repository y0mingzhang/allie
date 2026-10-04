"""Dry run of allie.lichess.chat on recorded games: the chat a game would have had.

record: replays each game's moves and clocks through the bot's model as a live game feeds it
(one Game, update + sync each ply) and saves the model's outputs. replay: feeds those to the
chat as the bot would, with scripted opponent messages, and prints the transcript.

usage: python analysis/lichess/chat_dryrun.py record --config bot.toml --out rec.pt GAME.pgn...
       python analysis/lichess/chat_dryrun.py replay rec.pt [--llm mock|cli|claude]
           [--cli-model claude-sonnet-5-5] [--stockfish PATH] [--prompts]
"""

import argparse
import json
import subprocess
import time
from types import SimpleNamespace

import chess
import chess.pgn
import torch
from allie.lichess import chat
from allie.lichess.chat import Chat, Chatter

BOT = "AllieTheChessBot"
# scripted opponent messages: (fraction of the game, text), one script per game in turn
SCRIPTS = [
    [(0.0, "hi gl hf"), (0.45, "why did you play that?"), (0.7, "are you a bot?")],
    [(0.05, "hello! how strong are you?"), (0.5, "what's the best move here?")],
    [(0.0, "hola, buena suerte"), (0.6, "uff, qué partida")],
    [(0.1, "!quiet"), (0.5, "hello?")],
    [(0.3, "you play like a human lol"), (0.6, "this bot is trash")],
]


def read(path):
    with open(path) as f:
        g = chess.pgn.read_game(f)
    moves, clocks, board = [], [], g.board()
    for node in g.mainline():
        moves.append(node.move.uci())
        clocks.append(node.clock())
        board.push(node.move)
    h = g.headers
    base, inc = map(int, h["TimeControl"].split("+"))
    white = h["White"] == BOT
    term = h.get("Termination", "Normal")
    status = (
        "mate" if board.is_checkmate() else "outoftime" if "Time" in term else "resign"
    )
    if h["Result"] == "1/2-1/2":
        status = "draw"
    winner = {"1-0": "white", "0-1": "black"}.get(h["Result"])
    return dict(id=h["GameId"], white=white, names=(h["White"], h["Black"]),
                ratings=(int(h["WhiteElo"]), int(h["BlackElo"])), base=base, inc=inc,
                moves=moves, clocks=clocks, status=status, winner=winner,
                rated=h["Event"].lower().startswith("rated"), speed=h["Event"].split()[1].lower())  # fmt: skip


def states(g):
    """Each ply's (wtime, btime), ms: each side's clock after its last move."""
    out, last = [], [g["base"] * 1000] * 2
    for k in range(len(g["moves"]) + 1):
        out.append(tuple(last))
        if k < len(g["moves"]) and g["clocks"][k] is not None:
            last[k % 2] = int(g["clocks"][k] * 1000)
    return out


def record(a):
    from allie.lichess.cli import model
    from allie.lichess.config import load
    from allie.lichess.engine import Engine, Game

    engine = Engine(model(load(a.config)))
    out = []
    for path in a.games:
        g = read(path)
        them = g["ratings"][g["white"]]  # the live bot mirrors the opponent's rating
        game = Game(engine, them, them, g["base"], g["inc"], g["speed"])
        logits = []
        for k, (w, b) in enumerate(states(g)):
            game.update(g["moves"][:k], w / 1000, b / 1000)
            logits.append(game.sync().half())
        out.append(g | dict(logits=torch.stack(logits), elo=(them, them)))
        print(g["id"], len(g["moves"]), "plies", flush=True)
    torch.save(out, a.out)
    engine.close()


class Cli:
    """Claude through the local `claude -p` (dry runs only): no tools, settings or MCP."""

    def __init__(self, name):
        self.name, self.calls = name, []

    def __call__(self, system, prompt):
        cmd = ["claude", "-p", "--model", self.name, "--tools", "", "--setting-sources", "",
               "--strict-mcp-config", "--no-session-persistence", "--output-format", "json",
               "--system-prompt", system]  # fmt: skip
        start = time.monotonic()
        r = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, check=True
        )
        out = json.loads(r.stdout)
        u = out.get("usage") or {}
        self.calls.append((time.monotonic() - start, out.get("duration_api_ms"), u))
        return out.get("result", "")


class Inline(Chatter):
    """The chat with its jobs run at once, in order (the bot runs them on the chat thread)."""

    def __init__(self, match, llm):
        super().__init__(match, llm)
        self.jobs = SimpleNamespace(put=lambda j: j and self.run(j))

    def serve(self):
        pass


class Replay:
    """One recorded game through the chat, as the bot's game thread would feed it."""

    def __init__(self, g, script, cfg, llm):
        self.g, self.script, self.llm = g, script, llm
        self.transcript, self.prompts, self.ply = [], [], 0
        white = g["white"]
        bot = SimpleNamespace(
            config=SimpleNamespace(chat=cfg), client=self, me=BOT.lower()
        )
        self.game = SimpleNamespace(moves=[], board=chess.Board(), logits=None, used=[],
                                    tokens=[], elo=g["elo"])  # fmt: skip
        self.opp = g["names"][white]
        match = SimpleNamespace(bot=bot, gid=g["id"], white=white, opponent=self.opp.lower(),
                                game=self.game, over=False)  # fmt: skip
        self.c = Inline(match, self.model)

    def chat(self, gid, text, room):
        self.transcript.append((self.ply, "Allie", text))

    def model(self, system, prompt):
        self.prompts.append((self.ply, prompt))
        return self.llm(system, prompt)

    def run(self):
        g, game, n = self.g, self.game, len(self.g["moves"])
        side = lambda j: {"name": g["names"][j], "rating": g["ratings"][j]}
        clock = {"initial": g["base"] * 1000, "increment": g["inc"] * 1000}
        full = {"type": "gameFull", "white": side(0), "black": side(1), "speed": g["speed"],
                "rated": g["rated"], "clock": clock}  # fmt: skip
        script = sorted((round(f * n), t) for f, t in self.script)
        clocks = states(g)
        mate = g["status"] == "mate"  # else the end follows a state with all the moves
        events = [(k, "started") for k in range(n + (not mate))] + [(n, g["status"])]
        for k, status in events:
            self.ply = k
            if (
                status == "started"
            ):  # as the bot: the moves, then the model's output there
                game.moves, game.logits = g["moves"][:k], g["logits"][k].float()
                game.board = chat.board_at(g["moves"], k)
            state = {"type": "gameState", "moves": " ".join(g["moves"][:k]),
                     "wtime": clocks[k][0], "btime": clocks[k][1], "status": status,
                     "winner": g["winner"]}  # fmt: skip
            self.c.seen(full | {"state": state} if k == 0 else state)
            for text in [t for j, t in script if j == k and status == "started"]:
                self.transcript.append((k, self.opp, text))
                line = {
                    "type": "chatLine",
                    "username": self.opp,
                    "room": "player",
                    "text": text,
                }
                self.c.heard(line)

    def show(self, prompts):
        g, n = self.g, len(self.g["moves"])
        result = {"white": "1-0", "black": "0-1"}.get(g["winner"], "1/2")
        us = "white" if g["white"] else "black"
        print(f"\n=== {g['id']}: {g['names'][0]} ({g['ratings'][0]}) - {g['names'][1]} "
              f"({g['ratings'][1]}), {g['base'] // 60}+{g['inc']}, {result} by {g['status']}, "
              f"{n} plies; Allie is {us} ===")  # fmt: skip
        for k, who, text in self.transcript:
            where = chat.san(g["moves"], k - 1) if k else "start"
            print(f"  [{where:>12}] {who}: {text}")
        for k, p in self.prompts if prompts else ():
            print(f"\n--- prompt at ply {k} ---\n{p}")


def replay(a):
    cfg = Chat(enabled=True, gap=0, stockfish=a.stockfish or "", llm=a.llm)
    llm = Cli(a.cli_model) if a.llm == "cli" else chat.model(cfg)
    for i, g in enumerate(torch.load(a.recording, weights_only=False)):
        r = Replay(g, SCRIPTS[i % len(SCRIPTS)], cfg, llm)
        r.run()
        r.show(a.prompts)
    if calls := getattr(llm, "calls", None):
        wall = sorted(c[0] for c in calls)
        api = sorted(c[1] or 0 for c in calls)
        tokens = {k: sum(c[2].get(k) or 0 for c in calls) for k in calls[0][2]
                  if isinstance(calls[0][2][k], int)}  # fmt: skip
        print(f"\n{len(calls)} CLI calls: median {wall[len(wall) // 2]:.1f} s wall, "
              f"{api[len(api) // 2] / 1000:.1f} s API; tokens {tokens}")  # fmt: skip
    if usage := getattr(llm, "usage", None):
        print("\nAPI usage:", usage)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("record")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("games", nargs="+")
    s = sub.add_parser("replay")
    s.add_argument("recording")
    s.add_argument("--llm", choices=["mock", "cli", "claude"], default="mock")
    s.add_argument("--cli-model", default="claude-sonnet-5-5")
    s.add_argument("--stockfish")
    s.add_argument("--prompts", action="store_true", help="print every prompt")
    a = p.parse_args()
    record(a) if a.command == "record" else replay(a)


if __name__ == "__main__":
    main()
