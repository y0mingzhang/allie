"""Dry run of allie.lichess.chat on recorded games: the chat a game would have had.

Each game's moves and clocks go to a real Chatter (the bot's model, the chat's model) as the
game stream would send them, with a script of opponent messages, a draw offer and a
resignation, or (--log, a live game's PGN) the opponent's real lines from the bot's log at
the plies they came. Prints each transcript (with the model's silences), every reply's
latency and cost, and the spend.

usage: python analysis/lichess/chat_dryrun.py --config bot.toml [--llm claude|mock]
           [--ledger PATH] [--seed N] [--prompts] [--log BOT.log]
           GAME.pgn:casual|rated:PLIES[:chatty|identity|quiet|silent]... | --log BOT.log GAME.pgn...
"""

import argparse
import bisect
import logging
import re
import tempfile
from datetime import datetime
from types import SimpleNamespace

import chess.pgn

from allie.lichess import chat
from allie.lichess.chat import Chatter

BOT = "AllieTheChessBot"
# the opponent's messages: (fraction of the game, message; "draw" offers a draw there),
# then the ones after the game
SCRIPTS = {
    "chatty": (
        [(0.0, "hi! gl"), (0.25, "wdyt about my next move?"), (0.45, "blunder?"),
         (0.6, "draw"), (0.8, "you play like a human lol")],
        ["gg, how did I play?", "what was the turning point?"],
    ),
    "identity": (
        [(0.0, "hi gl"), (0.2, "are you an engine?"), (0.35, "was that the best move?"),
         (0.5, "whos winning rn?"), (0.6, "draw"), (0.7, "can you play faster"),
         (0.8, "idm")],
        ["gg", "how did i play?"],
    ),
    "quiet": ([(0.6, "draw")], ["gg"]),  # never chats; gg after the game
    "silent": ([], []),
}  # fmt: skip


def read(path, plies):
    with open(path) as f:
        g = chess.pgn.read_game(f)
    h, moves, clocks = g.headers, [], []
    for node in list(g.mainline())[:plies]:
        moves.append(node.move.uci())
        clocks.append(node.clock())
    base, inc = map(int, h["TimeControl"].split("+"))
    return h, moves, clocks, base, inc


def from_log(log, h):
    """Game h's opponent lines from a bot log, as a script: each burst (lines 6 s apart at
    most) at the ply it came, by the bot's decision times (a move of theirs came as the bot
    began its next decision; a move of ours, its think time after it began); their draw
    offers (not one they had accepted); the lines after the game; and how it ended."""
    gid, white = h["GameId"], h["White"] == BOT
    opp, them = h["Black" if white else "White"].lower(), "Black" if white else "White"
    begun, made, lines, end = {}, {}, [], None
    with open(log) as f:
        for x in f:
            if f" game {gid} " not in x:
                continue
            t = datetime.strptime(x[:23], "%Y-%m-%d %H:%M:%S,%f").timestamp()
            if m := re.search(r" ply (\d+): \S+ .*? (\d+) ms(?: think ([\d.]+) s)?", x):
                k, ms, think = int(m[1]), int(m[2]) / 1000, float(m[3] or 0)
                begun.setdefault(k, t - ms)
                made.setdefault(k, t - ms + max(think, ms))
            elif m := re.search(r" chat \(player\) ([^:]+): (.*)", x):
                lines.append((t, m[1].lower(), m[2].strip()))
            elif m := re.search(r" over: (\w+), winner (\w+)", x):
                end = (t, m[1], None if m[2] == "None" else m[2])
    n = max([*begun, -1]) + 2
    # times[k]: when the k-th move was made (ours: sent; theirs: our next decision began)
    times = [made.get(k - 1) if (k - 1) % 2 != white else begun.get(k) for k in range(1, n)]
    for k in range(1, len(times)):
        times[k] = times[k] or times[k - 1]
    times = [x or 0.0 for x in times]
    during, after, last = [], [], -1.0
    for i, (t, who, text) in enumerate(lines):
        if who == "lichess" and text == f"{them} offers draw":
            nxt = next((y for _, w, y in lines[i + 1 :] if w == "lichess"), "")
            if nxt != "Draw offer accepted":
                during.append((bisect.bisect_right(times, t), "draw"))
        elif who == opp and end and t > end[0]:
            after.append(text)
        elif who == opp:
            k = bisect.bisect_right(times, t)
            if during and t - last <= 6 and isinstance(during[-1][1], list) and during[-1][0] == k:
                during[-1][1].append(text)
            else:
                during.append((k, [text]))
            last = t
    return (during, after), end and end[1:]


def replay(c, posts, path, rated, plies, script, end=None):
    h, moves, clocks, base, inc = read(path, plies)
    white = h["White"] == BOT
    opp = h["Black" if white else "White"]
    player = lambda n, r: {"id": n.lower(), "name": n, "rating": int(r)}
    clock = {"initial": base * 1000, "increment": inc * 1000}
    times = [base * 1000] * 2
    state = lambda k, **kw: {"type": "gameState", "moves": " ".join(moves[:k]),
                             "wtime": times[0], "btime": times[1], "winc": inc * 1000,
                             "binc": inc * 1000, "status": "started"} | kw  # fmt: skip
    full = {"type": "gameFull", "id": h["GameId"], "variant": {"key": "standard"},
            "speed": h["Event"].split()[1].lower(), "rated": rated, "clock": clock,
            "white": player(h["White"], h["WhiteElo"]),
            "black": player(h["Black"], h["BlackElo"]), "initialFen": "startpos",
            "state": state(0)}  # fmt: skip
    during, after = script
    script = {}  # ply -> its events: a message, a burst of them, or "draw"
    for f, x in during:
        script.setdefault(f if isinstance(f, int) else round(f * len(moves)), []).append(x)
    feed = lambda *es: [c.put(e) for e in es] and c.q.join()
    line = lambda text: {"type": "chatLine", "room": "player", "username": opp, "text": text}
    say = lambda k, *texts: ([posts.append((k, opp, x)) for x in texts], feed(*map(line, texts)))
    feed(full)
    for k in range(len(moves) + 1):
        if k:
            if clocks[k - 1] is not None:
                times[(k - 1) % 2] = int(clocks[k - 1] * 1000)
            feed(state(k))
        them = (k % 2 == 0) != white  # their move next
        for x in script.get(k, []):
            match x:
                case "draw" if them:
                    posts.append((k, opp, "(offers a draw)"))
                    feed(state(k, **{"wdraw" if not white else "bdraw": True}), state(k))
                case "draw":
                    script.setdefault(k + 1, []).append("draw")
                case list():
                    say(k, *x)
                case _:
                    say(k, x)
    # the game's real end when the replay reaches it, else the side worse off resigns
    board = chess.Board()
    for u in moves:
        board.push_uci(u)
    full_game = len(moves) == len(read(path, 10_000)[1])
    winner = {"1-0": "white", "0-1": "black"}.get(h["Result"]) if full_game else None
    if winner is None and not full_game:
        e = chat.score(c.views[max(c.views)].wdl) if c.views else 0.5  # ours
        winner = ("white" if white else "black") if e >= 0.5 else ("black" if white else "white")
    status = "mate" if board.is_checkmate() else "outoftime" if "Time" in h.get(
        "Termination", "") else "draw" if winner is None else "resign"  # fmt: skip
    if end:
        status, winner = end
    who = "Allie" if winner and (winner == "white") != white else opp  # the side that ended it
    posts.append((len(moves), who if winner else "--", f"({status})"))
    feed(state(len(moves), status=status, winner=winner), chat.END)
    for text in after:
        say(len(moves) + 1, text)
    c.done = True
    return h, moves, opp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--llm", choices=["claude", "mock"], default="mock")
    p.add_argument("--ledger", help="the spend ledger (default: the config's)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recent", help="the recent-lines file (default: a new temporary one)")
    p.add_argument("--prompts", action="store_true", help="print each call's new turn")
    p.add_argument("--log", help="a bot log: replay each live game (PGN) with its opponent's real lines")
    p.add_argument("games", nargs="+", help="PGN:casual|rated:plies[:script], or PGN with --log")
    a = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("allie.lichess.chat").addFilter(
        lambda r: "not posted" in r.getMessage() or r.levelno >= logging.WARNING
    )
    logging.getLogger("allie.lichess.chat").setLevel(logging.INFO)

    from allie.lichess.cli import model
    from allie.lichess.config import load
    from allie.lichess.engine import Engine

    config = load(a.config)
    config.chat.enabled, config.chat.llm, config.chat.gap, config.chat.typing = True, a.llm, 0.0, 0.0
    if a.ledger:
        config.chat.ledger = a.ledger
    config.chat.recent_file = a.recent or tempfile.mktemp(suffix=".json")
    engine = Engine(model(config))
    llm = chat.model(config.chat)
    for i, spec in enumerate(a.games):
        path, *rest = spec.split(":")
        kind, plies, script = (rest + ["casual", 10_000, "chatty"][len(rest) :])[:3]
        end, script = None, SCRIPTS.get(script)
        if a.log:
            h = read(path, 0)[0]
            kind, plies = h["Event"].split()[0], 10_000
            script, end = from_log(a.log, h)
        posts, turns = [], []
        ref = {}
        at = lambda ref=ref: len(ref["c"].sans) + bool(ref["c"].status)
        post = lambda gid, text, room, posts=posts, at=at, **kw: posts.append((at(), BOT, text))
        client = SimpleNamespace(chat_lines=lambda gid, **kw: [], chat=post)
        bot = SimpleNamespace(me=BOT.lower(), config=config, client=client, engine=engine)
        match = SimpleNamespace(bot=bot, gid=f"dry{i}", over=False)

        def logged(system, messages, turns=turns, posts=posts, at=at):
            turns.append(messages[-1]["content"])
            r = llm(system, messages)
            if r is not None and not r.speak:
                posts.append((at(), BOT, "(stays silent)"))
            return r

        c = ref["c"] = Chatter(match, logged, a.seed + i)
        h, moves, _ = replay(c, posts, path, kind == "rated", int(plies), script, end)
        b = chess.Board()
        sans = []
        for u in moves:
            sans.append(b.san(m := b.parse_uci(u)))
            b.push(m)
        us = "white" if h["White"] == BOT else "black"
        print(f"\n=== {kind} {h['TimeControl']}: {h['White']} - {h['Black']}, Allie is {us}; "
              f"{len(moves)} plies ===")  # fmt: skip
        k = 0
        for ply, who, text in posts:
            k = k if ply is None else ply
            where = "start" if k == 0 else f"{chat.num(k - 1)}{sans[k - 1]}" if k <= len(sans) else "after"
            print(f"  [{where:>14}] {who}: {text}")
        print("  replies:")
        for t in c.timings:
            print(f"    {t['reason']:28} first token {t['first']:.2f} s, model {t['model']:.2f} s, "
                  f"${t['usd']:.4f}{'' if t['text'] else ', silent'}")  # fmt: skip
        for j, turn in enumerate(turns if a.prompts else []):
            print(f"\n--- turn {j} ---\n{turn}")
    if usage := getattr(llm, "usage", None):
        print("\nAPI usage:", usage)
    engine.close()


if __name__ == "__main__":
    main()
