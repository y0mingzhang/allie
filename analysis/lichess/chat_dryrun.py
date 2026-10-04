"""Dry run of allie.lichess.chat on recorded games: the chat a game would have had.

Each game's moves and clocks go to a real Chatter (the bot's model, the chat's model) as the
game stream would send them, with a script of opponent messages, a draw offer and a
resignation. Prints each transcript (with the model's silences), every reply's latency
and cost, and the spend.

usage: python analysis/lichess/chat_dryrun.py --config bot.toml [--llm claude|mock]
           [--ledger PATH] [--seed N] [--prompts]
           GAME.pgn:casual|rated:PLIES[:chatty|quiet|silent]...
"""

import argparse
import logging
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


def replay(c, posts, path, rated, plies, script):
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
    during, after = SCRIPTS[script]
    script = {round(f * len(moves)): text for f, text in during}
    feed = lambda *es: [c.put(e) for e in es] and c.q.join()
    say = lambda k, text: (posts.append((k, opp, text)),
                           feed({"type": "chatLine", "room": "player", "username": opp,
                                 "text": text}))  # fmt: skip
    feed(full)
    for k in range(len(moves) + 1):
        if k:
            if clocks[k - 1] is not None:
                times[(k - 1) % 2] = int(clocks[k - 1] * 1000)
            feed(state(k))
        them = (k % 2 == 0) != white  # their move next
        match script.get(k):
            case "draw" if them:
                posts.append((k, opp, "(offers a draw)"))
                feed(state(k, **{"wdraw" if not white else "bdraw": True}), state(k))
            case "draw":
                script[k + 1] = "draw"
            case str() as text:
                say(k, text)
    winner = "white" if white else "black"
    posts.append((len(moves), opp, "(resigns)"))
    feed(state(len(moves), status="resign", winner=winner), chat.END)
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
    p.add_argument("--prompts", action="store_true", help="print each call's new turn")
    p.add_argument("games", nargs="+", help="PGN:casual|rated:plies")
    a = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    from allie.lichess.cli import model
    from allie.lichess.config import load
    from allie.lichess.engine import Engine

    config = load(a.config)
    config.chat.enabled, config.chat.llm, config.chat.gap = True, a.llm, 0.0
    if a.ledger:
        config.chat.ledger = a.ledger
    engine = Engine(model(config))
    llm = chat.model(config.chat)
    for i, spec in enumerate(a.games):
        path, kind, plies, script = (spec.split(":") + ["chatty"])[:4]
        posts, turns = [], []
        ref = {}
        at = lambda ref=ref: len(ref["c"].sans) + bool(ref["c"].status)
        post = lambda gid, text, room, posts=posts, at=at, **kw: posts.append(
            (at(), BOT, text)
        )
        client = SimpleNamespace(chat_lines=lambda gid, **kw: [], chat=post)
        bot = SimpleNamespace(
            me=BOT.lower(), config=config, client=client, engine=engine
        )
        match = SimpleNamespace(bot=bot, gid=f"dry{i}", over=False)

        def logged(system, messages, turns=turns, posts=posts, at=at):
            turns.append(messages[-1]["content"])
            r = llm(system, messages)
            if r is not None and not r.speak:
                posts.append((at(), BOT, "(stays silent)"))
            return r

        c = ref["c"] = Chatter(match, logged, a.seed + i)
        h, moves, opp = replay(c, posts, path, kind == "rated", int(plies), script)
        b = chess.Board()
        sans = []
        for u in moves:
            sans.append(b.san(m := b.parse_uci(u)))
            b.push(m)
        us = "white" if h["White"] == BOT else "black"
        print(f"\n=== {kind} {h['TimeControl']}: {h['White']} - {h['Black']}, Allie is {us}; "
              f"{len(moves)} plies, then {opp} resigns ===")  # fmt: skip
        k = 0
        for ply, who, text in posts:
            k = k if ply is None else ply
            where = (
                "start"
                if k == 0
                else f"{chat.num(k - 1)}{sans[k - 1]}"
                if k <= len(sans)
                else "after"
            )
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
