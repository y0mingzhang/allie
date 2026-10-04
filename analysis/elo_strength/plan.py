"""Opening book and game schedules for the Elo-strength benchmark (see DESIGN.md).

python plan.py openings DIR [--candidates 400 --keep 150]
python plan.py round1 DIR
python plan.py round2 DIR --ratings DIR/ratings-r1.json [--games 200]
"""

import argparse
import itertools
import json
import random
from collections import Counter
from pathlib import Path

GRID = (800, 1200, 1600, 2000, 2400, 2800)
ALLIE = [f"allie-h{r}" for r in GRID] + ["allie-a2800", "allie-s5-2800"]
MAIA = [f"maia3-h{r}" for r in GRID]
LADDER = [1320, 1500, 1700, 1900, 2100, 2300, 2500, 2700, 2900, 3190]


def openings(out, candidates, keep, plies=6, rating=1600, max_cp=75):
    import torch
    from allie.lichess.engine import Engine, Game, Play
    from allie.lichess.model import Model

    from play import BASE, INC, MODEL, stockfish

    torch.set_num_threads(8)
    engine = Engine(Model(MODEL))
    play = Play(mode="human", rating=rating, temperature=1.0)
    lines, seen = [], set()
    for i in range(candidates):
        g = Game(engine, rating, rating, BASE, INC, "blitz", seed=10_000 + i)
        moves, thinks, clocks = [], [], [float(BASE)] * 2
        for ply in range(plies):
            d = g.decide(play, None, clocks[ply % 2])
            if ply >= 2:
                clocks[ply % 2] += INC - d.think
            moves.append(d.move)
            thinks.append(round(d.think, 2))
            g.update(moves, *clocks)
        if tuple(moves) not in seen:
            seen.add(tuple(moves))
            lines.append(dict(moves=moves, think=thinks))
    engine.close()
    sf = stockfish()
    book = []
    for line in lines:
        _, cp = sf.go(line["moves"], depth=18)
        if cp is not None and abs(cp) <= max_cp:
            book.append(line | dict(cp=cp))
    sf.close()
    print(
        f"{candidates} sampled, {len(lines)} distinct, {len(book)} within {max_cp} cp"
    )
    first = Counter(" ".join(b["moves"][:2]) for b in book[:keep])
    print("first moves:", first.most_common(8))
    Path(out, "openings.json").write_text(json.dumps(book[:keep], indent=0) + "\n")


def pairing(a, b, games, start=0):
    """Opening k twice, colours swapped; ids are unique per (pair, opening, colour)."""
    return [
        dict(id=f"{a}|{b}|{k}|{w}", white=(a, b)[w], black=(b, a)[w], opening=k)
        for k in range(start, start + games // 2)
        for w in (0, 1)
    ]


def write(path, tasks):
    random.Random(0).shuffle(tasks)  # spread costly pairings over shards
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    print(f"{path}: {len(tasks)} games")


def scheduled(d):
    """Distinct games already scheduled per unordered pair."""
    ids = {
        json.loads(line)["id"]
        for f in Path(d, "tasks").glob("*.jsonl")
        for line in open(f)
    }
    return Counter(tuple(sorted(i.split("|")[:2])) for i in ids)


def split(d, name, tasks):
    """Games with an Allie player need the 11 GB model; the rest go to light jobs."""
    heavy = [t for t in tasks if "allie" in t["white"] + t["black"]]
    write(Path(d, f"tasks/{name}-allie.jsonl"), heavy)
    write(Path(d, f"tasks/{name}-light.jsonl"), [t for t in tasks if t not in heavy])


def round1(d):
    anchors = (1320, 1700, 2100, 2500, 2900)
    tasks = [
        t for a in ALLIE + MAIA for e in anchors for t in pairing(a, f"sf-{e}", 20)
    ]
    for chain in (ALLIE, MAIA, [f"sf-{e}" for e in LADDER]):
        tasks += [t for a, b in itertools.pairwise(chain) for t in pairing(a, b, 40)]
    tasks += [t for r in GRID for t in pairing(f"allie-h{r}", f"maia3-h{r}", 40)]
    split(d, "r1", tasks)


def round2(d, ratings, games, anchors=3):
    """Each Allie and Maia-3 player: `games` more games split over the `anchors` ladder levels
    nearest its round-1 rating (one end of the ladder if it sits beyond it)."""
    est = json.loads(Path(ratings).read_text())["ratings"]
    have, tasks = scheduled(d), []
    for a in ALLIE + MAIA:
        near = sorted(LADDER, key=lambda e: abs(e - est[a]["elo"]))[:anchors]
        for e in near:
            pair = tuple(sorted((a, f"sf-{e}")))
            tasks += pairing(*pair, games // anchors, start=have[pair] // 2)
        print(a, round(est[a]["elo"]), "->", sorted(near))
    # argmax at lower conditioning: does the strongest mode depend on its header?
    for a in ("allie-a1200", "allie-a2000"):
        tasks += [
            t for e in (1700, 1900, 2100, 2300, 2500) for t in pairing(a, f"sf-{e}", 40)
        ]
    tasks += [
        t
        for a, b in (("allie-a1200", "allie-a2000"), ("allie-a2000", "allie-a2800"))
        for t in pairing(a, b, 40)
    ]
    split(d, "r2", tasks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["openings", "round1", "round2"])
    p.add_argument("dir")
    p.add_argument("--candidates", type=int, default=400)
    p.add_argument("--keep", type=int, default=150)
    p.add_argument("--ratings")
    p.add_argument("--games", type=int, default=200)
    a = p.parse_args()
    match a.command:
        case "openings":
            openings(a.dir, a.candidates, a.keep)
        case "round1":
            round1(a.dir)
        case "round2":
            round2(a.dir, a.ratings, a.games)


if __name__ == "__main__":
    main()
