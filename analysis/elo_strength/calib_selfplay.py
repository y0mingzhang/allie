"""Self-play check of a calibrated mode: games (play.py output) -> positions in calib_positions.py's
format, one per move from ply 10, the played move standing in for the human's; then calib_mpv.py
scores them and `report` compares each (mode, time control, rating) cell with the human curves.

python calib_selfplay.py tasks OUT.jsonl --modes allie-c,allie-h --ratings 1200,1600,2000,2400 --games 20
python calib_selfplay.py positions GAMES_DIR OUT.npz
python calib_selfplay.py report CALIB_DIR SELFPLAY.npz MPV_DIR
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from allie.data.vocab import INCREMENTS_ID, MOVE_ID, SECONDS_ID
from allie.lichess.tokens import header

import calib_fit as cf
from play import speed

TCS = {"bullet": (60, 0), "blitz": (180, 2), "rapid": (600, 5), "classical": (1800, 20)}


def tasks(out, modes, ratings, games):
    rows = []
    for name, tc in TCS.items():
        for mode in modes:
            for r in ratings:
                p = f"{mode}{r}"
                rows += [
                    dict(
                        id=f"{p}|{p}|{name}|{k}",
                        white=p,
                        black=p,
                        opening=k,
                        tc=list(tc),
                    )
                    for k in range(games)
                ]
    Path(out).write_text("".join(json.dumps(t) + "\n" for t in rows))
    print(f"{out}: {len(rows)} games")


def positions(games_dir, out):
    """One position per move from ply 10: tokens, meta (game, 0, format, bin, rating, rating, ply,
    mover clock, move token, mode index)."""
    games = [
        json.loads(line)
        for f in sorted(Path(games_dir).glob("*.jsonl"))
        for line in open(f)
        if line.strip()
    ]
    modes = sorted({re.sub(r"\d+$", "", g["white"]) for g in games})
    tokens, off, meta = [], [0], []
    for gi, g in enumerate(games):
        base, inc = g["tc"]
        r = int(re.findall(r"\d+$", g["white"])[0])
        moves = g["moves"].split()
        head = header(base, inc, r, r)
        assert head[1] == SECONDS_ID.get(str(base)) and head[2] == INCREMENTS_ID.get(
            str(inc)
        )
        seq = head + [MOVE_ID[m] for m in moves]
        f = cf.FORMATS.index(speed(base, inc).replace("ultraBullet", "bullet"))
        for k in range(10, len(moves)):
            tokens.append(np.array(seq[: 11 + k], np.int16))
            off.append(off[-1] + 11 + k)
            clock = g["clocks"][k - 2] if k >= 2 else base
            meta.append(
                (
                    gi,
                    0,
                    f,
                    r // 200 * 200,
                    r,
                    r,
                    k,
                    int(clock),
                    seq[11 + k],
                    modes.index(re.sub(r"\d+$", "", g["white"])),
                )
            )
    np.savez_compressed(out, tokens=np.concatenate(tokens), offsets=np.array(off), meta=np.array(meta, np.int64),
                        modes=np.array(modes))  # fmt: skip
    print(f"{len(games)} games, {len(meta)} positions, modes {modes} -> {out}")


def report(calib, sp, mpv):
    """Per mode x format x rating: the bot's own moves' accuracy / blunder / top-1 next to the human
    curve's value at that rating, and the gap in Elo through the human curve's local slope."""
    d = Path(calib)
    human = cf.summarize(
        cf.table(
            np.load(d / "positions-human.npz")["meta"],
            cf.load_mpv(d / "mpv-human"),
            {},
            {},
        ),
        [],
    )
    z = np.load(sp)
    meta, modes = z["meta"], z["modes"]
    by = defaultdict(list)
    for r in cf.table(meta, cf.load_mpv(mpv), {}, {}):
        by[str(modes[meta[r["i"], 9]]), r["f"], r["elo"]].append(r["human"])
    out = {}
    for (mode, f, elo), v in sorted(by.items()):
        v = np.array(v)
        row = {}
        for m in cf.HEADLINE:
            k = cf.METRICS.index(m)
            grid, fit, _ = cf.curve(human, f, k)
            ref = float(np.interp(elo, grid, fit))
            x = float(np.nanmean(v[:, k]))
            log = m in cf.LOG
            gap = (np.log(x / ref) if log else x - ref) / cf.slope(human, f, k, elo)
            row[m] = (round(x, 4), round(ref, 4), round(gap))
        out[f"{mode}/{cf.FORMATS[f]}/{elo}"] = dict(moves=len(v), **row)
        print(
            f"{mode:8s} {cf.FORMATS[f]:9s} {elo:4d} moves {len(v):5d}  "
            + "  ".join(
                f"{m} {a} vs human {b} ({g:+d} Elo)" for m, (a, b, g) in row.items()
            )
        )
    Path(sp).with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["tasks", "positions", "report"])
    p.add_argument("args", nargs="+")
    p.add_argument("--modes", default="allie-c,allie-h")
    p.add_argument("--ratings", default="1200,1600,2000,2400")
    p.add_argument("--games", type=int, default=20)
    a = p.parse_args()
    match a.command:
        case "tasks":
            tasks(
                a.args[0],
                a.modes.split(","),
                [int(r) for r in a.ratings.split(",")],
                a.games,
            )
        case "positions":
            positions(*a.args)
        case "report":
            report(*a.args)


if __name__ == "__main__":
    sys.exit(main())
