"""Reconstruct the golden eval's blitz positions (games + scored plies) from strat.npz.

Writes games.npz (move tokens, Elos, time control) and sample.npz (the fixed subsample
of scored blitz moves every model is scored on).
"""
import json
import sys

import chess
import numpy as np

from allie import paths
from allie.data.vocab import BOS, INCREMENTS, MOVES, SECONDS

G = paths.DATA / "strat-eval-v1"
OUT = paths.DATA / "maia3-bench"
BLITZ = (4, 5, 6, 7)  # cells blitz/<1400, /1400-2000, /2000-2400, />=2400
PER_CELL = int(sys.argv[1]) if sys.argv[1:2] not in ([], ["verify"]) else 20_000
TERM = (2346, 2347)


def main():
    manifest = json.loads((G / "manifest.json").read_text())
    with np.load(G / "strat.npz") as z:
        rows, labels = z["rows"].astype(np.int64), z["labels"].astype(np.int64)
    with np.load(G / "feats.npz") as z:
        feats = z["feats"].astype(np.int64)
    assert rows.shape == labels.shape == feats.shape[:2]

    gmoves, goff, gmeta = [], [0], []
    sel = []  # game, ply, cell, mover clock, opponent clock
    for r, (tok, lab, ft) in enumerate(zip(rows, labels, feats)):
        starts = [i for i in np.flatnonzero(tok == BOS) if i + 1 < len(tok) and tok[i + 1] != BOS]
        for s, e in zip(starts, starts[1:] + [len(tok)]):
            mv = tok[s + 11 : e]
            end = np.flatnonzero((mv == TERM[0]) | (mv == TERM[1]))
            mv = mv[: end[0]] if len(end) else mv
            assert len(mv) == 0 or (mv.min() >= 378 and mv.max() <= 2345)
            lb = lab[s + 11 : s + 11 + len(mv)]
            if not np.isin(lb, BLITZ).any():
                continue
            welo = int(tok[s + 3 : s + 7] @ [1000, 100, 10, 1])
            belo = int(tok[s + 7 : s + 11] @ [1000, 100, 10, 1])
            g = len(gmeta)
            for m in np.flatnonzero(np.isin(lb, BLITZ)):
                f = ft[s + 10 + m]
                sel.append((g, int(m), int(lb[m]), int(f[0]), int(f[1])))
            gmoves.append(mv - 378)
            goff.append(goff[-1] + len(mv))
            gmeta.append((int(tok[s + 1]), int(tok[s + 2]), welo, belo, r, int(s)))

    gmoves = np.concatenate(gmoves).astype(np.int16)
    goff = np.array(goff, np.int64)
    gmeta = np.array(gmeta, np.int64)
    sel = np.array(sel, np.int64)
    print(f"{len(gmeta)} blitz games, {len(sel)} scored blitz moves", flush=True)
    counts = np.bincount(sel[:, 2], minlength=16)[list(BLITZ)]
    expect = np.array(manifest["scored_moves"])[list(BLITZ)]
    assert (counts == expect).all(), (counts, expect)

    rng = np.random.default_rng(20260922)
    keep = np.concatenate(
        [rng.choice(np.flatnonzero(sel[:, 2] == c), PER_CELL, replace=False) for c in BLITZ]
    )
    keep.sort()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / "games.npz", moves=gmoves, offsets=goff, meta=gmeta, sel=sel, keep=keep,
        per_cell=PER_CELL,
    )
    meta = dict(
        source=str(G), strat_sha256=manifest["sha256"], feats_sha256=manifest["feats_sha256"],
        games=len(gmeta), scored_blitz=len(sel), per_cell=PER_CELL, seed=20260922,
        cells=[manifest["cells"][c] for c in BLITZ],
        counts=counts.tolist(),
    )
    (OUT / "positions.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


def verify(n=400):
    """Replay a sample of games and check every scored target move is legal."""
    with np.load(OUT / "games.npz") as z:
        moves, off, meta, sel = z["moves"], z["offsets"], z["meta"], z["sel"]
    rng = np.random.default_rng(1)
    pick = rng.choice(len(meta), min(n, len(meta)), replace=False)
    by_game = {}
    for g, m, c, c0, c1 in sel:
        by_game.setdefault(g, []).append(m)
    checked = 0
    for g in pick:
        b = chess.Board()
        uci = [MOVES[i] for i in moves[off[g] : off[g + 1]]]
        want = set(by_game.get(g, []))
        for m, u in enumerate(uci):
            mv = chess.Move.from_uci(u)
            assert mv in b.legal_moves, (g, m, u, b.fen())
            if m in want:
                checked += 1
            b.push(mv)
    base, inc = SECONDS[meta[pick[0]][0] - 192], INCREMENTS[meta[pick[0]][1] - 10]
    print(f"replayed {len(pick)} games, {checked} scored targets all legal; e.g. tc {base}+{inc} "
          f"elos {meta[pick[0]][2]}/{meta[pick[0]][3]}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["verify"]:
        verify()
    else:
        main()
