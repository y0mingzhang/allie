"""Builds data.json, every number the video shows with its source, from measure.py's outputs on group storage.

prep.py WORK_DIR, where WORK_DIR holds measure.py's game.npz and game.npz.json, scan.npz, lesion-2.json, ratings.json,
recapture.json, heads-2.json and pick-cpu.jsonl.
"""

import json
import math
import sys
from pathlib import Path

import chess
import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path(sys.argv[1])
BLOCK = 11  # block 15's row among the routed blocks 4-24: the one the Router scene opens
HEADER = 11
SEQ_LEN = 512


def main():
    z = np.load(WORK / "game.npz")
    g = json.loads((WORK / "game.npz.json").read_text())
    meta, plies = g["game"], g["plies"]
    uci = [p["uci"] for p in plies]
    clocks = meta["clocks"]
    base, inc = meta["base"], meta["increment"]
    left = [base, base, *clocks]
    k = g["ply"]
    p = HEADER - 1 + k
    feats = z["feats"][p]
    thinks = [left[i] - clocks[i] + inc if i >= 2 else None for i in range(len(plies))]

    position = chess.Board(plies[k]["fen"])
    names = {m.uci(): position.san(m) for m in position.legal_moves}
    real = {u: round(q, 4) for _, u, q in plies[k]["top"]}
    round_dist = lambda d: {u: round(v, 4) for u, v in d.items() if v >= 0.004}

    checks, pawn = [], []
    for i, q in enumerate(plies):
        b = chess.Board(q["fen"])
        if b.is_check():
            checks.append(i)
        if b.piece_at(chess.G4) == chess.Piece(chess.PAWN, chess.BLACK):
            pawn.append(i)

    embed, board, clock = (z[n][p].astype(float) for n in ("embed", "board", "clock"))
    total = embed + board + clock
    router_input = z["router_input"][BLOCK, p].astype(float)
    step = len(total) // SEQ_LEN
    parts = {
        "_src": "the focus token's real vectors in the released weights, every 4th of the 2,048 numbers: Model.board, Model.clock and the embed table (game.npz)",
        "scale": round(float(np.percentile(np.abs(total), 98)), 3),
        "embed_scale": round(float(np.percentile(np.abs(embed), 98)), 3),
        "router_scale": round(float(np.percentile(np.abs(router_input), 98)), 3),
        "router_input": router_input[::step].round(2).tolist(),
        "embed": embed[::step].round(2).tolist(),
        "board": board[::step].round(2).tolist(),
        "clock": clock[::step].round(2).tolist(),
        "sum": total[::step].round(2).tolist(),
    }

    scores = z["scores"][BLOCK, p].astype(float)
    bias = z["bias"][BLOCK]
    s = 1 / (1 + np.exp(-scores))
    idx = z["idx"]
    route = {
        "_src": "router choices of the released weights on this game, all 21 routed blocks, per position (game.npz, measure.py game); blocks count from 1 as in DETAILS (1-24), experts and heads from 0",
        "blocks": [int(b) + 1 for b in z["blocks"]],
        "idx": idx[:, HEADER - 1 :].transpose(1, 0, 2).tolist(),
        "focus": {
            "row": BLOCK,
            "block": int(z["blocks"][BLOCK]) + 1,
            "score": s.round(4).tolist(),
            "selected": (s + bias).round(4).tolist(),
            "ids": idx[BLOCK, p].tolist(),
            "gates": z["gate"][BLOCK, p].astype(float).round(4).tolist(),
        },
    }

    attn = z["attn"].astype(np.float32)  # [block, head, token, key]
    h = json.loads((WORK / "heads-2.json").read_text())
    tens = h["mover_digits"][2][3][7]
    pick = [
        (3, 7, f"the rating of the player to move, {100 * tens:.0f}% on its tens digit"),
        (0, 17, "the time control"),
        (8, 10, "the player's own last move"),
    ]
    examples = []
    for blk, head, label in pick:
        at = lambda t: attn[blk, head, t, : t + 1].round(4).tolist()
        examples.append(
            dict(
                block=blk + 1, head=head, label=label, weights={str(t - (HEADER - 1)): at(t) for t in (p, p + 1, p + 2)}
            )
        )
    groups = np.array([h[k] for k in ("mover", "other", "control", "last", "own_last")])
    top = np.concatenate([groups, [np.array(h["header"]) - groups[:3].sum(0)]]).max(0)
    strength = np.array(h["xsa"])
    attention = {
        "_src": "attention weights of the released weights: this game for the shown heat (game.npz); shares from measure.py heads, positions from ply 4 on, over all 24 blocks and 32 heads. even_rating, even_header: the same shares if every head spread its attention evenly over the tokens it can see. head_top_*: each head's largest share among six groups (the mover's rating, the other's, the time control, the token itself, the mover's last move, the start token), on the same 40 games the shown heads were picked on. xsa_*: tanh of the weights' xsa table, the amount of its projection on the token's own value each head subtracts: 1 removes it all, below 0 adds to it. A last-8-plies mask limits direct attention only: older moves still flow through the tokens in between; a true re-read of the header and last 8 plies cost +0.055 [0.033, 0.081] against the mask's +0.030 on 25 games, in a separate audit (release-2.1/explainers/review/audit.Usie/recheck.py)",
        "games": h["games"],
        "positions": h["positions"],
        "mean_at_focus": attn[:, :, p, : p + 1].mean((0, 1)).round(4).tolist(),
        "header_share": round(h["header_share"], 3),
        "rating_share": round(h["rating_share"], 3),
        "even_header": round(h["even_header"], 3),
        "even_rating": round(h["even_rating"], 3),
        "head47_tens": tens,
        "head_top_over_half": int((top >= 0.5).sum()),
        "head_top_median": round(float(np.median(top)), 3),
        "xsa_median": round(float(np.median(strength)), 3),
        "xsa_negative": round(float((strength < 0).mean()), 3),
        "xsa_over_half": round(float((strength > 0.5).mean()), 3),
        "head47_mover": h["mover"][3][7],
        "head47_other": h["other"][3][7],
        "time_control_head": h["control"][0][17],
        "own_last_head": h["own_last"][8][10],
        "examples": examples,
    }

    out = {
        "_src": "how each block was measured is in its own _src. measure.py: the released Allie 2.1, PyTorch reference, on one September 2026 Lichess game and samples of that month, after all of its training data. The game and the attention heads ran on a CPU in int8, the expert scan and the lesions on a GPU in BF16, as Allie.from_pretrained loads each",
        "game": {
            "_src": "Lichess open database, September 2026, rated blitz (heldout-2026-09/store, "
            + g["shard"]
            + "); game "
            + meta["site"]
            + "; no player names are shown",
            "tc": [base, inc],
            "white": meta["white_elo"],
            "black": meta["black_elo"],
            "result": ["1-0", "0-1", "1/2-1/2"][meta["result"]],
            "opening": meta["opening"],
            "uci": uci,
            "san": [q["san"] for q in plies],
            "clocks": clocks,
            "thinks": thinks,
            "fens": [q["fen"] for q in plies],
            "in_check": checks,
            "pawn_g4": pawn,
        },
        "plies": [
            {
                "p": round(q["p"], 4),
                "surprise": round(-math.log(q["p"]), 3),
                "rank": q["rank"],
                "top": [[a, b, round(c, 4)] for a, b, c in q["top"][:5]],
                "wdl": [round(x, 3) for x in q["wdl"]],
                "expect": round(q["expect"], 2),
            }
            for q in plies
        ],
        "plies_src": "measure.py game: Model.forward reference, legal moves renormalized as Allie.analyze does; think expect from the 63 think bins and CENTRES",
        "mean_surprise": round(sum(-math.log(q["p"]) for q in plies) / len(plies), 3),
        "top1_hits": sum(q["rank"] == 1 for q in plies),
        "focus": {
            "_src": "measure.py game: rating sweep changes only the mover's four header digits; clock sweep only the mover's time left at the last token",
            "k": k,
            "san": names,
            "real": real,
            "left": int(feats[0]),
            "clocks": {"left": int(feats[0]), "other": int(feats[1]), "prev": int(feats[2])},
            "rating": {r: round_dist(v) for r, v in g["rating"].items()},
            "clock": {r: round_dist(v) for r, v in g["clock"].items()},
            "think": [round(x, 4) for x in plies[k]["think"]],
            "swing": swing(meta["site"], k),
        },
        "centres": [int(i) for i in range(16)] + [round(16 * math.exp(i / 7.06), 1) for i in range(47)],
        "parts": parts,
        "route": route,
        "attention": attention,
        "lesion": lesions(),
        "recapture": json.loads((WORK / "recapture.json").read_text())
        | {
            "_src": "measure.py recapture: the clock sweep on contested recaptures, so the shown position's rise and fall back can be compared with its like; rise_10s: 10 s against 180 s; drop_2s: 2 s against 10 s, negative for a drop; intervals over positions, about one a game"
        },
        "scan": scan(),
        "facts": facts(),
    }
    (HERE / "data.json").write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print((HERE / "data.json").stat().st_size, "bytes")


def swing(site, k):
    """How far rating moves the shown position, among all positions of the games it was picked from."""
    rows = [json.loads(line) for line in open(WORK / "pick-cpu.jsonl")]
    tv = np.array([p["tv"] for r in rows for p in r["plies"]])
    here = next(r for r in rows if r["site"] == site)["plies"][k]["tv"]
    return {
        "_src": "measure.py pick: total variation between the move distributions at both ratings 1000 and both 2500; blitz and 10-minute games, ratings 1300-2300, 50-120 plies; the position was hand-picked from this list",
        "swing": here,
        "median": float(np.median(tv)),
        "rank": int((tv > here).sum()) + 1,
        "positions": len(tv),
        "games": len(rows),
    }


RASTER_BLOCK = 8  # no order in its raw expert numbers, a clear one once sorted by when they run
DETECTORS = (
    (15, 126, "check", "of positions in check", "of its runs are in check"),
    (16, 243, "p@g4", "of positions with a Black pawn on g4", "of its runs have it on g4; most others, on g3"),
)
CARDS = (
    (14, 183, "elo<1000", "player rated under 1000"),
    (14, 235, "elo>=2600", "player rated 2600 or more"),
    (15, 67, "left<5", "under 5 seconds left"),
    (10, 95, "ply<10", "the first 10 plies"),
    (14, 159, "nextcastle", "about to castle"),
    (19, 106, "pieces<5", "4 or fewer pieces on the board"),
)


def scan():
    """Experts against position features. Each shown expert has the top F1 score for its feature on one half of the games,
    which asserts; every number here is measured on the other half."""
    z = np.load(WORK / "scan.npz")
    C, N, blocks, ov = z["C"], z["N"], (z["blocks"] + 1).tolist(), z["overlap"]
    names = [str(x) for x in z["names"]]
    ix = {n: i for i, n in enumerate(names)}
    white, black = ix["side=W"], ix["side=B"]
    pos, fires = N[:, white] + N[:, black], C[:, :, :, white] + C[:, :, :, black]
    row = lambda block: blocks.index(block)

    def f1(half):
        recall = C[half] / np.maximum(N[half][None, None], 1)
        precision = C[half] / np.maximum(fires[half][:, :, None], 1)
        return 2 * recall * precision / np.maximum(recall + precision, 1e-9)

    picking, held = f1(0), f1(1)

    def measured(block, expert, feature):
        i, j = row(block), ix[feature]
        assert picking[i, expert, j] == picking[:, :, j].max(), (block, expert, feature)
        hits, n, runs = C[1, i, expert, j], N[1, j], fires[1, i, expert]
        return dict(
            block=block,
            expert=expert,
            feature=feature,
            positions=int(n),
            rate=round(float(hits / n), 4),
            base=round(float(runs / pos[1]), 4),
            precision=round(float(hits / runs), 4),
            elsewhere=round(float((runs - hits) / (pos[1] - n)), 5),
            f1_picking=round(float(picking[i, expert, j]), 3),
            f1=round(float(held[i, expert, j]), 3),
        )

    def rate_where(block, expert, features):
        hits = sum(C[1, row(block), expert, ix[f]] for f in features)
        return round(float(hits / sum(N[1, ix[f]] for f in features)), 4)

    bins = ["ply<10", "ply<20", "ply<40", "ply<60", "ply<80", "ply<120", "ply>=120"]
    mid = np.array([5, 15, 30, 50, 70, 100, 140.0])
    j = row(RASTER_BLOCK)
    rate = C[0, j][:, [ix[b] for b in bins]] / N[0][[ix[b] for b in bins]]
    mean_ply = (rate * mid).sum(1) / np.maximum(rate.sum(1), 1e-12)
    rank = np.argsort(np.argsort(mean_ply)).tolist()

    usable = np.array([n != "header" and min(N[0, i], N[1, i]) >= 1500 for i, n in enumerate(names)])
    best = (picking * usable).argmax(2)
    quality = np.take_along_axis(held, best[:, :, None], 2)[:, :, 0]
    next_token, same_side = ov[:, 0] / ov[:, 1], ov[:, 2] / ov[:, 3]
    return {
        "_src": f"measure.py scan: the released weights, PyTorch reference, on {int(z['games']):,} games of September 2026, {int(pos.sum()):,} positions; each shown expert has the top F1 score for its feature on one random half of the games, every rate here is measured on the other half; precision: the share of its runs on target; firing rates count the positions predicting a move, not the header; rank: sort of block {RASTER_BLOCK}'s experts by their mean ply on the first half",
        "games": int(z["games"]),
        "positions": int(pos.sum()),
        "half": int(pos[1]),
        "features_tried": int(usable.sum()),
        "block": RASTER_BLOCK,
        "row": j,
        "rank": rank,
        "detectors": [measured(b, e, f) | {"of": w, "runs": r} for b, e, f, w, r in DETECTORS],
        "cards": [measured(b, e, f) | dict(what=w) for b, e, f, w in CARDS],
        "f1_over_0.8": int((quality >= 0.8).sum()),
        "f1_over_0.5": int((quality >= 0.5).sum()),
        "f1_median": round(float(np.median(quality)), 3),
        "experts": int(quality.size),
        "overlap_next": round(float(next_token[row(15)]), 2),
        "overlap_same_side": round(float(same_side[row(15)]), 2),
        "overlap_next_range": [round(float(next_token.min()), 2), round(float(next_token.max()), 2)],
        "clock_expert_above_30s": rate_where(15, 67, ["left<60", "left<120", "left<300", "left<900", "left>=900"]),
        "pawn_expert_on_g3": {
            "rate": rate_where(16, 243, ["p@g3"]),
            "share_of_runs": round(float(C[1, row(16), 243, ix["p@g3"]] / fires[1, row(16), 243]), 4),
        },
    }


def lesions():
    """The study's second run: its first let the best-k cases' expert count leak into the later cases, whose numbers are
    void; random experts ran before the leak, so its first draw counts as a second draw."""
    src = "measure.py lesion: the released weights with one part changed after training, paired on the same positions; CE over legal moves in nats, on a natural mix of a random sample of September 2026 games, so higher than the main test set's plain mean of 16 cells; intervals over games"
    check = "cross-check through the public Allie.analyze on the CPU (int8, Rust engine), first 1,076 positions of a separate sample of 40 games: loss 1.2887 real, +0.0077 clocks hidden, +0.0316 both ratings at 1626"
    out = json.loads((WORK / "lesion-2.json").read_text())
    no = out["cases"]["no_clock"]
    low = no["slices"]["under 30 s"]
    others = (no["ce_diff"] * out["plies"] - low["diff"] * low["n"]) / (out["plies"] - low["n"])
    return out | {
        "_src": src,
        "check": check,
        "best_k": "lesion_mlp keeps the router's top k by selection score with their original weights, not rescaled to sum to 4",
        "random_second_draw": 2.5237,
        "no_clock_other_positions": round(others, 4),
        "ratings_by_mover": json.loads((WORK / "ratings.json").read_text())
        | {
            "_src": "measure.py ratings: both ratings at the median, the lesion's games and case again, cut by the mover's rating; distance: mean points from the median"
        },
    }


def facts():
    """Numbers taken from the public documents, each with where it stands. README is README.md, DETAILS docs/DETAILS.md."""
    return {
        "_src": "README and DETAILS sentences named in sources; computed numbers show their arithmetic there",
        "tokens_read": "112B",
        "lichess_games": "7.9B",
        "otb_games": "5.1M",
        "loss_21": 1.2240,
        "loss_20": 1.2505,
        "loss_original": 1.3526,
        "bands": [
            ["Under 1400", 1.4473, 1.4312],
            ["1400-2000", 1.2888, 1.2681],
            ["2000-2400", 1.1659, 1.1362],
            ["2400+", 1.1001, 1.0603],
        ],
        "top1_blitz": "60.1%",
        "top1_2400": "64.5%",
        "ms": "14.3 ms",
        "gb": "1.29 GB",
        "trained": "Trained for 14.9 days on one machine with 8 GPUs",
        "total": "9.19B",
        "active": "1.23B",
        "hidden_dense": 5456,
        "hidden_expert": 256,
        "hidden_shared": 1360,
        "dense_vs_moe": [2.2, 2.5],
        "dense_interval": [1.99, 3.32],
        "dense_budget_gap": ["130", "1,300"],
        "clock_small": 0.007,
        "rating_small": 0.067,
        "rating_small_2400": 0.111,
        "links": [
            "lichess.org/@/AllieTheChessBot",
            "huggingface.co/yimingzhang/allie-2.1",
            "github.com/y0mingzhang/allie",
        ],
        "weak": [
            [
                "First 10 plies, the opening: no better than Allie 2.0",
                "+0.0008 [-0.0019, +0.0038] on the blitz benchmark",
            ],
            ["Below 1000: its lead over Allie 2.0 is within the noise", "0.006 to 0.007 on the blitz benchmark"],
            ["Favouring strong players costs weak ones", "+0.012 to +0.043 under 1400, measured on small models"],
            [
                "Sampled straight, it plays below humans of its rating",
                "most at high ratings and in classical; the bot searches to fix it",
            ],
            ["It is big", "18 GB download, 9.8 GB in memory; Maia-3 79M, another human-move model, is about 0.3 GB"],
            ["It knows only this game, two ratings and the clocks", "nothing about this player's history or style"],
        ],
        "sources": {
            "112B tokens read": "README, Allie 2.1 paragraph",
            "7.9B Lichess games": "README, Data",
            "5.1M over-the-board games; engine games under 1% of tokens": "DETAILS, Training, Data (Allie 2.1 keeps the same data: DETAILS, Allie 2.1, Training)",
            "1,968 move tokens": "src/allie/lichess/tokens.py MOVES; DETAILS, Evaluation",
            "2,432 rows and outputs; 2,048 width; 24 blocks; 32 heads; 256 experts; 16 chosen": "release-2.1/export/config.json and the weights' header",
            "13 planes, 3 convolution layers, 32 channels": "DETAILS, The model; Model.board",
            "eight clock hands": "Model.clock: 8 sines and 8 cosines of pi 2^k ln(1 + t) / 10 per time",
            "5,456 hidden units; 16 x 256 + 1,360": "DETAILS, Allie 2.1, Model",
            "66,896 stored hidden units, 12 times": "arithmetic: 256 x 256 + 1,360, and 66,896 / 5,456 = 12.3",
            "336 of 5,376 experts": "arithmetic: 21 x 16 and 21 x 256",
            "9.19B stored, 1.23B per move": "DETAILS, Allie 2.1, Model",
            "2.2 to 2.5 times the training compute; 90% intervals 1.99 to 3.32; 130 to 1,300 times below Allie 2.1's budget": "README, Scaling laws: read off two fitted laws, not measured head to head. figures/data.json cm: 2.17 [1.99, 2.33], 2.31 [2.06, 2.62], 2.50 [1.99, 3.32] at the sweep's 6.3e17, 1.9e18, 6.2e18 training FLOPs (DETAILS, Scaling laws); Allie 2.1's 8.27e20 (DETAILS, Allie 2.1) is 1,313 and 133 times the ends",
            "116 of 256 experts starved; centring plus Adam on every step": "DETAILS, Training stability: Allie 2.0's first attempt, which already had capped biases",
            "exclusive self-attention": "DETAILS, Allie 2.1, Model; strengths in the weights' xsa table: heads shown read other tokens, whose values it keeps",
            "small model: clocks +0.007; ratings +0.067, +0.111 at 2400+": "DETAILS, Design findings, Inputs (elo-conditioning/probe/moe1e18, the same test after training, on the main evaluation's positions)",
            "10% of training games without clocks": "DETAILS, The model; data/mix.py Games.dropped",
            "63 think bins, 3 result numbers, weight 0.2": "DETAILS, The model, Head; api.py TIME and WDL",
            "1.2240, 1.2505, 1.3526": "README, What Allie is: main test set, July 2026",
            "loss by rating band": "DETAILS, Allie 2.1, Main evaluation table",
            "60.1%, 64.5%": "README, What Allie is; DETAILS, Blitz benchmark: 20,000 positions in each of four rating bands; 64.5% is the 2400+ band",
            "original Allie trained on blitz only": "DETAILS, The original Allie: only blitz is in its training data, so the main evaluation's other time controls get its unknown time-control token",
            "first 10 plies +0.0008 [-0.0019, +0.0038]": "DETAILS, Allie 2.1, Blitz benchmark",
            "below 1000: 0.006 to 0.007, within the noise": "DETAILS, Allie 2.1, Blitz benchmark",
            "+0.012 to +0.043 under 1400": "README, Data; DETAILS, Design findings: small-model results",
            "sampled straight, plays below humans": "README, Calibrated play",
            "18 GB, 9.8 GB, 0.3 GB": "README, Comparison and Download",
            "14.3 ms, 1.29 GB": "DETAILS, Allie 2.1, Inference",
            "14.9 days, 8 GPUs": "README, Training",
            "only this game, two ratings and the clocks": "src/allie/lichess/tokens.py and Model.forward: the model's whole input",
        },
    }


if __name__ == "__main__":
    main()
