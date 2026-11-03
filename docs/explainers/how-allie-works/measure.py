"""Measurements behind How Allie Works: the released Allie 2.1 (PyTorch reference) on September 2026 Lichess games,
after all of its training data. Reads only the held-out September store.

scan OUT.npz --games N     experts' firing counts against position features, two independent halves of the games
lesion OUT.json --games N  the loss with one part of the model taken away at a time, paired on the same positions
ratings OUT.json --games N the loss with both ratings at the median, by the mover's rating, on the lesion study's games
recapture OUT.json --games N  the clock sweep on contested recaptures across blitz games
pick OUT.jsonl --games N   positions whose prediction moves most with rating, for choosing the shown game
heads OUT.json --games N   where each attention head looks, over a sample of games
game OUT.npz --site ID --shard S --ply K   one game in full: routing, router scores, attention, outputs; rating and
                           clock sweeps at ply K
"""

import argparse
import json
import zlib

import chess
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from allie import paths
from allie.lichess.api import CENTRES, TIME, WDL
from allie.lichess.model import Cache, Model, step
from allie.lichess.tokens import HEADER, MOVE_ID, MOVE_START, MOVES, START, UNK, advance, features, header

STORE = paths.DATA / "heldout-2026-09/store/2026-09"
WEIGHTS = paths.DATA / "release-2.1/export"
COLUMNS = ["site", "white_elo", "black_elo", "white_title", "black_title", "format", "rated_prefix", "base",
           "increment", "termination", "result", "moves", "clocks", "evals", "end", "opening"]  # fmt: skip
FORMATS = ("bullet", "blitz", "rapid", "classical")
PIECES = "PNBRQK"


def games(n, seed=0, keep=lambda g: True, formats=(1, 2, 3, 4)):
    """About n rated human games with full clocks, drawn in proportion to the store's buckets."""
    buckets = [b for b in json.loads((STORE / "buckets.json").read_text()) if b["format"] in formats]
    rate = n / sum(b["games"] for b in buckets)
    out, rng = [], np.random.default_rng(seed)
    for b in buckets:
        take = rng.binomial(b["games"], min(1.0, 8 * rate))  # filters below drop many; thinned at the end
        if not take:
            continue
        t = pq.ParquetFile(STORE / b["shards"][0]).read_row_group(0, columns=COLUMNS).slice(0, take)
        for g in t.to_pylist():
            if (
                g["rated_prefix"] and "BOT" not in (g["white_title"], g["black_title"])
                and g["base"] > 0 and 20 <= len(g["moves"]) <= 300 and len(g["clocks"]) == len(g["moves"])
                and keep(g)
            ):  # fmt: skip
                g["shard"] = b["shards"][0]
                out.append(g)
    return [out[i] for i in rng.permutation(len(out))[:n]]


def arrays(g, welo=None, belo=None):
    """tokens, clock features and board states up to the last move's position; one per position plus the header."""
    moves = [m + MOVE_START for m in g["moves"]]
    toks = header(g["base"], g["increment"], welo or g["white_elo"], belo or g["black_elo"]) + moves[:-1]
    feats = [
        features(p - HEADER + 1, g["base"], g["increment"], g["clocks"]) if p >= HEADER - 1 else [-1] * 3
        for p in range(len(toks))
    ]
    boards = [START]
    while len(boards) < len(toks):
        p = len(boards)
        boards.append(advance(boards[-1], toks[p]) if p >= HEADER else START)
    b = np.frombuffer(b"".join(boards), np.uint8).reshape(-1, 68)
    return torch.tensor(toks), torch.tensor(feats, dtype=torch.float32), torch.tensor(b)


def capture(model, keep_scores=False):
    """Records each routed block's chosen experts (and its router's input, scores and gates) on every forward."""
    log, mlp = {}, model.mlp

    def wrapped(i, h):
        if f"{i}.fc" not in model.w:
            w = model.w
            z = F.linear(h.float() - w[f"{i}.mu"], w[f"{i}.router"])
            idx = torch.topk(torch.sigmoid(z) + w[f"{i}.moe_bias"], model.topk, dim=-1).indices
            log[i] = idx.cpu()
            if keep_scores:
                log[i, "z"] = z.cpu()
                log[i, "h"] = (h.float() - w[f"{i}.mu"]).cpu()
                log[i, "gate"] = (model.topk**0.5 * torch.softmax(F.logsigmoid(z.gather(1, idx)), -1)).cpu()
        return mlp(i, h)

    model.mlp = wrapped
    return log


def batches(gs, limit=3000):
    b, n = [], 0
    for g in gs:
        m = HEADER + len(g["moves"]) - 1
        if b and n + m > limit:
            yield b
            b, n = [], 0
        b.append(g)
        n += m
    if b:
        yield b


def binned(v, edges):
    return int(np.searchsorted(edges, v, side="right"))


PLY = [10, 20, 40, 60, 80, 120]
COUNT = [5, 9, 13, 17, 21, 25, 29]
LEFT = [5, 10, 20, 30, 60, 120, 300, 900]
FRAC = [0.05, 0.1, 0.25, 0.5, 0.75]
THINK = [2, 4, 8, 16, 32, 64]
ELO = [1000, 1400, 1800, 2200, 2600]
NAMES = (
    ["header"]
    + [f"side={s}" for s in "WB"]
    + [f"ply<{e}" for e in PLY] + ["ply>=120"]
    + [f"pieces<{e}" for e in COUNT] + ["pieces>=29"]
    + [f"queens={q}" for q in range(3)]
    + [f"left<{e}" for e in LEFT] + ["left>=900"]
    + [f"frac<{e}" for e in FRAC] + ["frac>=0.75"]
    + [f"prevthink<{e}" for e in THINK] + ["prevthink>=64", "prevthink=none"]
    + [f"nextthink<{e}" for e in THINK] + ["nextthink>=64"]
    + [f"format={f}" for f in FORMATS]
    + [f"elo<{e}" for e in ELO] + ["elo>=2600"]
    + ["check", "prevcapture", "nextcapture", "nextcastle", "nextpromo"]
    + [f"result={r}" for r in ("win", "draw", "loss")]
    + [f"nextpiece={p}" for p in PIECES]
    + [f"to={chess.SQUARE_NAMES[s]}" for s in range(64)]
    + [f"from={chess.SQUARE_NAMES[s]}" for s in range(64)]
    + [f"{p}@{chess.SQUARE_NAMES[s]}" for p in PIECES + PIECES.lower() for s in range(64)]
)  # fmt: skip
INDEX = {n: i for i, n in enumerate(NAMES)}


def position_features(g, boards):
    """Multi-hot [tokens, features] of a game's arrays, in NAMES."""
    n, base = len(g["moves"]), g["base"]
    rows, b, prev_capture = [[INDEX["header"]] for _ in range(HEADER - 1)], chess.Board(), False
    res = g["result"]
    for k, token in enumerate(g["moves"]):
        white = k % 2 == 0
        c = g["clocks"]
        left = (base if k < 2 else c[k - 2]) if base else None
        prev = (c[k - 4] - c[k - 2] + g["increment"]) if k >= 4 else None
        think = ((base if k < 2 else c[k - 2]) - c[k] + g["increment"]) if k >= 2 else None
        move = chess.Move.from_uci(MOVES[token])
        kind = b.piece_type_at(move.from_square)
        elo = g["white_elo"] if white else g["black_elo"]
        fs = [
            "side=W" if white else "side=B",
            f"format={FORMATS[g['format'] - 1]}",
            (f"ply<{PLY[binned(k, PLY)]}" if k < PLY[-1] else "ply>=120"),
            (f"elo<{ELO[binned(elo, ELO)]}" if elo < ELO[-1] else "elo>=2600"),
            f"nextpiece={PIECES[kind - 1]}",
            f"to={chess.SQUARE_NAMES[move.to_square]}",
            f"from={chess.SQUARE_NAMES[move.from_square]}",
        ]
        count = chess.popcount(b.occupied)
        fs.append(f"pieces<{COUNT[binned(count, COUNT)]}" if count < COUNT[-1] else "pieces>=29")
        fs.append(f"queens={chess.popcount(b.queens)}" if chess.popcount(b.queens) < 3 else "queens=2")
        fs.append(f"left<{LEFT[binned(left, LEFT)]}" if left < LEFT[-1] else "left>=900")
        frac = left / base
        fs.append(f"frac<{FRAC[binned(frac, FRAC)]}" if frac < FRAC[-1] else "frac>=0.75")
        fs.append("prevthink=none" if prev is None or prev < 0 else
                  (f"prevthink<{THINK[binned(prev, THINK)]}" if prev < THINK[-1] else "prevthink>=64"))  # fmt: skip
        if think is not None and think >= 0:
            fs.append(f"nextthink<{THINK[binned(think, THINK)]}" if think < THINK[-1] else "nextthink>=64")
        if b.is_check():
            fs.append("check")
        if k and prev_capture:
            fs.append("prevcapture")
        capture_now = b.is_capture(move)
        if capture_now:
            fs.append("nextcapture")
        if b.is_castling(move):
            fs.append("nextcastle")
        if move.promotion:
            fs.append("nextpromo")
        if res == 2:
            fs.append("result=draw")
        elif res in (0, 1):
            fs.append("result=win" if (res == 0) == white else "result=loss")
        ids = [INDEX[f] for f in fs]
        for s, p in b.piece_map().items():
            ids.append(INDEX[f"{p.symbol()}@{chess.SQUARE_NAMES[s]}"])
        rows.append(ids)
        prev_capture = capture_now
        b.push(move)
    assert len(rows) == HEADER - 1 + n == len(boards)
    return rows


def load():
    """The PyTorch reference: BF16 on a GPU, int8 weights on a CPU, as Allie.from_pretrained loads them."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Model(WEIGHTS, device, torch.bfloat16, None, device == "cpu", "torch")


def scan(args):
    model = load()
    log = capture(model)
    gs = games(args.games, args.seed)
    print(len(gs), "games", sum(len(g["moves"]) for g in gs), "plies", flush=True)
    blocks = [i for i in range(model.layers) if f"{i}.fc" not in model.w]
    C = torch.zeros(2, len(blocks), 256, len(NAMES), dtype=torch.float64, device=model.device)
    N = torch.zeros(2, len(NAMES), dtype=torch.float64, device=model.device)
    overlap = np.zeros((len(blocks), 4))  # shared experts with the next token, pairs; with the same side's next, pairs
    for bi, batch in enumerate(batches(gs)):
        items, rows, split = [], [], []
        for g in batch:
            x = arrays(g)
            items.append((Cache(model), *x))
            rows += position_features(g, x[2])
            split += [zlib.crc32(g["site"].encode()) & 1] * len(x[0])
        step(model, items, every=True)
        T = len(rows)
        fr = torch.tensor([t for t, r in enumerate(rows) for _ in r], device=model.device)
        fc = torch.tensor([f for r in rows for f in r], device=model.device)
        F_ = torch.zeros(T, len(NAMES), device=model.device, dtype=torch.float32)
        F_[fr, fc] = 1
        sp = torch.tensor(split, device=model.device)
        for s in (0, 1):
            N[s] += F_[sp == s].sum(0).double()
        for j, i in enumerate(blocks):
            idx = log[i].to(model.device)
            hot = torch.zeros(T, 256, device=model.device, dtype=torch.float32).scatter_(1, idx, 1)
            for s in (0, 1):
                C[s, j] += (hot[sp == s].T @ F_[sp == s]).double()
            lo = 0
            for it in items:
                m = len(it[1])
                h = hot[lo + HEADER - 1 : lo + m]
                overlap[j] += [(h[1:] * h[:-1]).sum().item(), len(h) - 1, (h[2:] * h[:-2]).sum().item(), len(h) - 2]
                lo += m
        log.clear()
        if bi % 20 == 0:
            print(bi, flush=True)
    np.savez_compressed(args.out, C=C.cpu().numpy(), N=N.cpu().numpy(), overlap=overlap, blocks=blocks,
                        names=np.array(NAMES), games=len(gs))  # fmt: skip


def legal(b, z):
    moves = list(b.legal_moves)
    ids = [MOVE_ID[m.uci()] for m in moves]
    p = torch.softmax(z[ids].double(), 0).numpy()
    return moves, p


def logits(model, items):
    return step(model, items, every=True).float().cpu()


def pick(args):
    model = load()
    tcs = {(180, 0), (180, 2), (300, 0), (300, 3), (600, 0)}
    keep = lambda g: (g["base"], g["increment"]) in tcs and 1300 <= min(g["white_elo"], g["black_elo"]) \
        and max(g["white_elo"], g["black_elo"]) <= 2300 and 50 <= len(g["moves"]) <= 120 and g["termination"] in (0, 1)  # fmt: skip
    gs = games(args.games, args.seed, keep, formats=(2, 3))
    print(len(gs), "games", flush=True)
    out = open(args.out, "w")
    for batch in batches(gs, 2000):
        zs = []
        for elo in (None, 1000, 2500):
            items = [(Cache(model), *arrays(g, elo, elo)) for g in batch]
            zs.append(logits(model, items))
        lo = 0
        for g in batch:
            m = HEADER + len(g["moves"]) - 1
            real, low, high = (z[lo + HEADER - 1 : lo + m] for z in zs)
            lo += m
            b, plies = chess.Board(), []
            ce = top = 0.0
            for k, token in enumerate(g["moves"]):
                played = chess.Move.from_uci(MOVES[token])
                moves, p = legal(b, real[k])
                _, pl = legal(b, low[k])
                _, ph = legal(b, high[k])
                j = moves.index(played)
                ce -= np.log(p[j])
                top += p.argmax() == j
                tv = 0.5 * np.abs(pl - ph).sum()
                plies.append(dict(k=k, tv=round(float(tv), 3), real=[b.san(moves[p.argmax()]), round(float(p.max()), 3)],
                                  low=[b.san(moves[pl.argmax()]), round(float(pl.max()), 3)],
                                  high=[b.san(moves[ph.argmax()]), round(float(ph.max()), 3)],
                                  played=[b.san(played), round(float(p[j]), 3)],
                                  w=round(float(torch.softmax(real[k][WDL].double(), 0)[0]), 3)))  # fmt: skip
                b.push(played)
            best = sorted(plies, key=lambda x: -x["tv"])[:5]
            out.write(json.dumps(dict(site=g["site"], shard=g["shard"], white=g["white_elo"], black=g["black_elo"], tc=[g["base"], g["increment"]],
                                      result=g["result"], term=g["termination"], end=g["end"], opening=g["opening"], n=len(g["moves"]),
                                      ce=round(ce / len(g["moves"]), 3), top1=round(top / len(g["moves"]), 3), clocks=g["clocks"],
                                      san=[p["played"][0] for p in plies], best=best, plies=plies)) + "\n")  # fmt: skip
        out.flush()


def lesion_mlp(model, mode):
    """The block's MLP with its routing changed: "random" experts, the best int experts only, "shared" alone."""

    def mlp(i, h):
        if f"{i}.fc" in model.w:
            return model.ffn(h, f"{i}.fc", f"{i}.proj")
        w, k = model.w, model.topk
        z = F.linear(h.float() - w[f"{i}.mu"], w[f"{i}.router"])
        idx = torch.topk(torch.sigmoid(z) + w[f"{i}.moe_bias"], k, dim=-1).indices
        gate = k**0.5 * torch.softmax(F.logsigmoid(z.gather(1, idx)), -1)
        shared = model.ffn(h, f"{i}.shared_up", f"{i}.shared_down")
        if mode == "shared":
            return shared
        if mode == "random":
            idx = torch.rand(len(h), model.w[f"{i}.router"].shape[0], device=h.device).topk(k, dim=-1).indices
        else:
            idx, gate = idx[:, :mode], gate[:, :mode]
        model.keep = idx.shape[1]
        return model.group(i, h, idx, gate.to(h.dtype).float()).to(h.dtype) + shared

    return mlp


def attention_mask(allowed):
    """Restricts every attention to keys allowed(t, n) says, [t, n] booleans over the game's tokens."""
    sdpa = F.scaled_dot_product_attention

    def wrapped(q, k, v, attn_mask=None, scale=None):
        return sdpa(q, k, v, attn_mask=attn_mask & allowed(q.shape[-2], k.shape[-2], q.device), scale=scale)

    F.scaled_dot_product_attention = wrapped
    return lambda: setattr(F, "scaled_dot_product_attention", sdpa)


def lesion(args):
    model = load()
    gs = games(args.games, args.seed)
    elos = np.array([e for g in gs for e in (g["white_elo"], g["black_elo"])])
    median = int(np.median(elos))
    plies = sum(len(g["moves"]) for g in gs)
    print(len(gs), "games", plies, "plies, median rating", median, flush=True)
    legal_ids, played = [], []
    for g in gs:
        b, row = chess.Board(), []
        for token in g["moves"]:
            row.append(np.array([MOVE_ID[m.uci()] for m in b.legal_moves]))
            b.push(chess.Move.from_uci(MOVES[token]))
        legal_ids.append(row)
        played.append(np.array(g["moves"]) + MOVE_START)

    def plain(g):
        return arrays(g)

    def no_rating(g):
        return arrays(g, median, median)

    def no_clock(g):
        t, f, b = arrays(g)
        return t, torch.full_like(f, -1), b

    def no_board(g):
        t, f, b = arrays(g)
        b = b.clone()
        b[:, 67] = 0
        return t, f, b

    def no_control(g):
        t, f, b = arrays(g)
        t = t.clone()
        t[1], t[2] = UNK, UNK
        return t, f, b

    def header_and(m):
        return lambda t, n, dev: (torch.arange(n, device=dev)[None] < HEADER) | (
            torch.arange(n, device=dev)[None] >= torch.arange(t, device=dev)[:, None] - m + 1
        )

    self_only = lambda t, n, dev: torch.arange(n, device=dev)[None] == torch.arange(t, device=dev)[:, None]
    cases = dict(
        full=dict(),
        random_experts=dict(mlp="random"),
        best_8=dict(mlp=8),
        best_4=dict(mlp=4),
        best_1=dict(mlp=1),
        shared_only=dict(mlp="shared"),
        no_ratings=dict(data=no_rating),
        no_clock=dict(data=no_clock),
        no_time_control=dict(data=no_control),
        no_board=dict(data=no_board),
        last_8_plies=dict(att=header_and(8)),
        last_2_plies=dict(att=header_and(2)),
        no_attention=dict(att=self_only),
    )
    # slices of positions: the mover's time left, the first plies, the mover's rating
    slices = {"under 30 s": lambda g, k, left, elo: left < 30, "first 10 plies": lambda g, k, left, elo: k < 10,
              "under 1400": lambda g, k, left, elo: elo < 1400, "2400+": lambda g, k, left, elo: elo >= 2400}  # fmt: skip
    masks = []
    for g in gs:
        c = [g["base"], g["base"], *g["clocks"]]
        masks.append(
            [
                [f(g, k, c[k], g["white_elo"] if k % 2 == 0 else g["black_elo"]) for f in slices.values()]
                for k in range(len(g["moves"]))
            ]
        )
    original = model.mlp
    out, base, base_sl = {}, None, None
    for name, c in cases.items():
        model.keep = model.topk  # the best-k cases shrink it
        model.mlp = lesion_mlp(model, c["mlp"]) if "mlp" in c else original
        restore = attention_mask(c["att"]) if "att" in c else None
        sums, sl, g0 = [], np.zeros((len(gs), len(slices), 2)), 0
        for batch in batches(gs):
            z = logits(model, [(Cache(model), *c.get("data", plain)(g)) for g in batch]).numpy()
            lo = 0
            for j, g in enumerate(batch):
                n = len(g["moves"])
                ce = top = raw = 0.0
                for k in range(n):
                    row, ids = z[lo + HEADER - 1 + k], legal_ids[g0 + j][k]
                    q = row[ids]
                    q = q - np.logaddexp.reduce(q)
                    i = np.flatnonzero(ids == played[g0 + j][k])[0]
                    ce -= q[i]
                    top += q.argmax() == i
                    m = row[MOVE_START : MOVE_START + len(MOVES)]
                    raw -= row[played[g0 + j][k]] - np.logaddexp.reduce(m)
                    mask = np.array(masks[g0 + j][k], float)
                    sl[g0 + j, :, 0] += mask * -q[i]
                    sl[g0 + j, :, 1] += mask
                sums.append((ce, raw, top, n))
                lo += HEADER + n - 1
            g0 += len(batch)
        if restore:
            restore()
        a = np.array(sums)
        if base is None:
            base, base_sl = a, sl
        rng = np.random.default_rng(1)
        boots = rng.integers(0, len(a), (2000, len(a)))
        rate = lambda x, i: x[boots, i].sum(1) / x[boots, 3].sum(1)
        row = dict(ce=a[:, 0].sum() / plies, raw=a[:, 1].sum() / plies, top1=a[:, 2].sum() / plies)
        for key, i in (("ce", 0), ("raw", 1), ("top1", 2)):
            d = rate(a, i) - rate(base, i)
            row[key + "_diff"] = float(a[:, i].sum() / plies - base[:, i].sum() / plies)
            row[key + "_ci"] = [float(x) for x in np.percentile(d, [2.5, 97.5])]
        row["slices"] = {}
        for j, s_name in enumerate(slices):
            r = lambda x: x[boots, j, 0].sum(1) / x[boots, j, 1].sum(1)
            d = r(sl) - r(base_sl)
            row["slices"][s_name] = dict(n=int(sl[:, j, 1].sum()), ce=float(sl[:, j, 0].sum() / sl[:, j, 1].sum()),
                                         diff=float(sl[:, j, 0].sum() / sl[:, j, 1].sum() - base_sl[:, j, 0].sum() / base_sl[:, j, 1].sum()),
                                         ci=[float(x) for x in np.percentile(d, [2.5, 97.5])])  # fmt: skip
        out[name] = row
        print(name, json.dumps(row), flush=True)
        json.dump(dict(games=len(gs), plies=plies, median=median, cases=out), open(args.out, "w"), indent=1)


def game_ces(model, gs, data=arrays):
    """Per game, the CE over legal moves of each move played."""
    out = []
    for batch in batches(gs):
        z = logits(model, [(Cache(model), *data(g)) for g in batch]).numpy()
        lo = 0
        for g in batch:
            b, row = chess.Board(), []
            for k, token in enumerate(g["moves"]):
                moves = [m.uci() for m in b.legal_moves]
                q = z[lo + HEADER - 1 + k][[MOVE_ID[u] for u in moves]]
                row.append(np.logaddexp.reduce(q) - q[moves.index(MOVES[token])])
                b.push(chess.Move.from_uci(MOVES[token]))
            out.append(np.array(row))
            lo += HEADER + len(g["moves"]) - 1
    return out


def ratings(args):
    """Is a wrong rating costlier for strong players, or for players far from the median? The lesion's "no_ratings" case
    on its own games, cut by the mover's rating, with the mean distance from the median of each cut."""
    model = load()
    gs = games(args.games, args.seed)
    median = int(np.median([e for g in gs for e in (g["white_elo"], g["black_elo"])]))
    real = game_ces(model, gs)
    wrong = game_ces(model, gs, lambda g: arrays(g, median, median))
    elo = [np.array([g["white_elo"] if k % 2 == 0 else g["black_elo"] for k in range(len(g["moves"]))]) for g in gs]
    boots = np.random.default_rng(1).integers(0, len(gs), (2000, len(gs)))
    rows = []
    for lo, hi in ((0, 1000), (1000, 1400), (1400, 1800), (1800, 2200), (2200, 2400), (2400, 9999)):
        masks = [(lo <= e) & (e < hi) for e in elo]
        d = np.array([(w - r)[m].sum() for w, r, m in zip(wrong, real, masks)])
        n = np.array([m.sum() for m in masks])
        ci = np.percentile(d[boots].sum(1) / n[boots].sum(1), [2.5, 97.5])
        far = sum(np.abs(e[m] - median).sum() for e, m in zip(elo, masks)) / n.sum()
        rows.append(dict(band=[lo, hi], n=int(n.sum()), distance=round(float(far)), diff=float(d.sum() / n.sum()),
                         ci=[float(x) for x in ci]))  # fmt: skip
        print(rows[-1], flush=True)
    json.dump(
        dict(games=len(gs), plies=int(sum(map(len, real))), median=median, bands=rows), open(args.out, "w"), indent=1
    )


def recapture(args):
    """Does the clock sweep's shape generalise? Every position of a sample of 3+x and 5+x blitz games where the mover can
    recapture on the square just captured on, from ply 10, at least 20 s left, the recapture at 5% to 60% at the real clock;
    only the mover's time left changes, as in clock_sweep. Points are probability on all moves to that square."""
    model = load()
    gs = games(args.games, args.seed, lambda g: g["base"] in (180, 300), formats=(2,))
    seconds = (180, 60, 30, 10, 5, 2)
    rows = []

    def on_square(z, b, sq):
        moves, q = legal(b, z)
        return float(sum(x for m, x in zip(moves, q) if m.to_square == sq))

    for gi, g in enumerate(gs):
        moves = [chess.Move.from_uci(MOVES[t]) for t in g["moves"]]
        toks, feats, boards = arrays(g)
        cache, b, prev_capture = Cache(model), chess.Board(), False
        for k, m in enumerate(moves):
            p = HEADER - 1 + k
            z = (
                step(model, [(cache, toks[cache.n : p + 1], feats[cache.n : p + 1], boards[cache.n : p + 1])])[0]
                .float()
                .cpu()
            )
            sq = moves[k - 1].to_square if k else None
            if k >= 10 and prev_capture and feats[p, 0] >= 20 and any(x.to_square == sq for x in b.legal_moves):
                real = on_square(z, b, sq)
                if 0.05 <= real <= 0.6:
                    row = []
                    for s in seconds:
                        cache.truncate(p)
                        f = feats[p : p + 1].clone()
                        f[0, 0] = s
                        row.append(
                            on_square(
                                step(model, [(cache, toks[p : p + 1], f, boards[p : p + 1])])[0].float().cpu(), b, sq
                            )
                        )
                    cache.truncate(p)
                    step(model, [(cache, toks[p : p + 1], feats[p : p + 1], boards[p : p + 1])])
                    rows.append(row + [real, float(feats[p, 0])])
            prev_capture = b.is_capture(m)
            b.push(m)
        if (gi + 1) % 100 == 0:
            print(gi + 1, "games", len(rows), "positions", np.mean(rows, 0)[:6].round(4).tolist(), flush=True)
    r = np.array(rows)
    rise, drop = r[:, 3] - r[:, 0], r[:, 5] - r[:, 3]
    rng = np.random.default_rng(0)
    ci = lambda x: [float(q) for q in np.percentile(x[rng.integers(0, len(x), (2000, len(x)))].mean(1), [2.5, 97.5])]
    out = dict(games=len(gs), positions=len(r), seconds=seconds, mean=r[:, :6].mean(0).round(4).tolist(), rise_10s=float(rise.mean()),
               rise_ci=ci(rise), rise_share=float((rise > 0).mean()), drop_2s=float(drop.mean()), drop_ci=ci(drop),
               drop_share=float((drop < 0).mean()), both_share=float(((rise > 0) & (drop < 0)).mean()))  # fmt: skip
    json.dump(out, open(args.out, "w"), indent=1)
    print(out, flush=True)


def attention_log():
    """Records every attention's weights while installed, as the reference's scaled_dot_product_attention sees them."""
    log, sdpa = [], F.scaled_dot_product_attention

    def wrapped(q, k, v, attn_mask=None, scale=None):
        w = (q @ k.transpose(-1, -2)).float() * scale
        if attn_mask is not None:
            w = w.masked_fill(~attn_mask, float("-inf"))
        log.append(torch.softmax(w, -1).half().cpu())
        return sdpa(q, k, v, attn_mask=attn_mask, scale=scale)

    F.scaled_dot_product_attention = wrapped
    return log, lambda: setattr(F, "scaled_dot_product_attention", sdpa)


def outputs(g, z):
    """Per position: legal moves by probability, the played move's rank and probability, result and think outputs."""
    b, rows = chess.Board(), []
    for k, token in enumerate(g["moves"]):
        played = chess.Move.from_uci(MOVES[token])
        moves, p = legal(b, z[k])
        order = np.argsort(-p)
        j = moves.index(played)
        t = torch.softmax(z[k][TIME].double(), 0).numpy()
        rows.append(dict(k=k, fen=b.fen(), san=b.san(played), uci=played.uci(), p=float(p[j]), rank=int((p > p[j]).sum()) + 1,
                         n=len(moves), top=[[b.san(moves[i]), moves[i].uci(), round(float(p[i]), 5)] for i in order[:12]],
                         wdl=[round(float(x), 5) for x in torch.softmax(z[k][WDL].double(), 0)],
                         think=[round(float(x), 5) for x in t], expect=float(t @ CENTRES)))  # fmt: skip
        b.push(played)
    return rows


def sweep(model, g, ply):
    """Legal-move probabilities at one position as only its mover's rating varies."""
    b, out = chess.Board(), {}
    for token in g["moves"][:ply]:
        b.push(chess.Move.from_uci(MOVES[token]))
    for r in range(800, 2801, 100):
        z = logits(model, [(Cache(model), *(arrays(g, r, None) if ply % 2 == 0 else arrays(g, None, r)))])
        moves, p = legal(b, z[HEADER - 1 + ply])
        out[r] = {m.uci(): round(float(q), 4) for m, q in zip(moves, p) if q >= 0.002}
    return out


def clock_sweep(model, g, ply, seconds=(2, 5, 10, 20, 30, 60, 120, 180, 300, 600)):
    """Legal-move probabilities at one position as only the mover's time left changes, at its own token."""
    toks, feats, boards = arrays(g)
    p, cache, b, out = HEADER - 1 + ply, Cache(model), chess.Board(), {}
    step(model, [(cache, toks[:p], feats[:p], boards[:p])])
    for token in g["moves"][:ply]:
        b.push(chess.Move.from_uci(MOVES[token]))
    for s in seconds:
        f = feats[p : p + 1].clone()
        f[0, 0] = s
        z = step(model, [(cache, toks[p : p + 1], f, boards[p : p + 1])])[0].float().cpu()
        cache.truncate(p)
        moves, q = legal(b, z)
        out[s] = {m.uci(): round(float(x), 4) for m, x in zip(moves, q) if x >= 0.002}
    return out


def heads(args):
    model = load()
    gs = games(args.games, args.seed)
    keys = ("mover", "other", "control", "last", "own_last", "header")
    total, count = np.zeros((len(keys), model.layers, model.heads)), 0
    digits, even = np.zeros((4, model.layers, model.heads)), np.zeros(2)
    for g in gs:
        att, restore = attention_log()
        logits(model, [(Cache(model), *arrays(g))])
        restore()
        a = torch.stack(att).float().numpy()  # [block, head, token, key]
        for t in range(HEADER + 3, a.shape[2]):
            white = (t - (HEADER - 1)) % 2 == 0
            mover, other = (range(3, 7), range(7, 11)) if white else (range(7, 11), range(3, 7))
            for i, ks in enumerate((mover, other, range(1, 3), [t], [t - 1], range(HEADER))):
                total[i] += a[:, :, t, list(ks)].sum(-1)
            digits += a[:, :, t, list(mover)].transpose(2, 0, 1)
            even += (HEADER / (t + 1), 8 / (t + 1))
            count += 1
    m = total / count
    out = {k: m[i].round(4).tolist() for i, k in enumerate(keys)}
    out |= dict(
        games=len(gs),
        positions=count,
        header_share=float(m[5].mean()),
        rating_share=float((m[0] + m[1]).mean()),
        even_header=float(even[0] / count),
        even_rating=float(even[1] / count),
        mover_digits=(digits / count).round(4).tolist(),
        xsa=model.xsa.float().cpu().numpy().round(4).tolist(),
    )
    json.dump(out, open(args.out, "w"))
    print(
        json.dumps(
            {k: out[k] for k in ("games", "positions", "header_share", "rating_share", "even_header", "even_rating")}
        ),
        flush=True,
    )


def one_game(args):
    model = load()
    t = pq.read_table(STORE / args.shard, columns=COLUMNS).to_pylist()
    g = next(x for x in t if x["site"].endswith(args.site))
    x = arrays(g)
    log = capture(model, keep_scores=True)
    att, restore = attention_log()
    z = logits(model, [(Cache(model), *x)])
    restore()
    blocks = sorted(k for k in log if isinstance(k, int))
    idx = torch.stack([log[i] for i in blocks]).numpy().astype(np.uint8)
    gate = torch.stack([log[i, "gate"] for i in blocks]).numpy().astype(np.float16)
    scores = torch.stack([log[i, "z"] for i in blocks]).numpy().astype(np.float16)
    bias = torch.stack([model.w[f"{i}.moe_bias"].cpu() for i in blocks]).numpy()
    router_input = torch.stack([log[i, "h"] for i in blocks]).numpy().astype(np.float16)
    attn = torch.stack(att).numpy()  # [block, head, token, key]
    parts = dict(
        embed=F.embedding(x[0].to(model.device), model.w["embed"]).float().cpu().numpy().astype(np.float16),
        clock=model.clock(x[1].to(model.device)).float().cpu().numpy().astype(np.float16),
        board=model.board(x[2].to(model.device)).float().cpu().numpy().astype(np.float16),
    )
    np.savez_compressed(args.out, idx=idx, gate=gate, scores=scores, bias=bias, attn=attn, blocks=blocks, router_input=router_input,
                        tokens=x[0].numpy(), feats=x[1].numpy(), boards=x[2].numpy(), **parts)  # fmt: skip
    del model.mlp
    keys = "site white_elo black_elo base increment termination result end opening clocks".split()
    json.dump(dict(game={k: g[k] for k in keys}, shard=args.shard, ply=args.ply, plies=outputs(g, z[HEADER - 1 :]),
                   rating=sweep(model, g, args.ply), clock=clock_sweep(model, g, args.ply)), open(args.out + ".json", "w"))  # fmt: skip


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("scan", scan), ("pick", pick), ("lesion", lesion), ("heads", heads), ("ratings", ratings), ("recapture", recapture),
    ):  # fmt: skip
        s = sub.add_parser(name)
        s.add_argument("out")
        s.add_argument("--games", type=int, default=12000)
        s.add_argument("--seed", type=int, default=20261103)
        s.set_defaults(fn=fn)
    s = sub.add_parser("game")
    s.add_argument("out")
    s.add_argument("--site", required=True)
    s.add_argument("--shard", required=True)
    s.add_argument("--ply", type=int, required=True, help="the position for the rating and clock sweeps")
    s.set_defaults(fn=one_game)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
