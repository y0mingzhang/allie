"""Human think times, resignations, draws and flags against Allie 2.0's heads, on held-out games.

    python analysis/lichess/human.py score --model DIR --out DIR   # GPU: heads at every position
    python analysis/lichess/human.py fit --data DIR                  # CPU: hazards and comparisons

Games: the main evaluation's (July 2026, never trained on; its build already left out the dev and
test splits, which nothing here opens), found in data-v1 by token hash; both players human, every
clock known, ended normally or on time. Up to --per-cell games per format x average-Elo band.
"""

import argparse
import json
from pathlib import Path

import chess
import numpy as np
import pyarrow.parquet as pq
import torch

from allie.data import mix as cm
from allie.data.store import hash64
from allie.data.vocab import BOS, TERM_NORMAL, TERM_OTHER
from allie.eval import build as ev
from allie.lichess.model import Cache, Model, step
from allie.lichess.tokens import HEADER, MOVES, START, advance, features

TIME, WDL = slice(2350, 2413), slice(2413, 2416)
CENTRES = np.r_[np.arange(16), 16 * np.exp(np.arange(47) / 7.06)]
FORMATS = ["bullet", "blitz", "rapid", "classical"]
BANDS = ["<1400", "1400-2000", "2000-2400", ">=2400"]


def eval_games():
    """The token hashes (data.store.hash64) of the complete games in the main evaluation's
    rows, which its build already cleared of the dev and test splits."""
    with np.load(ev.OUT / "strat.npz") as z:
        rows = z["rows"].astype(np.int64)
    hashes = set()
    for r in rows:
        start = np.flatnonzero((r == BOS) & (np.r_[r[1:], BOS] != BOS))
        for a in start:
            end = np.flatnonzero((r[a:] == TERM_NORMAL) | (r[a:] == TERM_OTHER))
            if len(end):  # games cut at the row's end have no termination token: skipped
                hashes.add(hash64(r[a : a + end[0] + 1]))
    return hashes


def select(per_cell, per_bucket=None):
    """The evaluation's games between two humans, with every clock, ended normally or on time:
    up to per_cell per format x average-Elo band."""
    want = eval_games()
    buckets = [b for b in json.loads((ev.MONTH / "buckets.json").read_text())
               if 1 <= b["code"] // 10000 - 1 <= 4]  # fmt: skip
    cols = cm.COLUMNS + ["clocks", "token_hash", "result", "site", "end"]
    picked, counts = [], np.zeros((4, 4), int)
    for b in buckets:
        f = b["code"] // 10000 - 2
        for shard in b["shards"]:
            h = pq.read_table(ev.MONTH / shard, columns=["token_hash"]).column("token_hash").to_numpy()
            hit = np.flatnonzero(np.isin(h, np.fromiter(want, np.uint64)))
            if not len(hit):
                continue
            t = pq.read_table(ev.MONTH / shard, columns=cols).take(hit)
            g = cm.Games(t, clock=True, aux=True)
            site, end = t.column("site").to_pylist(), t.column("end").to_numpy()
            for i in range(len(hit)):
                moves = g.off[i + 1] - g.off[i]
                c = g.cval[g.coff[i] : g.coff[i + 1]]
                band = int(np.searchsorted(ev.UPPER, (g.welo[i] + g.belo[i]) // 2, side="right"))
                if not (not g.wbot[i] and not g.bbot[i] and g.base[i] > 0 and len(c) == moves
                        and g.term[i] in (0, 1) and g.result[i] in (0, 1, 2) and 4 <= moves <= 1000
                        and counts[f, band] < per_cell):  # fmt: skip
                    continue
                toks, _ = g.tokens(i, True, True)
                picked.append(dict(
                    site=site[i], fmt=f, band=band, welo=int(g.welo[i]), belo=int(g.belo[i]),
                    base=int(g.base[i]), inc=int(g.inc[i]), term=int(g.term[i]),
                    result=int(g.result[i]), end=int(end[i]), tokens=toks[:-1].astype(np.int64),
                    clocks=c.astype(np.int64),
                ))  # fmt: skip
                counts[f, band] += 1
    return picked, counts


def feats(g):
    """Clock features of every position, the one after the last move included."""
    n, f = len(g["tokens"]) - HEADER, features
    head = [[-1] * 3] * (HEADER - 1)
    return head + [f(k, g["base"], g["inc"], g["clocks"].tolist()) for k in range(n + 1)]


def boards(tokens):
    b = [START] * HEADER
    for t in tokens[HEADER:]:
        b.append(advance(b[-1], int(t)))
    return np.frombuffer(b"".join(b), np.uint8).reshape(-1, 68)


def score(a):
    games, counts = select(a.per_cell)
    print("games per format x band", counts.tolist(), flush=True)
    if a.threads:
        torch.set_num_threads(a.threads)
    m = Model(a.model, a.device, torch.bfloat16, int8=a.int8, backend="torch")
    out = {k: [] for k in ("game", "k", "wdl", "time", "legal")}
    lo = 0
    while lo < len(games):
        hi, total = lo, 0
        while hi < len(games) and (
            hi == lo or total + len(games[hi]["tokens"]) <= a.tokens
        ):
            total += len(games[hi]["tokens"])
            hi += 1
        items = [(Cache(m, len(g["tokens"])), torch.as_tensor(g["tokens"]),
                  torch.as_tensor(feats(g), dtype=torch.float32), torch.as_tensor(boards(g["tokens"])))
                 for g in games[lo:hi]]  # fmt: skip
        z = step(m, items, every=True).float()
        at = 0
        for j, g in zip(range(lo, hi), games[lo:hi]):
            n = len(g["tokens"]) - HEADER  # moves; positions predicting move k = 0..n
            zz = z[at + HEADER - 1 : at + len(g["tokens"])]
            at += len(g["tokens"])
            board, legal = chess.Board(), []
            for t in g["tokens"][HEADER:]:
                legal.append(board.legal_moves.count())
                board.push_uci(MOVES[t - 378])
            legal.append(board.legal_moves.count())
            out["game"].append(np.full(n + 1, j))
            out["k"].append(np.arange(n + 1))
            out["wdl"].append(torch.softmax(zz[:, WDL], -1).cpu().numpy())
            out["time"].append(torch.softmax(zz[:, TIME], -1).half().cpu().numpy())
            out["legal"].append(np.array(legal))
        lo = hi
        print(f"{hi} / {len(games)} games", flush=True)
    path = Path(a.out)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "positions.npz", **{k: np.concatenate(v) for k, v in out.items()})
    keys = ("fmt", "band", "welo", "belo", "base", "inc", "term", "result", "end")
    off = np.cumsum([0] + [len(g["clocks"]) for g in games])
    np.savez(path / "games.npz", **{k: np.array([g[k] for g in games]) for k in keys},
             clocks=np.concatenate([g["clocks"] for g in games]), clock_offsets=off,
             site=np.array([g["site"] for g in games]))  # fmt: skip


def simulate(a):
    """Allie against itself in held-out games' settings (both ratings, the time control), with
    virtual clocks: a move costs max(think, --compute) + --lag seconds (live, the think time
    includes compute; network and server lag come on top)."""
    import threading

    from allie.lichess.engine import Engine, Game, Play

    if a.threads:
        torch.set_num_threads(a.threads)
    G = dict(np.load(Path(a.data) / "games.npz"))
    rng = np.random.default_rng(a.seed)
    pick = []
    for f in [int(x) for x in a.formats.split(",")]:
        for b in range(4):
            idx = np.flatnonzero((G["fmt"] == f) & (G["band"] == b) & (np.arange(len(G["fmt"])) % 2 == 1))
            pick += rng.choice(idx, min(a.per_cell, len(idx)), replace=False).tolist()
    engine = Engine(Model(a.model, a.device, torch.bfloat16, int8=a.device == "cpu"))
    play = Play(rating=0, lag=a.lag)
    done = []
    speed = {0: "bullet", 1: "blitz", 2: "rapid", 3: "classical"}
    out = [None] * len(pick)

    def one(j, i):
        base, inc = int(G["base"][i]), int(G["inc"][i])
        game = Game(engine, int(G["welo"][i]), int(G["belo"][i]), base, inc, speed[int(G["fmt"][i])], seed=j)
        board, moves, clock, spent = chess.Board(), [], [float(base)] * 2, []
        kind, loser, how, last = None, -1, None, None
        while kind is None:
            side = len(moves) % 2
            d = game.decide(play, None, clock[side])
            last = d.wdl
            if d.resign:
                kind, loser, how = "resign", side, "on turn"
                break
            cost = max(d.think, a.compute) + a.lag
            spent.append(cost)
            if len(moves) >= 2:
                clock[side] -= cost
                if clock[side] < 0:
                    bare = board.has_insufficient_material(side == 1)  # the opponent's colour
                    kind, loser = ("flag-draw", -1) if bare else ("flag", side)
                    break
                clock[side] += inc
            board.push_uci(d.move)
            moves.append(d.move)
            game.update(moves, clock[0], clock[1])
            o = board.outcome(claim_draw=True)
            if o:
                kind = "mate" if o.termination == chess.Termination.CHECKMATE else "draw-rule"
                loser = -1 if o.winner is None else int(o.winner)  # winner True = white: loser 1
            elif d.offer_draw and game.accept_draw(play, white=side == 1):
                kind = "agreed"
            elif game.concede(play, clock[side], white=side == 0):
                kind, loser, how = "resign", side, "after own move"
            elif len(moves) >= 400:
                kind = "draw-rule"
        out[j] = dict(game=int(i), fmt=int(G["fmt"][i]), band=int(G["band"][i]), kind=kind,
                      loser=loser, how=how, plies=len(moves), spent=spent, clock=clock,
                      loss_at_end=last[2])  # fmt: skip

    jobs, errors = list(enumerate(pick)), []
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                if not jobs or errors:
                    return
                j, i = jobs.pop()
            try:
                one(j, i)
            except Exception as e:  # noqa: BLE001 - reported below, the run fails
                errors.append(repr(e))
            with lock:
                done.append(j)
                if len(done) % 20 == 0:
                    print(f"{len(done)} / {len(pick)} games", flush=True)

    threads = [threading.Thread(target=worker) for _ in range(a.concurrent)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    engine.close()
    if errors or any(o is None for o in out):
        raise RuntimeError(f"simulation failed: {errors[:3]}")
    Path(a.out).write_text(json.dumps(out) + "\n")


KINDS = ("mate", "resign", "flag", "agreed", "draw-rule", "flag-draw")


def outcomes(g):
    """Each game's moves, loser (0 white, 1 black, -1 none) and end (an index into KINDS)."""
    n = np.diff(g["clock_offsets"])
    loser = np.where(g["result"] == 0, 1, np.where(g["result"] == 1, 0, -1))
    normal = np.where(g["end"] == 1, 0, np.where(loser >= 0, 1, np.where(g["end"] > 1, 4, 3)))
    return n, loser, np.where(g["term"] == 1, np.where(loser >= 0, 2, 5), normal)


def load(path):
    """Per position k (0..n): the heads' outputs for the side to move and the clocks of both."""
    path = Path(path)
    P, G = dict(np.load(path / "positions.npz")), dict(np.load(path / "games.npz"))
    n, loser, kind = outcomes(G)
    gi, k = P["game"], P["k"]
    off, clocks = G["clock_offsets"], G["clocks"]
    base, inc = G["base"][gi], G["inc"][gi]
    own = lambda j: np.where(j < 0, base, clocks[off[gi] + np.maximum(j, 0)])
    mover = k % 2
    P |= dict(
        n=n[gi], mover=mover, fmt=G["fmt"][gi], base=base, inc=inc,
        elo=np.where(mover == 0, G["welo"][gi], G["belo"][gi]),
        elo_other=np.where(mover == 0, G["belo"][gi], G["welo"][gi]),
        clock=own(k - 2), clock_other=own(k - 1),
        spent=np.where((k >= 2) & (k < n[gi]), own(k - 2) - own(np.minimum(k, n[gi] - 1)) + inc, -1),
        kind=kind[gi], loser=loser[gi], half=gi % 2,
    )  # fmt: skip
    return P, G


def sample_time(time, rng):
    """One draw per row of the think-time head (bin, then uniform within it), as behaviour.think."""
    c = time.astype(np.float64).cumsum(1)
    b = (c < rng.random(len(c))[:, None] * c[:, -1:]).sum(1).clip(0, 62)
    u = rng.uniform(-0.5, 0.5, len(c))
    return np.where(b < 16, np.maximum(b + u, 0), 16 * np.exp((b - 16 + u) / 7.06))


def logistic(x, y, ridge=1e-3, iters=50):
    """Maximum-likelihood logistic regression (Newton), a tiny ridge off the intercept."""
    w = np.zeros(x.shape[1])
    r = np.full(x.shape[1], ridge)
    r[0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(x @ w, -50, 50)))
        step = np.linalg.solve((x * (p * (1 - p))[:, None]).T @ x + np.diag(r), x.T @ (p - y) + r * w)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def q(x, qs=(10, 25, 50, 75, 90, 99)):
    return [round(float(v), 3) for v in np.percentile(x, qs)] if len(x) else []


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def risk_set(P):
    """Every position twice, once for each player: the side to move (on its turn, instead of
    moving) and the other side (right after its own move). wdl from that player's view; the
    event is the game's resignation by that player at that position. Positions from ply 2,
    and the final one only when a player could still act (resignation, flag, agreement)."""
    k, n, kind = P["k"], P["n"], P["kind"]
    able = (k >= 2) & ((k < n) | np.isin(kind, (1, 2, 3)))
    rows = []
    for on in (1, 0):
        side = np.where(on, P["mover"], 1 - P["mover"])
        wdl = P["wdl"] if on else P["wdl"][:, ::-1]
        rows.append(dict(
            row=np.flatnonzero(able), on=np.full(able.sum(), on), wdl=wdl[able], k=k[able],
            elo=np.where(on, P["elo"], P["elo_other"])[able], fmt=P["fmt"][able],
            clock=np.where(on, P["clock"], P["clock_other"])[able], base=P["base"][able],
            game=P["game"][able], side=side[able], half=P["half"][able],
            y=((kind == 1) & (k == n) & (P["loser"] == side))[able],
            lost=(P["loser"] == side)[able],
        ))  # fmt: skip
    return {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}


KNOTS = (0.5, 0.7, 0.85, 0.95, 0.99, 0.999)


def design(R, knots=None):
    """behaviour.features (knots: behaviour.resign_features) for every row, vectorized
    (checked against them in fit)."""
    w, d, loss = R["wdl"].T
    left = np.clip(R["clock"] / R["base"], 0, 1.5)
    f = R["fmt"]
    cols = [np.ones(len(w)), lg(loss), lg(w), lg(d), np.minimum(R["k"], 200) / 100,
            (R["elo"] - 1500) / 500, left, f == 0, f == 2, f == 3, R["on"]]  # fmt: skip
    if knots is not None:
        hinge = [np.maximum(0, lg(loss) - lg(np.float64(k))) for k in knots]
        cols += hinge + [R["on"] * v for v in hinge]
    return np.stack([c.astype(np.float64) for c in cols], 1)


def first(key, order, fire):
    """Per key, the first row (in `order`) where fire holds, -1 if none."""
    out = np.full(key.max() + 1, -1)
    rows = order[fire[order]][::-1]
    out[key[rows]] = rows  # reversed, so each key's earliest row is written last
    return out


def resignations(R, rows, h, floor, rng):
    """Where each rule would make each player of the held-out human games resign: the fitted
    hazard (sampled), the bot's previous rule (P(loss) >= 0.97 on 3 own turns in a row, from
    ply 20) and the original Allie's (from ply 2; its resign token is approximated by an
    expected score under 0.05)."""
    key = R["game"][rows] * 2 + R["side"][rows]
    order = rows[np.lexsort((1 - R["on"][rows], R["k"][rows], key))]
    key_o = R["game"][order] * 2 + R["side"][order]
    w, d, loss = R["wdl"][order].T
    turn = R["on"][order] == 1
    bad = turn & (loss >= 0.97) & (R["k"][order] >= 20)
    run = np.zeros(len(order), int)
    last = {}
    for i in np.flatnonzero(turn):  # consecutive own turns, per player
        prev = last.get(key_o[i])
        run[i] = (run[prev] + 1 if prev is not None and bad[prev] else 1) if bad[i] else 0
        last[key_o[i]] = i
    hazard = np.where(loss >= floor, h[order], 0.0)
    keys = np.full(R["game"].max() * 2 + 2, -1)
    keys[key_o] = 1
    fire = dict(
        calibrated=rng.random(len(order)) < hazard,
        before=run >= 3,
        original=turn & (w + d / 2 < 0.05),
    )
    return key_o, order, {name: first(key_o, np.arange(len(order)), f) for name, f in fire.items()}


def fit(a):
    from allie.lichess import behaviour as bh

    P, G = load(a.data)
    rng = np.random.default_rng(0)
    n, loser, kind = outcomes(G)
    fit_half, check_half = 0, 1
    report = {"games": int(len(n)), "positions": int(len(P["k"])),
              "split": "coefficients and thresholds fitted on even game ids; the odd ones validate, and informed the choice of the hazard's form, floor and acceptance percentile"}  # fmt: skip
    hold = np.arange(len(n)) % 2 == check_half
    report["endings (held-out human games)"] = {
        FORMATS[f]: {KINDS[i]: round(float(np.mean(kind[hold & (G["fmt"] == f)] == i)), 3) for i in range(6)}
        for f in range(4)
    }  # fmt: skip

    # think time: the head's draws against the human moves
    played = P["spent"] >= 0
    t = sample_time(P["time"][played], rng)
    s, c = P["spent"][played].astype(float), P["clock"][played].astype(float)
    inc, half = P["inc"][played], P["half"][played]
    # the reserve covers live lag: Lichess charged the bot 0.33 s more than its own think time
    # per move on average, 2.2 s at the 99th percentile (634 moves, 2026-10-04)
    reserve = 2.5
    ratio = (s - inc) / np.maximum(c - reserve, 1)
    share = float(np.clip(np.percentile(ratio[half == fit_half], 99.9), 0.1, 0.5))
    guard = dict(reserve=reserve, share=round(share, 3))
    hard = np.maximum(0, c - guard["reserve"])
    capped = np.minimum(np.minimum(t, hard * guard["share"] + inc), hard)
    old = np.minimum(t, 0.1 * c)  # before: a draw capped at 10% of the clock left
    strata = {
        "all": np.ones(len(s), bool), **{FORMATS[f]: P["fmt"][played] == f for f in range(4)},
        "plies < 20": P["k"][played] < 20, "plies >= 60": P["k"][played] >= 60,
        "clock < 10% of base": c < 0.1 * P["base"][played], "one legal move": P["legal"][played] == 1,
    }  # fmt: skip
    report["think"] = {
        name: dict(n=int((m & (half == check_half)).sum()),
                   **{k: q(v[m & (half == check_half)]) for k, v in
                      (("human", s), ("head draw", t), ("before (10% cap)", old), ("after (guard)", capped))},
                   before_capped=round(float((t > 0.1 * c)[m & (half == check_half)].mean()), 3),
                   after_capped=round(float((t > capped)[m & (half == check_half)].mean()), 3))
        for name, m in strata.items()
    }  # fmt: skip
    report["think"]["guard"] = guard | dict(
        human_moves_over=round(float((s > hard * guard["share"] + inc)[half == check_half].mean()), 4))

    # resignation: a discrete-time hazard on both players' opportunities
    R = risk_set(P)
    X = design(R, KNOTS)
    for r in rng.choice(len(X), 300, replace=False):  # the engine computes the same features
        x = bh.resign_features(tuple(R["wdl"][r]), int(R["k"][r]), int(R["elo"][r]), int(R["fmt"][r]),
                               float(R["clock"][r]), float(R["base"][r]), bool(R["on"][r]), KNOTS)  # fmt: skip
        assert np.allclose(x, X[r], atol=1e-5), (x, X[r])
    y = R["y"].astype(float)
    tr = R["half"] == fit_half
    # humans almost never resign below this P(loss) (5th percentile of their resignations):
    # below it, a resignation is mostly a game that could still be saved
    floor = round(float(np.percentile(R["wdl"][(y == 1) & tr, 2], 5)), 3)
    coef = logistic(X[tr & (R["wdl"][:, 2] >= floor)], y[tr & (R["wdl"][:, 2] >= floor)])
    h = 1 / (1 + np.exp(-np.clip(X @ coef, -50, 50)))
    te = (R["half"] == check_half) & (R["wdl"][:, 2] >= floor)
    edges = np.percentile(h[te], [0, 50, 80, 90, 95, 98, 99, 99.5, 100])
    calibration = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = te & (h >= lo) & (h <= hi)
        calibration.append(dict(predicted=round(float(h[m].mean()), 4), observed=round(float(y[m].mean()), 4), n=int(m.sum())))
    rows = np.flatnonzero(R["half"] == check_half)
    key, order, fired = resignations(R, rows, h, floor, rng)
    players = np.unique(key)
    lost = np.zeros(key.max() + 1, bool)
    lost[key[R["lost"][order]]] = True
    resigned = np.zeros(key.max() + 1, bool)
    resigned[key[R["y"][order]]] = True
    events = order[R["y"][order]]
    out = {"humans": dict(players=int(len(players)), resign=round(float(resigned[players].mean()), 3),
                          on_turn=round(float(R["on"][events].mean()), 3),
                          loss_at_resign=q(R["wdl"][events, 2], (5, 25, 50)), ply_at_resign=q(R["k"][events], (25, 50, 75)))}  # fmt: skip
    for name, f in fired.items():
        ok = f[players] >= 0
        r = order[f[players][ok]]
        out[name] = dict(
            resign=round(float(ok.mean()), 3),
            resign_in_games_not_lost=round(float((ok & ~lost[players]).mean()), 4),
            loss_at_resign=q(R["wdl"][r, 2], (5, 25, 50)), ply_at_resign=q(R["k"][r], (25, 50, 75)),
        )  # fmt: skip
    by_loss = []
    edges = [floor] + [k for k in KNOTS if k > floor] + [1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        for on in (1, 0):
            m = te & (R["wdl"][:, 2] >= lo) & (R["wdl"][:, 2] < hi) & (R["on"] == on)
            if m.any():
                by_loss.append(dict(loss=[lo, hi], on_turn=on, n=int(m.sum()), observed=round(float(y[m].mean()), 4),
                                    predicted=round(float(h[m].mean()), 4)))  # fmt: skip
    report["resign"] = dict(floor=floor, coef=coef.round(4).tolist(), calibration=calibration,
                            calibration_by_loss=by_loss, players=out)  # fmt: skip

    # draws: agreements happen on someone's turn; offer and accept rules are heuristics (the data
    # show agreements, not offers or declines)
    turn = (P["k"] < P["n"]) | np.isin(P["kind"], (1, 2, 3))
    min_ply = 20
    D = dict(wdl=P["wdl"], k=P["k"], elo=P["elo"], fmt=P["fmt"], clock=P["clock"], base=P["base"],
             on=np.ones(len(P["k"])))  # fmt: skip
    XD = design(D)
    yd = ((P["kind"] == 3) & (P["k"] == P["n"])).astype(float)
    dm = turn & (P["k"] >= min_ply)
    dcoef = logistic(XD[dm & (P["half"] == fit_half)], yd[dm & (P["half"] == fit_half)])
    hd = 1 / (1 + np.exp(-np.clip(XD @ dcoef, -50, 50)))
    agreed = yd == 1
    e = P["wdl"][:, 0] + P["wdl"][:, 1] / 2
    at = np.maximum(e, 1 - e)[agreed & (P["half"] == fit_half)]
    # three quarters of human agreements happen at or below this expected score for the better side
    accept = round(float(np.percentile(at, 75)), 3) if len(at) else 0.5
    dt = dm & (P["half"] == check_half)
    report["draw"] = dict(accept=accept, expected_score_at_agreement=q(e[agreed & (P["half"] == check_half)], (5, 25, 50, 75, 95)),
                          agreements_predicted=round(float(hd[dt].sum())), agreements_observed=int(yd[dt].sum()))  # fmt: skip

    # flags: how far behind was the player who flagged?
    flag = (P["kind"] == 2) & (P["k"] == P["n"]) & (P["half"] == check_half)
    side = flag & (P["loser"] == P["mover"])
    report["flag"] = dict(flagger_on_turn=round(float(side.sum() / max(flag.sum(), 1)), 3),
                          flagger_loss=q(P["wdl"][side, 2], (10, 25, 50, 75, 90)))  # fmt: skip

    manifest = json.loads((ev.OUT / "manifest.json").read_text())
    model = json.loads((Path(a.model) / "config.json").read_text()) if a.model else {}
    params = dict(
        resign=dict(coef=coef.round(5).tolist(), floor=floor, knots=list(KNOTS)),
        draw=dict(coef=dcoef.round(5).tolist(), accept=accept, min_ply=min_ply),
        guard=guard,
        provenance=dict(
            fit="analysis/lichess/human.py fit", games=int(len(n)), fitted_on="even game ids",
            source="the main evaluation's games (strat-eval-v1, rows sha256 " + manifest["sha256"][:16] + ")",
            model=model.get("name"), model_sha256=model.get("checkpoint_sha256"), weights=a.weights,
        ),
    )
    Path(a.params).write_text(json.dumps(params, indent=1) + "\n")
    Path(a.data, "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


def compare(a):
    """The simulated games (bot vs bot) against the held-out human games they copy."""
    G = dict(np.load(Path(a.data) / "games.npz"))
    sim = json.loads(Path(a.sim).read_text())
    n, loser, kind = outcomes(G)
    off = G["clock_offsets"]
    out = {}
    for f in range(4):
        S = [x for x in sim if x["fmt"] == f]
        if not S:
            continue
        idx = np.array([x["game"] for x in S])
        human_spent = []
        for i in idx:
            c, inc, base = G["clocks"][off[i] : off[i + 1]], G["inc"][i], G["base"][i]
            prev = np.r_[base, base, c[:-2]]
            human_spent.append((prev - c + inc)[2:])
        human_spent = np.concatenate(human_spent)
        bot_spent = np.concatenate([np.array(x["spent"][2:]) for x in S])
        hk = kind[idx]
        bk = np.array([KINDS.index(x["kind"]) for x in S])
        res = [x for x in S if x["kind"] == "resign"]
        out[FORMATS[f]] = dict(
            games=len(S),
            endings=dict(human={KINDS[i]: round(float((hk == i).mean()), 3) for i in range(6)},
                         bot={KINDS[i]: round(float((bk == i).mean()), 3) for i in range(6)}),
            plies=dict(human=q(n[idx], (25, 50, 75)), bot=q([x["plies"] for x in S], (25, 50, 75))),
            resign_after_own_move=round(float(np.mean([x["how"] == "after own move" for x in res])), 3) if res else None,
            seconds_per_move=dict(human=q(human_spent, (10, 25, 50, 75, 90, 99)), bot=q(bot_spent, (10, 25, 50, 75, 90, 99))),
            flagged_while_not_losing=sum(x["kind"] == "flag" and x["loss_at_end"] < 0.5 for x in S),
        )  # fmt: skip
    text = json.dumps(out, indent=1)
    Path(a.sim).with_suffix(".compare.json").write_text(text + "\n")
    print(text)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--out", required=True)
    s.add_argument("--per-cell", type=int, default=800)
    s.add_argument("--tokens", type=int, default=8192)
    s.add_argument("--int8", action="store_true", help="int8 weights, as the bot serves on CPU")
    s.add_argument("--threads", type=int)
    m = sub.add_parser("simulate")
    m.add_argument("--model", required=True)
    m.add_argument("--data", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--device", default="cpu")
    m.add_argument("--threads", type=int)
    m.add_argument("--formats", default="0,1,2,3")
    m.add_argument("--per-cell", type=int, default=25)
    m.add_argument("--concurrent", type=int, default=4)
    m.add_argument("--compute", type=float, default=0.05, help="seconds of compute per move")
    m.add_argument("--lag", type=float, default=0.3, help="seconds of network and server lag per move")
    m.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("--data", required=True)
    c.add_argument("--sim", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--data", required=True)
    f.add_argument("--params", default=str(Path(__file__).parents[2] / "src/allie/lichess/behaviour.json"))
    f.add_argument("--model", help="the export the heads came from (provenance)")
    f.add_argument("--weights", default="int8 (cpu)", help="precision the heads ran in (provenance)")
    a = p.parse_args()
    {"score": score, "fit": fit, "simulate": simulate, "compare": compare}[a.command](a)



# ---- fit: everything below reads positions.npz / games.npz on CPU ----------------------------

if __name__ == "__main__":
    main()
