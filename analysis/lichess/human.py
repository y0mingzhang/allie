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
    m = Model(a.model, a.device, torch.bfloat16, int8=a.int8)
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
            out["wdl"].append(torch.softmax(zz[:, WDL], -1).numpy())
            out["time"].append(torch.softmax(zz[:, TIME], -1).half().numpy())
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
    """Allie against itself in the human games' settings (both ratings, time control), with
    virtual clocks: each side spends its decision's think time plus --lag seconds a move."""
    import threading

    from allie.lichess.engine import Engine, Game, Play

    if a.threads:
        torch.set_num_threads(a.threads)
    G = dict(np.load(Path(a.data) / "games.npz"))
    rng = np.random.default_rng(a.seed)
    pick = []
    for f in [int(x) for x in a.formats.split(",")]:
        for b in range(4):
            idx = np.flatnonzero((G["fmt"] == f) & (G["band"] == b))
            pick += rng.choice(idx, min(a.per_cell, len(idx)), replace=False).tolist()
    engine = Engine(Model(a.model, a.device, torch.bfloat16, int8=a.device == "cpu"))
    play = Play(rating=0)
    speed = {0: "bullet", 1: "blitz", 2: "rapid", 3: "classical"}
    out = [None] * len(pick)

    def one(j, i):
        base, inc = int(G["base"][i]), int(G["inc"][i])
        game = Game(engine, int(G["welo"][i]), int(G["belo"][i]), base, inc, speed[int(G["fmt"][i])], seed=j)
        board, moves, clock, thinks, kind, loser, wdls = chess.Board(), [], [float(base)] * 2, [], None, -1, []
        while kind is None:
            side = len(moves) % 2
            d = game.decide(play, None, clock[side])
            wdls.append(d.wdl)
            if d.resign:
                kind, loser = "resign", side
                break
            thinks.append(d.think)
            if len(moves) >= 2:
                clock[side] -= d.think + a.lag
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
                loser = -1 if o.winner is None else int(o.winner)  # winner True = white, loser black
                break
            if d.offer_draw and game.accept_draw(play, white=side == 1):
                kind = "agreed"
            elif len(moves) >= 400:
                kind = "draw-rule"
        out[j] = dict(game=int(i), fmt=int(G["fmt"][i]), band=int(G["band"][i]), kind=kind,
                      loser=loser, plies=len(moves), think=thinks, clock=clock,
                      loss_at_end=wdls[-1][2])  # fmt: skip

    jobs = list(enumerate(pick))
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                if not jobs:
                    return
                j, i = jobs.pop()
            one(j, i)

    threads = [threading.Thread(target=worker) for _ in range(a.concurrent)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    engine.close()
    Path(a.out).write_text(json.dumps(out) + "\n")


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
    m.add_argument("--lag", type=float, default=0.1, help="seconds lost per move (compute, network)")
    m.add_argument("--seed", type=int, default=0)
    f = sub.add_parser("fit")
    f.add_argument("--data", required=True)
    f.add_argument("--params", default=str(Path(__file__).parents[2] / "src/allie/lichess/behaviour.json"))
    a = p.parse_args()
    {"score": score, "fit": fit, "simulate": simulate}[a.command](a)



# ---- fit: everything below reads positions.npz / games.npz on CPU ----------------------------

KINDS = ("mate", "resign", "flag", "agreed", "draw-rule", "flag-draw")


def outcomes(g):
    """Each game's end, its loser (0 white, 1 black, -1 none) and the resignation's position
    (the resigner's last turn: k = n if they resigned instead of moving, n - 1 if they moved
    and then resigned on the opponent's turn), -1 otherwise."""
    n = np.diff(g["clock_offsets"])
    loser = np.where(g["result"] == 0, 1, np.where(g["result"] == 1, 0, -1))
    normal = np.where(g["end"] == 1, 0, np.where(loser >= 0, 1, np.where(g["end"] > 1, 4, 3)))
    kind = np.where(g["term"] == 1, np.where(loser >= 0, 2, 5), normal)
    at = np.where(kind == 1, np.where(n % 2 == loser, n, n - 1), -1)
    return n, loser, kind, at


def load(path):
    path = Path(path)
    P, G = dict(np.load(path / "positions.npz")), dict(np.load(path / "games.npz"))
    n, loser, kind, at = outcomes(G)
    gi, k = P["game"], P["k"]
    off = G["clock_offsets"]
    clocks = G["clocks"]
    base, inc = G["base"][gi], G["inc"][gi]
    own = lambda j: np.where(j < 0, base, clocks[off[gi] + np.maximum(j, 0)])
    mover = k % 2
    P |= dict(
        n=n[gi], mover=mover, fmt=G["fmt"][gi], base=base, inc=inc,
        elo=np.where(mover == 0, G["welo"][gi], G["belo"][gi]),
        clock=own(k - 2), opp=own(k - 1),
        spent=np.where((k >= 2) & (k < n[gi]), own(k - 2) - own(np.minimum(k, n[gi] - 1)) + inc, -1),
        kind=kind[gi], loser=loser[gi], at=at[gi],
    )  # fmt: skip
    return P, G


def sample_time(time, rng):
    """One draw per row of the think-time head (bin, then uniform within it), as behaviour.think."""
    p = time.astype(np.float64)
    c = p.cumsum(1)
    b = (c < rng.random(len(p))[:, None] * c[:, -1:]).sum(1).clip(0, 62)
    u = rng.uniform(-0.5, 0.5, len(p))
    return np.where(b < 16, np.maximum(b + u, 0), 16 * np.exp((b - 16 + u) / 7.06))


def logistic(x, y, ridge=1e-3, iters=50):
    """Maximum-likelihood logistic regression (Newton), tiny ridge off the intercept."""
    w = np.zeros(x.shape[1])
    r = np.full(x.shape[1], ridge)
    r[0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(x @ w, -50, 50)))
        g = x.T @ (p - y) + r * w
        h = (x * (p * (1 - p))[:, None]).T @ x + np.diag(r)
        step = np.linalg.solve(h, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def q(x, qs=(10, 25, 50, 75, 90, 99)):
    return [round(float(v), 1) for v in np.percentile(x, qs)] if len(x) else []


def design(P):
    """behaviour.features for every row, vectorized (checked against it in fit)."""
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    w, d, loss = P["wdl"].T
    left = np.clip(P["clock"] / P["base"], 0, 1.5)
    f = P["fmt"]
    cols = [np.ones(len(w)), lg(loss), lg(w), lg(d), np.minimum(P["k"], 200) / 100,
            (P["elo"] - 1500) / 500, left, f == 0, f == 2, f == 3]  # fmt: skip
    return np.stack([c.astype(np.float64) for c in cols], 1)


def first_events(game, prob, rng):
    """Per game, the row index of the first sampled event (rows in game order), -1 if none."""
    hit = rng.random(len(prob)) < prob
    first = np.full(game.max() + 1, -1)
    rows = np.flatnonzero(hit)[::-1]
    first[game[rows]] = rows  # reversed, so the earliest row of each game is written last
    return first


def resign_rules(P, rows, rng, h, floor):
    """Each rule's resignation row per (game, side) along the human games, -1 if it never fires."""
    k, loss = P["k"][rows], P["wdl"][rows, 2]
    w, d = P["wdl"][rows, 0], P["wdl"][rows, 1]
    side = P["game"][rows] * 2 + P["mover"][rows]
    old = (loss >= 0.97) & (k >= 20)
    run = np.zeros(len(rows), int)  # consecutive own turns at P(loss) >= 0.97
    for i in range(len(rows)):
        run[i] = old[i] * (run[i - 1] + 1 if i and side[i - 1] == side[i] else old[i])
    return dict(
        calibrated=first_events(side, np.where(loss >= floor, h, 0.0), rng),
        current=first_events(side, (run >= 3).astype(float), rng),
        original=first_events(side, ((w + d / 2 < 0.05) & (k >= 2)).astype(float), rng),
    ), side


def fit(a):
    from allie.lichess import behaviour as bh

    P, G = load(a.data)
    rng = np.random.default_rng(0)
    test = P["game"] % 2 == 1
    k, n, kind = P["k"], P["n"], P["kind"]
    report = {"games": len(G["fmt"]), "positions": len(k)}
    names = {i: s for i, s in enumerate(KINDS)}
    report["endings"] = {
        FORMATS[f]: {names[i]: round(float(np.mean(outcomes(G)[2][G["fmt"] == f] == i)), 3) for i in range(6)}
        for f in range(4)
    }  # fmt: skip

    # think time: the head's draws against the human moves
    played = P["spent"] >= 0
    t = sample_time(P["time"][played], rng)
    s = P["spent"][played].astype(float)
    c = P["clock"][played].astype(float)
    strata = {
        "all": np.ones(len(s), bool),
        **{FORMATS[f]: P["fmt"][played] == f for f in range(4)},
        "plies < 20": P["k"][played] < 20,
        "plies >= 60": P["k"][played] >= 60,
        "clock < 10% of base": c < 0.1 * P["base"][played],
        "one legal move": P["legal"][played] == 1,
    }
    report["think"] = {name: dict(n=int(m.sum()), human=q(s[m]), model=q(t[m]),
                                  human_mean=round(float(s[m].mean()), 2), model_mean=round(float(t[m].mean()), 2))
                       for name, m in strata.items()}  # fmt: skip
    ratio = (s - P["inc"][played]) / np.maximum(c - 1, 1)
    report["think"]["share of clock above inc, human quantiles"] = q(ratio, (50, 90, 99, 99.9))
    # guard: share of (clock - reserve) that 99.9% of human moves stay under, plus the increment
    guard = dict(reserve=1.0, increment=1.0, share=round(float(np.clip(np.percentile(ratio, 99.9), 0.1, 0.5)), 3))
    budget = np.maximum(0, c - guard["reserve"]) * guard["share"] + P["inc"][played] * guard["increment"]
    report["think"]["guard"] = guard | dict(human_moves_over=round(float((s > budget).mean()), 4),
                                            model_draws_over=round(float((t > budget).mean()), 4))  # fmt: skip
    old = np.minimum(t, 0.1 * c)  # before: capped at 10% of the clock left
    new = np.minimum(t, budget)
    for name, m in strata.items():
        report["think"][name] |= dict(before=q(old[m]), after=q(new[m]),
                                      before_capped=round(float((t[m] > 0.1 * c[m]).mean()), 3),
                                      after_capped=round(float((t[m] > budget[m]).mean()), 3))  # fmt: skip

    # resignation: per-turn hazard on the rows where the side to move could still act
    able = (k < n) | ((k == n) & np.isin(kind, (1, 2, 3)))
    y = ((kind == 1) & (k == P["at"])).astype(float)
    X = design(P)
    i = rng.choice(np.flatnonzero(able), 200, replace=False)
    for r in i:  # the engine computes the same features
        x = bh.features(tuple(P["wdl"][r]), int(k[r]), int(P["elo"][r]), int(P["fmt"][r]),
                        float(P["clock"][r]), float(P["base"][r]))  # fmt: skip
        assert np.allclose(x, X[r], atol=1e-5), (x, X[r])
    loss_at = P["wdl"][(y == 1), 2]
    floor = round(float(np.percentile(loss_at, 1)), 3)
    train = able & ~test & (P["wdl"][:, 2] >= floor)
    coef = logistic(X[train], y[train])
    h = 1 / (1 + np.exp(-np.clip(X @ coef, -50, 50)))
    ev_rows = able & test & (P["wdl"][:, 2] >= floor)
    bins = np.percentile(h[ev_rows], [0, 50, 80, 90, 95, 98, 99, 99.5, 100])
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = ev_rows & (h >= lo) & (h <= hi)
        cal.append(dict(predicted=round(float(h[m].mean()), 4), observed=round(float(y[m].mean()), 4), n=int(m.sum())))
    report["resign"] = dict(floor=floor, loss_at_human_resignation=q(loss_at, (1, 5, 25, 50, 75)), calibration=cal)
    rows = np.flatnonzero(able & test)
    first, side = resign_rules(P, rows, rng, h[rows], floor)
    lost = P["loser"][rows] == P["mover"][rows]
    human = (y[rows] == 1)
    sides = np.unique(side)
    hs = np.zeros(side.max() + 1, bool)
    hs[side[human]] = True
    lost_side = np.zeros(side.max() + 1, bool)
    lost_side[side[lost]] = True
    played_side = np.zeros(side.max() + 1, bool)
    played_side[sides] = True
    out = {"human": dict(resigns=round(float(hs[sides].mean()), 3),
                         loss_at_resign=q(P["wdl"][rows[human], 2], (5, 25, 50)),
                         ply_at_resign=q(k[rows[human]], (25, 50, 75)))}  # fmt: skip
    for rule, f in first.items():
        fired = f[sides] >= 0
        r = f[sides][fired]
        out[rule] = dict(
            resigns=round(float(fired.mean()), 3),
            resigns_in_games_not_lost=round(float((fired & ~lost_side[sides]).mean()), 4),
            loss_at_resign=q(P["wdl"][rows[r], 2], (5, 25, 50)),
            ply_at_resign=q(k[rows[r]], (25, 50, 75)),
        )
    report["resign"]["on human games (test half), per side"] = out

    # draws: the expected score at which humans agree, and a per-turn offer hazard
    agreed = (kind == 3) & (k == n)
    e = P["wdl"][agreed, 0] + P["wdl"][agreed, 1] / 2
    accept = round(float(np.percentile(np.maximum(e, 1 - e), 95)), 3)
    yd = agreed.astype(float)
    min_ply = 20
    dtrain = able & ~test & (k >= min_ply)
    dcoef = logistic(X[dtrain], yd[dtrain])
    hd = 1 / (1 + np.exp(-np.clip(X @ dcoef, -50, 50)))
    dtest = able & test & (k >= min_ply)
    report["draw"] = dict(accept=accept, expected_score_at_agreement=q(e, (5, 25, 50, 75, 95)),
                          offers_predicted=round(float(hd[dtest].sum())), agreed_observed=int(yd[dtest].sum()))  # fmt: skip

    # flags: was the flagger losing?
    flag = (kind == 2) & (k == n)
    report["flag"] = dict(rate={FORMATS[f]: round(float(np.mean(outcomes(G)[2][G["fmt"] == f] == 2)), 3) for f in range(4)},
                          flagger_loss=q(P["wdl"][flag, 2], (10, 25, 50, 75)))  # fmt: skip

    params = dict(
        resign=dict(coef=coef.round(5).tolist(), floor=floor),
        draw=dict(coef=dcoef.round(5).tolist(), accept=accept, min_ply=min_ply),
        guard=guard,
        provenance=dict(data=str(a.data), games=report["games"], fit="analysis/lichess/human.py fit"),
    )
    Path(a.params).write_text(json.dumps(params, indent=1) + "\n")
    Path(a.data, "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
