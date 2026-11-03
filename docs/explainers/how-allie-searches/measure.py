"""Every number the video shows, measured or read from its source, into data.json with a source note each.

    ALLIE_DATA=... python measure.py [sources aggregates position]   # CPU node, 16 cores, 32 GB, minutes

Stockfish only scores moves here (results of analysis/elo_strength/calib_mpv.py); it trains and steers nothing."""

import json
import sys
from pathlib import Path
from time import monotonic

import chess
import numpy as np
import torch

from allie import paths
from allie.lichess import behaviour, calibration as C, tree, treers
from allie.lichess.api import Allie
from allie.lichess.engine import FORMATS, Engine, Game
from allie.lichess.tokens import MOVE_START, MOVES, SECONDS
from allie.search import kl
from allie.search.native import from_prefix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "analysis/elo_strength"))
import calib_cells as cc  # noqa: E402
import calib_fit as cf  # noqa: E402

R = paths.DATA / "release-2.1"
SEPT = paths.DATA / "results/recipe10x/elo-strength/calib-sept"
FMTS = ("bullet", "blitz", "rapid", "classical")
BINS = list(range(800, 2601, 200))
RUNGS = C.LADDER
INDEX, PLY = 146127, 52  # the demo position: its row in the September 2026 calibration set
SERIOUS = 0.1  # a move Allie takes seriously: at least this share of its prediction
THREADS = 12  # calibration.COST's


def table():
    """September's positions, flat over legal moves: Allie 2.1's prior and searched values (NaN past the largest
    rung the position has), Stockfish's cp; per position its meta, heads and the human move's flat index."""
    meta = np.load(SEPT / "positions-human.npz")["meta"]
    mpv = cf.load_mpv(SEPT / "mpv-human")
    pos = {}
    for b in (256, 512, 1024):
        for z in cf.chunks(R / f"calib/sept-{b}"):
            tags = [r for r in RUNGS if f"cov_x{r}k" in z]
            for n, i in enumerate(z["index"].tolist()):
                a, e = z["offsets"][n], z["offsets"][n + 1]
                legal = z["legal"][a:e].astype(int)
                o = pos.setdefault(
                    i,
                    dict(legal=legal, prior=z["prior"][a:e].astype(float), heads=z["heads"][n].astype(float),
                         x=np.full((len(RUNGS), e - a), np.nan)),
                )  # fmt: skip
                perm = np.argsort(legal)[np.argsort(np.argsort(o["legal"]))]
                for r in tags:
                    k = RUNGS.index(r)
                    if np.isnan(o["x"][k, 0]):
                        o["x"][k] = z[f"cov_x{r}k"][a:e][perm]
    keep = sorted(i for i, o in pos.items() if i in mpv and set(o["legal"].tolist()) == set(mpv[i]))
    off = np.cumsum([0] + [len(pos[i]["legal"]) for i in keep])
    return dict(
        off=off,
        meta=meta[keep],
        index=np.array(keep),
        heads=np.stack([pos[i]["heads"] for i in keep]),
        prior=np.concatenate([pos[i]["prior"] for i in keep]),
        x=np.concatenate([pos[i]["x"].T for i in keep]),
        cp=np.concatenate([[mpv[i][t] for t in pos[i]["legal"].tolist()] for i in keep]).astype(float),
        human=np.array([a + pos[i]["legal"].tolist().index(int(meta[i][8])) for i, a in zip(keep, off)]),
    )


def aggregates():
    T = table()
    off, prior, meta, heads, human = T["off"], T["prior"], T["meta"], T["heads"], T["human"]
    starts, N = off[:-1], len(human)
    at = np.repeat(np.arange(N), np.diff(off))
    add = lambda v: np.add.reduceat(v, starts)  # noqa: E731
    lost = cf.winp(np.maximum.reduceat(T["cp"], starts)[at]) - cf.winp(T["cp"])
    acc, blunder = np.clip(103.1668 * np.exp(-0.04354 * lost) - 3.1669, 0, 100), lost >= 15
    played = np.zeros(len(prior), bool)
    played[human] = True
    f, b, elo, game = meta[:, 2], meta[:, 3], meta[:, 4], meta[:, 0]
    x = T["x"].copy()  # a rung a position lacks reads as the largest it has
    for k in range(1, len(RUNGS)):
        x[:, k] = np.where(np.isnan(x[:, k]), x[:, k - 1], x[:, k])
    time = np.exp(heads[:, :63] - heads[:, :63].max(1, keepdims=True))
    time /= time.sum(1, keepdims=True)
    think, rule, r = time @ C.SECONDS, np.array(C.RULES)[f], (elo - 1700) / 1000
    beta = rule[:, 0] * np.exp(rule[:, 1] * r) * (think / 10) ** (rule[:, 2] + rule[:, 3] * r) + rule[:, 4]
    # each position's rungs as the bot's think-time draws give them: calib_unified.load's scoring
    cc.MARGIN = C.MARGIN
    P = np.load(SEPT / "positions-human.npz")
    tok = P["tokens"][P["offsets"][T["index"]] + 2].astype(int)
    inc = np.where((tok >= 10) & (tok <= 190), tok - 10, 0)
    cost = np.array(RUNGS) * C.COST
    W = np.stack(
        [cc.weights(heads[i, :63], float(meta[i, 7]) if meta[i, 7] >= 0 else None, inc[i], cost) for i in range(N)]
    )
    lacking = np.isnan(T["x"][starts])
    for k in range(len(RUNGS) - 1, -1, -1):
        W[:, k] += W[:, k + 1] * lacking[:, k]
        W[:, k + 1] *= ~lacking[:, k]
    lp = np.log(np.maximum(prior, 1e-300))

    def tilted(v, bt):
        z = lp + bt[at] * v
        p = np.exp(z - np.maximum.reduceat(z, starts)[at])
        return p / add(p)[at]

    def played_as_rule(scale=1.0):
        return W[:, 0][at] * prior + sum(W[:, k + 1][at] * tilted(x[:, k], beta * scale) for k in range(len(RUNGS)))

    def stats(p, sel):
        m, n = sel[at], sel.sum()
        return dict(
            accuracy=round(float((acc * p)[m].sum() / n), 3),
            blunder=round(float(100 * (blunder * p)[m].sum() / n), 3),
            ce=round(float(np.log(prior[human[sel]]).mean() - np.log(np.maximum(p[human[sel]], 1e-12)).mean()), 4),
        )

    mix = played_as_rule()
    rng = np.random.default_rng(20261103)
    big = prior >= SERIOUS
    spot = {}
    for i, name in enumerate(FMTS):
        o = {k: [] for k in ("blunder", "lo", "hi", "n", "fine", "hits", "predicted", "tilted")}
        for c in BINS:
            m = ((f == i) & (b == c))[at] & big
            g = np.unique(game[at][m & blunder], return_inverse=True)[1]
            hits, pred = np.bincount(g, played[m & blunder].astype(float)), np.bincount(g, prior[m & blunder])
            draws = rng.multinomial(len(hits), np.ones(len(hits)) / len(hits), 1000)
            lo, hi = np.percentile((draws @ hits) / (draws @ pred), [2.5, 97.5])
            o["blunder"].append(round(float(hits.sum() / pred.sum()), 3))
            o["lo"].append(round(float(lo), 4))
            o["hi"].append(round(float(hi), 4))
            o["n"].append(int((m & blunder).sum()))
            o["fine"].append(round(float(played[m & ~blunder].sum() / prior[m & ~blunder].sum()), 3))
            o["hits"].append(int(hits.sum()))
            o["predicted"].append(round(float(pred.sum()), 2))
            o["tilted"].append(round(float(hits.sum() / mix[m & blunder].sum()), 3))
        spot[name] = o

    def by_think(i, lo, hi):
        """The blunders' ratio within one time control's ratings, by tercile of predicted think time."""
        sel = (f == i) & (b >= lo) & (b <= hi)
        cuts = np.quantile(think[sel], [1 / 3, 2 / 3])
        tier = np.digitize(think, cuts)
        ms = [(sel & (tier == t))[at] & big & blunder for t in range(3)]
        return dict(
            cuts=[round(float(v), 1) for v in cuts],
            ratio=[round(float(played[m].sum() / prior[m].sum()), 3) for m in ms],
            moves=[int(m.sum()) for m in ms],
        )

    fine = big & ~blunder
    spot |= dict(
        threshold=SERIOUS,
        pooled_fine=round(float(played[fine].sum() / prior[fine].sum()), 3),
        fine_moves=int(fine.sum()),
        blunder_moves=int((big & blunder).sum()),
        mass_share=round(float((prior * (big & blunder)).sum() / (prior * blunder).sum()), 3),
        by_think=dict(classical_2000_2600=by_think(3, 2000, 2600)),
    )

    def reach(i, c):
        """A cell's stats with the rule's beta scaled at every position, its rungs still mixed as played."""
        sel = (f == i) & (b == c)
        return [dict(scale=s, beta=round(float(np.median(beta[sel])) * s, 2)) | stats(played_as_rule(s), sel) for s in (1, 2, 3, 4, 6, 8)]  # fmt: skip

    def roads(i, c):
        sel = (f == i) & (b == c)
        m, n = sel[at], sel.sum()
        temperature = []
        for t in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1):
            z = lp / t
            p = np.exp(z - np.maximum.reduceat(z, starts)[at])
            temperature.append(dict(t=t) | stats(p / add(p)[at], sel))
        top = np.zeros(len(prior))
        top[[a + int(np.argmax(prior[a:e])) for a, e in zip(starts[sel], off[1:][sel])]] = 1
        return dict(
            n=int(n),
            human=dict(accuracy=round(float(acc[played & m].mean()), 3), blunder=round(float(100 * blunder[played & m].mean()), 3)),
            raw=stats(prior, sel),
            shipped=stats(mix, sel),
            argmax=dict(accuracy=round(float((acc * top)[m].sum() / n), 3), blunder=round(float(100 * (blunder * top)[m].sum() / n), 3)),
            temperature=temperature,
            tilt=[dict(scale=s) | stats(played_as_rule(s), sel) for s in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0)],
            leaves=[dict(leaves=n_) | stats(tilted(x[:, k], beta), sel) for k, n_ in enumerate(RUNGS)],
        )  # fmt: skip

    tv = 0.5 * add(np.abs(mix - prior))
    plot = json.loads((R / "calib-fit/win21-sept.plot.json").read_text())["cells"]
    worst = 0.0
    for i, name in enumerate(FMTS):
        for c in BINS:
            s, q = stats(mix, (f == i) & (b == c)), plot[f"{name}/{c}"]["calibrated"]
            worst = max(worst, abs(s["accuracy"] - q["accuracy"]), abs(s["blunder"] - 100 * q["blunder"]))
    print("the rule replayed here against the README figure's data: worst gap", worst, flush=True)
    return dict(
        _src_blind_spot="Moves Allie 2.1's raw prediction gives at least 10%, on the 197,191 September 2026 calibration positions (none in the fit): humans' plays over Allie's summed probability, per time control and 200-point bin, split by Stockfish (blunder: 15+ points of win chance lost; fine: the rest). hits, predicted: the blunder ratio's numerator and denominator. tilted: the same hits over the rule's probability as played. lo, hi: 95%, 1,000 bootstrap draws over games. by_think: the ratio by tercile of predicted think time, within classical 2000-2600; it barely moves. Priors release-2.1/calib/sept-{256,512,1024}, Stockfish calib-sept/mpv-human. New for this video.",
        blind_spot=spot,
        _src_tilt_reach="bullet/800, blitz/800: the rule's beta times scale at every position, rungs mixed as played; beta is the cell's median after scaling. ce: nats against the raw prediction. Nothing here tests a higher floor on other cells.",
        tilt_reach={"bullet/800": reach(0, 800), "blitz/800": reach(1, 800)},
        _src_roads="The same positions, per cell. temperature: the raw prediction to the power 1/t. tilt: the shipped rule with beta times scale, its rungs mixed by the think-time draws as calib_unified scores it. shipped: scale 1. leaves: the shipped beta at one search size (a position lacking the rung reads its largest). ce: nats against the raw prediction. replay_gap: worst difference from the README figure's own calibrated accuracy or blunder rate, which interpolates beta on a grid.",
        roads=dict(classical_2400=roads(3, 2400)),
        replay_gap=round(worst, 3),
        _src_typical="Median over positions of the share of probability the shipped rule moves (total variation from the raw prediction), per cell, rule as played.",
        typical={name: [round(float(np.median(tv[(f == i) & (b == c)])), 4) for c in BINS] for i, name in enumerate(FMTS)},
    )  # fmt: skip


def position():
    z = np.load(SEPT / "positions-human.npz")
    a, e = z["offsets"][INDEX], z["offsets"][INDEX + 1]
    toks, feats, meta = z["tokens"][a:e].astype(int), z["feats"][a:e], z["meta"][INDEX]
    assert meta[6] == PLY
    ucis = [MOVES[t - MOVE_START] for t in toks[11:]]
    base, inc = int(SECONDS[toks[1] - 192]), int(toks[2] - 10)
    white, black = (int("".join(map(str, toks[s : s + 4]))) for s in (3, 7))
    clocks = [int(feats[11 + k][1]) for k in range(len(ucis))]
    speed = FMTS[meta[2]]
    allie = Allie.from_pretrained(str(R / "hf"), device="cpu", threads=THREADS)
    torch.set_num_threads(1)
    engine = Engine(allie.model)

    def game(w, bl):
        g = Game(engine, w, bl, base, inc, speed, seed=0)
        g.update(ucis)
        g.clocks = list(clocks)
        return g

    g = game(white, black)
    assert np.array_equal(np.array(g.features(), np.float32)[-1], feats[-1].astype(np.float32))
    legal, p, wdl, time = g.position()
    bd = g.board.copy()
    san = {u: bd.san(chess.Move.from_uci(u)) for u in legal}
    for s in cf.chunks(SEPT / "mpv-human"):
        if len(n := np.flatnonzero(s["index"] == INDEX)):
            a, e = s["offsets"][n[0]], s["offsets"][n[0] + 1]
            stock = {san[MOVES[int(t) - MOVE_START]]: int(c) for t, c in zip(s["tokens"][a:e], s["cp"][a:e])}
            break

    # the game's tree: kl.py's forest on the released model, the search allie_fast.KL (the bot's) ports
    leaves = RUNGS[-1]
    games, zs = tree.games_of(tree.KL(C.GROW, C.READ, views=C.VIEWS), g)
    trees = [tree.Tree(x, y, capacity=2 * leaves + 256) for x, y in zip(games, zs)]
    bridges = []
    for t, x in zip(trees, games):
        t.concurrent = True
        bridges.append(t.handles([x.tokens], [np.array(x.features(), np.float32)], "zero"))
    F = kl.Forest(
        bridges,
        [from_prefix(np.asarray(g.tokens))],
        [len(g.tokens)],
        trees[0].capacity,
        prior=g.sync()[None].double().numpy(),
    )
    kl.grow(F, leaves, **C.GROW)
    E, s = len(F.rootw), C.READ["squash"]
    kid, cost = F.ekid[:E], F.cost[: F.size]
    unexpanded = np.repeat(np.arctanh(s * np.clip(F.value[: F.n], -1, 1)), F.count[: F.n])
    root = [MOVES[t - MOVE_START] for t in F.etok[:E]]

    def values(r):
        """The root moves' values with the tree of r leaves."""
        keep = cost <= r
        V = kl.backup(F.view(), keep, **C.READ)[0]
        return np.where((kid >= 0) & keep[np.maximum(kid, 0)], -V[np.maximum(kid, 0)], unexpanded)

    vs = {r: values(r) for r in RUNGS}
    steps = [*range(1, 128), *range(128, leaves + 1, 8)]
    sofar = np.array([values(r)[:2] for r in steps])
    top = np.zeros(F.size, int)
    for n in range(1, F.size):
        top[n] = n if F.parent[n] == 0 else top[F.parent[n]]
    moves = [
        dict(san=san[u], uci=u, p=round(float(F.ep[i]), 4), stockfish=stock[san[u]],
             v={str(r): round(float(vs[r][i]), 3) for r in RUNGS},
             leaves={str(r): int(((top == kid[i]) & (cost <= r)).sum()) for r in RUNGS})
        for i, u in enumerate(root)
    ]  # fmt: skip
    lines = {}
    for n in range(1, F.size):
        if F.depth[n] <= 4:
            chain, j = [], n
            while j:
                chain.append(MOVES[F.token[j] - MOVE_START])
                j = F.parent[j]
            b_ = bd.copy()
            lines[str(n)] = " ".join(b_.san_and_push(chess.Move.from_uci(u)) for u in chain[::-1])
    grown = dict(
        parent=F.parent[: F.size].tolist(),
        depth=F.depth[: F.size].tolist(),
        cost=F.cost[: F.size].tolist(),
        terminal=F.terminal[: F.size].astype(int).tolist(),
        prior=[round(float(q), 4) for q in F.prior[: F.size]],
        white_q=[
            round(float(q), 3)
            for q in (F.wdl[: F.size, 0] - F.wdl[: F.size, 2]) * np.where(F.depth[: F.size] % 2 == 0, 1, -1)
        ],
        lines=lines,
    )

    # the same game asked at each rating, as the bot would search it, then the game's own, timed
    bot = treers.KL(C.GROW, C.READ, views=C.VIEWS)
    by_rating = []
    for rating in BINS:
        gr = game(rating, rating)
        tm = gr.position()[3]
        found, prior, q = bot(gr, leaves)
        bt = float(C.beta(rating, tm, FORMATS[speed]))
        pi = C.tilt(prior, q, bt)
        by_rating.append(dict(
            rating=rating, beta=round(bt, 3), think=round(float(tm @ C.SECONDS), 2),
            p={san[m]: round(float(v), 4) for m, v in zip(found, prior)},
            v={san[m]: round(float(v), 3) for m, v in zip(found, q)},
            pi={san[m]: round(float(v), 4) for m, v in zip(found, pi)},
        ))  # fmt: skip
    t0 = monotonic()
    found, _, q = bot(game(white, black), leaves)
    took = monotonic() - t0
    gap = float(np.abs(np.array([q[found.index(u)] for u in root]) - vs[leaves]).max())

    # the think-time draws as behaviour.think makes them, the rung as calibration.affordable picks it
    rng = np.random.default_rng(20261103)
    reserve = behaviour.PARAMETERS["guard"]["reserve"]
    draws = np.array([behaviour.think(time, rng, float(meta[7]), inc, PLY, 0.3) for _ in range(100000)])
    rung = np.array([C.affordable(float(meta[7]), reserve, t - C.MARGIN) for t in draws])
    share = {str(r): round(float((rung == r).mean()), 4) for r in (0, *RUNGS)}
    beta = float(C.beta(meta[4], time, FORMATS[speed]))
    pi_at = {str(r): C.tilt(F.ep[:E], vs[r], beta) for r in RUNGS}
    played = share["0"] * F.ep[:E] + sum(share[str(r)] * pi_at[str(r)] for r in RUNGS)

    return dict(
        _src=f"Ply {PLY} of a September 2026 game of the calibration set (row {INDEX} of positions-human.npz), picked by hand among the positions where a move Stockfish calls a blunder holds at least 20% of the prediction and the tilt removes it, replayed on the released Allie 2.1 (release-2.1/hf: int8, Rust engine) with the bot's settings (calibration.py): treers.KL, the bot's searcher, for by_rating and search_seconds; kl.py's forest for the tree and each rung's values (rung r: the nodes of cost <= r); forest_vs_bot: their largest difference in a root value. values_so_far: the two moves' values on the log-odds scale, arctanh(0.98 (W - L)), half the log odds, at each leaf count in leaf_steps; the video reads them as White's expected score, (1 + tanh(v) / 0.98) / 2. KL search: src/allie/search/kl.py, after piKL (Jacob et al., 2022). stockfish: calib-sept/mpv-human, depth 10, cp for White. white_q: the node's own read, W - L for White, under the view. search_seconds is wall time on the measuring node: it moves with the node's load (5.6-6.8 s seen).",
        ply=PLY, threads=THREADS, cpu=next(s.split(":")[1].strip() for s in open("/proc/cpuinfo") if s.startswith("model name")), fen=bd.fen(), format=speed, time_control=[base, inc], white=white, black=black,
        clock=int(meta[7]), human=san[MOVES[int(meta[8]) - MOVE_START]],
        beta=round(beta, 3), wdl=[round(float(v), 3) for v in wdl], think_mean=round(float(time @ C.SECONDS), 2),
        time_probs=[round(float(v), 5) for v in time],
        moves=moves, tree=grown, leaf_steps=steps, values_so_far={san[root[i]]: [round(float(v), 3) for v in sofar[:, i]] for i in range(2)}, by_rating=by_rating, search_seconds=round(took, 2), forest_vs_bot=round(gap, 4),
        _src_think="behaviour.think drawn 100,000 times at this position's clock, the rung by calibration.affordable at COST. played: each root move's chance over those draws, rung by rung, at the game's own beta.",
        rung_share=share,
        think_quantiles={str(q): round(float(np.quantile(draws, q)), 2) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        pi_at={r: {san[u]: round(float(v), 4) for u, v in zip(root, pi)} for r, pi in pi_at.items()},
        played={san[u]: round(float(v), 4) for u, v in zip(root, played)},
    )  # fmt: skip


def sources():
    cells = json.loads((R / "video-refresh/cells.json").read_text())["sept"]
    plot = json.loads((R / "calib-fit/win21-sept.plot.json").read_text())["cells"]
    sept = next(iter(json.loads((R / "calib-fit/win21-sept.json").read_text())["scores"].values()))
    sept20 = next(iter(json.loads((R / "calib-fit/var20-sept.json").read_text())["scores"].values()))
    held = json.loads((R / "calib-fit/bot21-fit-july.json").read_text())["thinkx"]["outside_held"]
    rows = json.loads((R / "load-own/21-four-c1.0.json").read_text())["rows"]
    median = lambda v, p=50: round(float(np.percentile(v, p)), 2)  # noqa: E731
    out = {}
    for f in FMTS:
        for c in BINS:
            k, v, h = f"{f}/{c}", cells["cells"][f"{f}/{c}"], plot[f"{f}/{c}"]
            out[k] = dict(
                human=[round(h["human"]["accuracy"], 2), round(100 * h["human"]["blunder"], 2)],
                human_se=[round(h["human"]["accuracy_se"], 3), round(100 * h["human"]["blunder_se"], 3)],
                raw=[round(h["raw"]["accuracy"], 2), round(100 * h["raw"]["blunder"], 2)],
                calibrated=[round(h["calibrated"]["accuracy"], 2), round(100 * h["calibrated"]["blunder"], 2)],
                raw_elo=[round(v["raw"][0]), round(v["raw"][2])],
                elo=[round(v["calibrated"][0]), round(v["calibrated"][2])],
                ce=round(v["calibrated"][4], 4),
                beta=round(v["beta_median"], 3),
                leaves=v["median_leaves"],
                think=round(v["think_median"], 1),
                no_search=round(v["no_search"], 3),
                n=v["n"],
            )
    misses = [
        dict(cell=c, test=t, z=m[t], elo=m["elo"])
        for c, m in cells["outside"].items()
        for t in ("acc", "blunder")
        if abs(m[t]) > 1.96
    ]
    assert len(misses) == sept["tests_outside"]
    variants = {
        k: v["tests_outside"] for k, v in json.loads((R / "calib-fit/var21b-july.json").read_text())["fits"].items()
    }
    one_rule = [v for k, v in variants.items() if not k.endswith("/format")]
    candidates = [
        v["tests_outside"] for v in json.loads((R / "calib-fit/var21-sept.json").read_text())["scores"].values()
    ]
    assert len(variants) == 8 and min(variants.values()) == 5
    elo = [cells["cells"][f"classical/{c}"]["elo"] for c in BINS]
    curve = [plot[f"classical/{c}"]["human"] for c in BINS]
    raw24, at24 = plot["classical/2400"]["raw"], elo[BINS.index(2400)]
    return dict(
        _src_cells="September 2026 games, none in the rule's fit (197,191 positions; earlier candidates were scored on them, tests.candidates_sept): release-2.1/calib-fit/win21-sept.plot.json (accuracy, blunder rate, the README figure's data) (blunder_points: calib_fit.move_metrics) and release-2.1/video-refresh/cells.json (Elo gaps on the humans' curve: accuracy, blunder rate; median beta and leaves at COST; share with no search). stockfish_depth: analysis/elo_strength/calib_mpv.py. rms: per time control, debiased, cells.json; allie20: calib-fit/var20-sept.json, the same curve and games. raw_read: classical 2400, Allie alone against the humans on accuracy and blunder rate, not from games played. slope: through the human curve's local slope, the cell's raw_elo. direct: where its numbers sit on the human curve itself, between cells at their mean rating, the blunder rate in logs.",
        cells=out,
        misses=misses,
        blunder_points=15,
        stockfish_depth=10,
        rms={k: {f: round(v) for f, v in cells["rms"][k].items()} for k in ("raw", "calibrated")} | dict(allie20=sept20["rms"]),
        raw_read=dict(
            slope=out["classical/2400"]["raw_elo"],
            direct=[
                round(float(np.interp(raw24["accuracy"], [h["accuracy"] for h in curve], elo)) - at24),
                round(float(np.interp(-np.log(raw24["blunder"]), [-np.log(h["blunder"]) for h in curve], elo)) - at24),
            ],
        ),
        _src_tests="Allie 2.1: calib-fit/win21-sept.json (11), calib-fit/bot21-fit-july.json outside_held (5). Allie 2.0: calib-fit/var20-sept.json (8), src/allie/lichess/calibration.py at commit cad41559 (2 held out). Rules and parameters: docs/DETAILS.md, Calibrated play. variants, one_rule_july: calib-fit/var21b-july.json, its 8 variants' held-out misses and the range of its four one-rule variants. candidates_sept: calib-fit/var21-sept.json, the three earlier candidates scored on September. ce: September, nats against the raw prediction, the same files. bullet_2000_july_z: bot21-fit-july.json, the halves pooled.",
        tests=dict(of=80, chance=4, sept=sept["tests_outside"], july=sum(abs(m[t]) > 1.96 for m in held.values() for t in ("acc", "blunder")),
                   allie20=dict(july=2, sept=sept20["tests_outside"], rules=1, parameters=4), rules=4, parameters=20, variants=len(variants), one_rule_july=[min(one_rule), max(one_rule)], candidates_sept=[min(candidates), max(candidates)]),
        ce=dict(allie21=sept["ce"], allie20=sept20["ce"]),
        bullet_2000_july_z=held["bullet/2000"]["ce"],
        _src_rule="src/allie/lichess/calibration.py; the fit's size from its docstring (July 2026 games).",
        rule=dict(rules=[list(r) for r in C.RULES], ladder=list(RUNGS), cost=C.COST, margin=C.MARGIN, view=C.VIEWS[0], fit_positions=198794),
        _src_live="release-2.1/load-own/21-four-c1.0.json (four games at a human pace, docs/DETAILS.md Inference). ms_per_move: a move without search, README. batched_leaf_ms: a leaf in a search, batched: 5.2 ms in the four-game test (DETAILS), 5.7 ms in this position's search (position.search_seconds over 1,024 leaves); the rule budgets rule.cost. offset: src/allie/lichess/README.md, measured on Allie 2.0 only (2,223 games, 2026-10-06 to 10-17, 53 +- 8 Elo above its rating in games decided on the board); Allie 2.1's own live measure has not happened.",
        live=dict(search_median=median([r["searched"] for r in rows]), search_p90=median([r["searched"] for r in rows], 90),
                  by_format={f: median([r["searched"] for r in rows if r["speed"] == f]) for f in ("bullet", "blitz", "classical")},
                  decisions=len(rows), ms_per_move=14.3, batched_leaf_ms=[5, 6], offset=-50, offset_measured=53, offset_se=8, offset_games=2223),
    )  # fmt: skip


if __name__ == "__main__":
    path = HERE / "data.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    for name in sys.argv[1:] or ("sources", "aggregates", "position"):
        data[name] = globals()[name]()
        path.write_text(json.dumps(data, indent=1) + "\n")
