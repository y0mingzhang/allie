"""The Rust searches (allie_fast.Coverage, allie_fast.KL) against the golden Python/C++ chain (search_golden.py's
.npz): every position's searches replayed from the stored root and leaf logits, with the handles of every call,
the moves, values and output probabilities compared bit for bit; with --cpp also today's chain driven by the same
logits (the C++ tree's compact arrays and values, kl.py's forest and values, tree.py Nodes' boards and clocks),
and the searches' own cost per leaf (no network), Rust against C++/Python. No model runs; a login node does.

usage: search_equal.py [--golden FILE] [--pgns GLOB] [--searchers coverage,kl] [--budgets 32,128] [--positions N] [--cpp]
"""

import argparse
import glob
import json
import sys
import time
from types import SimpleNamespace

import allie_fast
import numpy as np
from scipy.special import softmax
from search_profile import load_game, read_pgn

from allie.lichess import tree
from allie.lichess.engine import Game
from allie.lichess.tokens import advance
from allie.search.board import advance_clocks, predicted_seconds, root_other_previous
from allie.search.policy import policy

R = "/home/yimingz3/src/allie/results/lichess-search/golden-cpp.npz"
PGNS = "/data/group_data/dei-group/yimingz3/allie/lichess/live-games/*.pgn"
GROW, READ = tree.KL.GROW, tree.KL.READ


class Bookkeeping:
    """tree.Nodes.__call__ without the network: the per-node boards and clocks it keeps, from the logits."""

    def __init__(self, tokens, feats, inc, root, cap, predicted=True):
        self.parent, self.length, self.token = (
            np.full(cap, -1, np.int64),
            np.zeros(cap, np.int64),
            np.zeros(cap, np.int64),
        )
        self.board, self.feats, self.other, self.elapsed = (
            [None] * cap,
            np.full((cap, 3), -1.0),
            np.full(cap, -1.0),
            np.zeros(cap),
        )
        self.length[0], self.board[0], self.feats[0] = (
            len(tokens),
            allie_fast.encode_boards(tokens)[-1].tobytes(),
            feats[-1],
        )
        self.other[0], self.inc, self.predicted = (
            root_other_previous(tokens, feats, inc),
            inc,
            predicted,
        )
        if predicted:
            self.elapsed[0] = predicted_seconds(root[None])[0]

    def __call__(self, h, z):
        ids, parents, toks, lengths = np.asarray(h, np.int64).T
        self.parent[ids], self.token[ids], self.length[ids] = parents, toks, lengths
        for i, p, t in zip(ids, parents, toks):
            self.board[i] = advance(self.board[p], int(t))
        f, o = advance_clocks(
            self.feats[parents],
            self.other[parents],
            self.length[parents],
            np.full(len(ids), self.inc),
            self.elapsed[parents],
        )
        self.feats[ids], self.other[ids] = f, o
        if self.predicted:
            self.elapsed[ids] = predicted_seconds(z)

    def same(self, rs, handles):
        """The Rust tree's boards and clocks equal these at the evaluated nodes (the root's and the handles')."""
        ids = [0, *handles[:, 0].tolist()]
        boards = np.frombuffer(b"".join(self.board[i] for i in ids), np.uint8).reshape(
            -1, 68
        )
        return np.array_equal(
            rs.feats(ids), self.feats[ids].astype(np.float32)
        ) and np.array_equal(rs.boards(ids), boards)


class Stored:
    """A bridge replaying stored leaf logits in call order, with Nodes' bookkeeping."""

    def __init__(self, root, leaves, book):
        self.root_logits, self.leaves, self.book, self.lo = root[None], leaves, book, 0

    def __call__(self, h):
        z = self.leaves[self.lo : self.lo + len(h)].astype(np.float64)
        self.lo += len(h)
        self.book(h, z)
        return z


def drive(select, update, handles, calls, leaves):
    """Run a Rust search by its select/update protocol on the golden leaves, checking every call's handles against
    the golden rows; the first mismatch, or None."""
    lo, call = 0, 0
    while (h := select()) is not None:
        if not len(h):
            continue
        want = (
            handles[lo : lo + calls[call]]
            if call < len(calls)
            else np.zeros((0, 4), np.int64)
        )
        if h.shape != want.shape or not np.array_equal(h, want):
            return dict(call=call, rust=h[:4].tolist(), golden=want[:4].tolist())
        update(leaves[lo : lo + len(h)].astype(np.float64))
        lo, call = lo + len(h), call + 1
    if call != len(calls) or lo != len(leaves):
        return dict(call=call, calls=len(calls), leaves_used=lo, leaves=len(leaves))
    return None


def coverage(g, tokens, root, feats, inc, budget, par, cell, cpp):
    """(outputs, first mismatch, Rust seconds, reference seconds, reference mismatches) of a coverage replay."""
    handles, calls, leaves = g["handles"], g["calls"], g["leaf_logits"]
    t0 = time.perf_counter()
    rs = allie_fast.Coverage(
        tokens.tolist(), root, feats, inc, budget, par["cpuct"], "predicted"
    )
    bad = drive(
        lambda: None if rs.done else rs.select(), rs.update, handles, calls, leaves
    )
    if bad:
        return None, bad, time.perf_counter() - t0, None, []
    legal = allie_fast.Position.from_tokens(tokens.tolist()).legal()
    ids = np.array([legal], np.int64) - 378
    bk = par["backup"]
    q = rs.reduce(np.log(bk["tau"]), bk["exponent"], bk["count_scale"], budget, ids)[0]
    rust = time.perf_counter() - t0
    logits = root[None, 378:2346][np.arange(1)[:, None], ids].astype(float)
    params = (
        par["old_parameters"]["unchanged"]
        if budget in (64, 1000)
        else par["budget_policies"][str(budget or 128)]
    )
    prob = policy(
        [dict(prefix=tokens.tolist(), cell=cell)],
        root[None],
        q,
        ids,
        np.ones_like(ids, bool),
        np.array([feats[-1, 0]]),
        params,
    )
    out = dict(
        moves=np.array(legal, np.int32),
        probabilities=prob[0],
        prior=softmax(logits, axis=1)[0],
        q=q[0],
    )
    if cpp is None:
        return out, None, rust, None, []
    native, value = cpp
    t0 = time.perf_counter()
    cx = native.Coverage([tokens.tolist()], root[None], [budget], [par["cpuct"]], 1)
    book, lo = Bookkeeping(tokens, feats, inc, root, 4 * budget + 256), 0
    while not cx.done:
        h = cx.select()
        if len(h):
            z = leaves[lo : lo + len(h)].astype(np.float64)
            book(h, z)
            cx.update(z)
            lo += len(h)
    theirs = cx.compact()
    qc = value.Backup(theirs, budget, bk["count_scale"]).reduce(
        np.log(bk["tau"]), bk["exponent"], ids.astype(np.int32)
    )[0]
    ref = time.perf_counter() - t0
    mine = rs.compact()
    wrong = [k for k in theirs if not np.array_equal(mine[k], theirs[k])] + (
        ["q"] if not np.array_equal(q, qc) else []
    )
    return (
        out,
        None,
        rust,
        ref,
        wrong + (["bookkeeping"] if not book.same(rs, handles) else []),
    )


def kl_search(g, tokens, root, feats, inc, budget, cpp):
    """As coverage(), for the KL search: kl.py's forest on the same logits is the reference."""
    handles, calls, leaves = g["handles"], g["calls"], g["leaf_logits"]
    cap = 2 * budget + 256
    t0 = time.perf_counter()
    rs = allie_fast.KL(tokens.tolist(), [root], None, feats, inc, cap, "zero")
    bad = drive(
        lambda: rs.select(budget, **GROW),
        lambda z: rs.update([z]),
        handles,
        calls,
        leaves,
    )
    if bad:
        return None, bad, time.perf_counter() - t0, None, []
    moves, prior, q = rs.root_q(**READ)
    rust = time.perf_counter() - t0
    out = dict(moves=np.array(moves, np.int32), prior=prior, q=q)
    if cpp is None:
        return out, None, rust, None, []
    from allie.search import kl
    from allie.search.native import from_prefix

    t0 = time.perf_counter()
    book = Bookkeeping(tokens, feats, inc, root, cap, predicted=False)
    F = kl.Forest(Stored(root, leaves, book), [from_prefix(tokens)], [len(tokens)], cap)
    kl.grow(F, budget, **GROW)
    V = kl.backup(F.view(), **READ)[0]
    m2, p2, q2 = kl.root_q(F, V, 0)
    q2 = np.where(
        np.isnan(q2), np.arctanh(READ["squash"] * np.clip(F.value[0], -1, 1)), q2
    )
    ref = time.perf_counter() - t0
    wrong = [
        k
        for k, (x, y) in dict(moves=(moves, m2), prior=(prior, p2), q=(q, q2)).items()
        if not np.array_equal(x, y)
    ]
    if (
        not np.array_equal(rs.values(**READ)[0], V)
        or F.size != rs.size
        or F.spent[0] != rs.spent
    ):
        wrong.append("forest")
    return (
        out,
        None,
        rust,
        ref,
        wrong + (["bookkeeping"] if not book.same(rs, handles) else []),
    )


def golden_outputs(f, tag, name):
    parts = ("moves", "prior", "q") + (("probabilities",) if name == "coverage" else ())
    return {k: f[f"{tag}/{k}"] for k in parts}


def outputs_of(res, name):
    """A searcher's result as the golden stores it."""
    from allie.lichess.tokens import MOVE_ID

    out = dict(moves=np.array([MOVE_ID[mv] for mv in res[0]], np.int32))
    out.update(
        zip(
            ("probabilities", "prior", "q") if name == "coverage" else ("prior", "q"),
            (np.asarray(x, np.float64) for x in res[1:]),
        )
    )
    return out


def compare(got, want):
    """(identical, same move list, the largest absolute difference over the arrays)."""
    same = np.array_equal(got["moves"], want["moves"])
    diffs = [
        float(np.abs(got[k] - want[k]).max())
        if got[k].shape == want[k].shape
        else np.inf
        for k in want
        if k != "moves"
    ]
    return same and max(diffs) == 0.0, same, max(diffs) if same else np.inf


def native(a, f, keys, pgns):
    """The all-Rust searches on the golden positions: see the module docstring."""
    import torch

    from allie.lichess import treers
    from allie.lichess.engine import Engine
    from allie.lichess.model import Model

    torch.set_num_threads(1)
    m = Model(a.model, "cpu", torch.bfloat16, None, True, "rust", a.threads)
    engine, srv = Engine(m), m.fast.server
    searchers = dict(coverage=treers.Coverage(), kl=treers.KL())
    budgets = [int(b) for b in a.budgets.split(",")]
    modes = [("chunk8", 8)] + ([("whole", 0)] if a.whole else [])
    rows = {}
    for name in a.searchers.split(","):
        for b in budgets:
            for mode, _ in modes:
                rows[f"{name}/{b}/{mode}"] = dict(
                    searches=0,
                    identical=0,
                    same_moves=0,
                    max_abs=0.0,
                    leaves=0,
                    seconds=0.0,
                    first=[],
                )
    roots, t0 = dict(positions=0, differing=0), time.perf_counter()
    for key in keys:
        stem, ply = key.split("@")
        game = load_game(engine, read_pgn(pgns[stem], int(ply)))
        assert game.tokens == f[key + "/tokens"].tolist(), key
        roots["positions"] += 1
        roots["differing"] += not np.array_equal(
            game.sync().double().numpy(), f[key + "/root_logits"]
        )
        for name in a.searchers.split(","):
            for b in budgets:
                want = golden_outputs(f, f"{key}/{name}/{b}", name)
                for mode, chunk in modes:
                    srv.chunk = chunk
                    t1 = time.perf_counter()
                    res = searchers[name](game, b)
                    r = rows[f"{name}/{b}/{mode}"]
                    r["seconds"] += time.perf_counter() - t1
                    r["leaves"] += game.last_search["evaluated"]
                    identical, same, mx = compare(outputs_of(res, name), want)
                    r["searches"] += 1
                    r["identical"] += identical
                    r["same_moves"] += same
                    r["max_abs"] = max(r["max_abs"], mx)
                    if not identical and len(r["first"]) < 4:
                        r["first"].append(
                            dict(key=key, max_abs=mx, same_moves=bool(same))
                        )
        if roots["positions"] % 20 == 0:
            print(
                json.dumps(
                    dict(
                        done=roots["positions"],
                        seconds=round(time.perf_counter() - t0, 1),
                        identical={
                            k: f"{r['identical']}/{r['searches']}"
                            for k, r in rows.items()
                        },
                    )
                ),
                flush=True,
            )
    for r in rows.values():
        r["us_per_leaf"] = round(1e6 * r.pop("seconds") / max(r["leaves"], 1), 1)
    srv.chunk = 0
    views = ("true", "r2800")
    vrows = dict(
        positions=0,
        identical=0,
        same_moves=0,
        max_abs=0.0,
        merged_identical=0,
        merged_max_abs=0.0,
    )
    for key in keys[: a.views_positions]:
        stem, ply = key.split("@")
        game = load_game(engine, read_pgn(pgns[stem], int(ply)))
        want = outputs_of(tree.Coverage(views=views)(game, 128), "coverage")
        identical, same, mx = compare(
            outputs_of(
                treers.Coverage(views=views, merged=False)(game, 128), "coverage"
            ),
            want,
        )
        vrows["positions"] += 1
        vrows["identical"] += identical
        vrows["same_moves"] += same
        vrows["max_abs"] = max(vrows["max_abs"], mx)
        identical, same, mx = compare(
            outputs_of(treers.Coverage(views=views)(game, 128), "coverage"), want
        )
        vrows["merged_identical"] += identical
        vrows["merged_max_abs"] = max(vrows["merged_max_abs"], mx)
    out = dict(
        model=a.model,
        threads=m.fast.threads,
        isa=m.fast.isa,
        roots=roots,
        searches=rows,
        views=vrows,
        server=srv.stats(),
        seconds=round(time.perf_counter() - t0, 1),
    )
    print(json.dumps(out), flush=True)
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(json.dumps(out) + "\n")
    engine.close()
    bad = (
        any(
            r["identical"] != r["searches"]
            for k, r in rows.items()
            if k.endswith("chunk8")
        )
        or vrows["identical"] != vrows["positions"]
    )
    return int(bad or roots["differing"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--golden", default=R)
    p.add_argument("--pgns", default=PGNS)
    p.add_argument("--searchers", default="coverage,kl")
    p.add_argument("--budgets", default="32,128")
    p.add_argument("--positions", type=int, default=10**9)
    p.add_argument(
        "--cpp",
        action="store_true",
        help="also drive today's chain: its arrays compared, per-leaf timing",
    )
    p.add_argument(
        "--native",
        action="store_true",
        help="the all-Rust searches with the real model (a CPU job)",
    )
    p.add_argument("--model")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument(
        "--whole",
        action="store_true",
        help="--native: also one step per call (the loop's own batching)",
    )
    p.add_argument("--views-positions", type=int, default=20)
    p.add_argument("--out")
    a = p.parse_args()
    f = np.load(a.golden)
    keys = sorted({k.split("/")[0] for k in f.files})[: a.positions]
    pgns = {s.split("/")[-1][:-4]: s for s in glob.glob(a.pgns)}
    if a.native:
        return native(a, f, keys, pgns)
    cpp = None
    if a.cpp:
        from allie.search.native import load

        cpp = load(), load("value")
    par = json.loads(tree.CALIBRATION.read_text())
    for b, key in tree.POLICY.items():
        par["budget_policies"].setdefault(str(b), par["budget_policies"][key])
    budgets = [int(b) for b in a.budgets.split(",")]
    report, first = {}, []
    for name in a.searchers.split(","):
        report[name] = dict(
            positions=0,
            searches=0,
            calls=0,
            handles=0,
            nodes=0,
            mismatches=0,
            reference_mismatches=0,
            seconds=0.0,
            reference_seconds=0.0,
        )
    for key in keys:
        stem, ply = key.split("@")
        g = read_pgn(pgns[stem], int(ply))
        tokens, root, feats = (
            f[key + "/tokens"],
            f[key + "/root_logits"],
            f[key + "/features"],
        )
        inc = int(tokens[2]) - 10 if 10 <= tokens[2] <= 190 else -1
        assert inc == g["inc"] and len(tokens) == 11 + int(ply), (key, inc, g["inc"])
        cell = Game.cell(
            SimpleNamespace(speed=g["speed"]), (g["white"], g["black"])[int(ply) % 2]
        )
        for name in report:
            n = report[name]
            n["positions"] += 1
            for budget in budgets:
                tag = f"{key}/{name}/{budget}"
                golden = {
                    k: f[f"{tag}/{k}"]
                    for k in (
                        "handles",
                        "calls",
                        "leaf_logits",
                        "moves",
                        "prior",
                        "q",
                        *(["probabilities"] if name == "coverage" else []),
                    )
                }
                if name == "coverage":
                    out, bad, rust, ref, wrong = coverage(
                        golden, tokens, root, feats, inc, budget, par, cell, cpp
                    )
                else:
                    out, bad, rust, ref, wrong = kl_search(
                        golden, tokens, root, feats, inc, budget, cpp
                    )
                n["searches"] += 1
                n["calls"] += len(golden["calls"])
                n["handles"] += len(golden["handles"])
                n["seconds"] += rust
                if bad:
                    n["mismatches"] += 1
                    first.append(dict(searcher=name, key=key, budget=budget, **bad))
                    continue
                diff = [k for k in out if not np.array_equal(out[k], golden[k])]
                n["nodes"] += len(golden["handles"]) + 1
                if diff:
                    n["mismatches"] += 1
                    first.append(
                        dict(
                            searcher=name,
                            key=key,
                            budget=budget,
                            outputs=diff,
                            max_abs={
                                k: float(np.abs(out[k] - golden[k]).max()) for k in diff
                            },
                        )
                    )
                if ref is not None:
                    n["reference_seconds"] += ref
                if wrong:
                    n["reference_mismatches"] += 1
                    first.append(
                        dict(searcher=name, key=key, budget=budget, reference=wrong)
                    )
    for name, n in report.items():
        leaves = n.pop("handles")
        n["leaves"] = leaves
        s, r = n.pop("seconds"), n.pop("reference_seconds")
        if leaves:
            n["us_per_leaf"] = dict(
                rust=round(1e6 * s / leaves, 2),
                **(dict(today=round(1e6 * r / leaves, 2)) if cpp else {}),
            )
    print(json.dumps(dict(report, first=first[:6])), flush=True)
    return int(
        any(n["mismatches"] or n["reference_mismatches"] for n in report.values())
    )


if __name__ == "__main__":
    sys.exit(main())
