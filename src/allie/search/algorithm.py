"""Production entry point for the frozen coverage search and repaired Allie.

The caller supplies past tokens and causal clock features, never a target move.
Neural backends implement reset() and handles(prefixes, features, clock_rule).
Search state and neural caches belong to one batch; no cross-game memory.
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .native import from_prefix, load
from .policy import policy, route, output

CALIBRATION = Path(__file__).with_name("calibration.json")


class Search:
    def __init__(self, oracle, *, batch_size=128, threads=4, calibration=None):
        if not 1 <= batch_size <= 128 or not 1 <= threads <= 8:
            raise ValueError("batch_size must be 1..128 and threads 1..8")
        self.oracle, self.batch_size, self.threads = oracle, batch_size, threads
        self.parameters = json.loads(CALIBRATION.read_text()) if calibration is None else calibration
        if not np.isfinite(self.parameters["cpuct"]) or self.parameters["cpuct"] < 0:
            raise ValueError("coverage cpuct must be finite and nonnegative")

    def predict(self, queries, *, method="coverage", budget="adaptive", clock_rule="predicted",
                project=False, allie_alpha=.9, allie_beta=2., allie_cpuct=1.25):
        """Return one distribution per query, in native legal-move order.

        Each query: prefix (11 header tokens + past moves), cell (0..15),
        optional features (len(prefix),3): mover seconds, opponent seconds,
        mover's previous OWN think time. Unknown entries are -1.
        Fixed coverage budgets: 64/128/256/512/1000; legal uses no search.
        Allie is a comparison method with an explicit fixed simulation budget.
        """
        if method not in ("coverage", "legal", "allie") or clock_rule not in ("predicted", "zero"):
            raise ValueError("unknown search method or clock rule")
        if project and (method != "coverage" or budget != 1000):
            raise ValueError("critic projection is calibrated only for fixed coverage budget 1000")
        if method == "coverage" and budget not in ("adaptive", 64, 128, 256, 512, 1000):
            raise ValueError("no frozen calibration for this coverage budget")
        if method == "allie" and (not isinstance(budget, int) or not 0 <= budget <= 1000):
            raise ValueError("Allie requires a fixed budget in 0..1000")
        if method == "allie" and (not np.isfinite([allie_alpha, allie_beta, allie_cpuct]).all()
                                  or allie_alpha <= 0 or allie_beta < 0 or allie_cpuct < 0):
            raise ValueError("Allie alpha must be positive; beta and cpuct nonnegative; all finite")
        rows, features = [], []
        for q in queries:
            prefix = np.asarray(q["prefix"])
            if prefix.ndim != 1 or not np.issubdtype(prefix.dtype, np.integer) or not 11 <= len(prefix) < 1025:
                raise ValueError("prefix must contain 11..1024 integer tokens")
            if ((prefix < 0) | (prefix >= 2350)).any():
                raise ValueError("invalid input token")
            if (prefix[0] != 2348 or not (192 <= prefix[1] < 378 or prefix[1] == 2349)
                    or not (10 <= prefix[2] < 192 or prefix[2] == 2349)
                    or ((prefix[3:11] < 0) | (prefix[3:11] > 9)).any()):
                raise ValueError("header must be START, base-time token, increment token, and eight Elo digits")
            cell = q["cell"]
            if not isinstance(cell, (int, np.integer)) or not 0 <= cell < 16:
                raise ValueError("cell must be an integer in 0..15")
            board = from_prefix(prefix)
            if board.outcome() >= 0:
                raise ValueError("terminal position has no human-move prediction")
            feat = np.asarray(q.get("features", np.full((len(prefix), 3), -1.)), np.float32)
            if feat.shape != (len(prefix), 3) or not np.isfinite(feat).all() or ((feat < 0) & (feat != -1)).any():
                raise ValueError("features must be finite (prefix length,3), seconds >=0 or missing -1")
            rows.append(dict(prefix=prefix.tolist(), cell=int(cell), legal=board.legal()))
            features.append(feat)
        result = []
        for lo in range(0, len(rows), self.batch_size):
            result.extend(self._batch(rows[lo:lo+self.batch_size], features[lo:lo+self.batch_size],
                                      method, budget, clock_rule, project, allie_alpha, allie_beta, allie_cpuct))
        return result

    def _batch(self, rows, features, method, budget, clock_rule, project, alpha, beta, cpuct):
        n = len(rows)
        self.oracle.reset()
        bridge = self.oracle.handles([r["prefix"] for r in rows], features, clock_rule)
        root = np.asarray(bridge.root_logits)
        if root.shape != (n, 2432) or not np.isfinite(root).all():
            raise ValueError("oracle must return finite (batch,2432) logits")
        prefill = self.oracle.new_tokens
        width = max(len(r["legal"]) for r in rows)
        ids = np.zeros((n, width), np.int32)
        mask = np.zeros((n, width), bool)
        for i, r in enumerate(rows):
            ids[i, :len(r["legal"])] = np.array(r["legal"]) - 378
            mask[i, :len(r["legal"])] = True
        logits = root[:, 378:2346][np.arange(n)[:, None], ids].astype(float)
        legal = softmax(np.where(mask, logits, -np.inf), axis=1)
        forced = mask.sum(1) == 1
        cells = np.array([r["cell"] for r in rows])
        par = self.parameters
        allocated = np.zeros(n, int)
        q = np.zeros_like(logits)
        nodes = np.zeros(n, int)
        probability = legal
        if method == "coverage":
            if budget == "adaptive":
                choices = [128, 256, 512, 1000]
                allocated = np.array(choices)[route(cells, None, par["adaptive"], choices)]
            else:
                allocated.fill(budget)
            allocated[forced] = 0
            initial = np.minimum(allocated, 128) if budget == "adaptive" else allocated
            tree = load().Coverage([r["prefix"] for r in rows], root, initial.tolist(),
                                   [par["cpuct"]]*n, self.threads)
            self._advance(tree, bridge)
            if budget == "adaptive":
                tree.grow(allocated.tolist())
                self._advance(tree, bridge)
            compact = tree.compact()
            reducer = load("value")
            if project:
                value = reducer.Projection(compact, 1000)
                value.project(par["projection_strength"])
            else:
                value = reducer.Backup(compact, int(allocated.max()), par["backup"]["count_scale"])
            q = value.reduce(np.log(par["backup"]["tau"]), par["backup"]["exponent"], ids)[0]
            nodes = np.asarray(tree.evals)
            probability = np.zeros_like(q)
            seconds = np.array([f[-1, 0] for f in features])
            for b in np.unique(allocated):
                take = allocated == b
                key = str(b if b else 128)
                params = par["old_parameters"]["lambda3"] if project else (
                    par["old_parameters"]["unchanged"] if budget in (64, 1000) else par["budget_policies"][key])
                probability[take] = policy([rows[i] for i in np.flatnonzero(take)], root[take], q[take],
                                           ids[take], mask[take], seconds[take], params)
        elif method == "allie":
            allocated.fill(budget)
            allocated[forced] = 0
            tree = load().Allie([r["prefix"] for r in rows], root, allocated.tolist(), [cpuct]*n)
            tree.first_prior = tree.preserve_depth = True
            self._advance(tree, bridge)
            for i, (moves, counts, values, prior) in enumerate(tree.summaries()):
                # Native root edges use the same deterministic legal order.
                np.testing.assert_array_equal(moves, ids[i, :mask[i].sum()])
                q[i, :len(moves)] = values
            probability = output(logits, q, mask, alpha, beta, "reverse")
            nodes = np.asarray(bridge.per_root_queries)
        if int(nodes.sum()) != bridge.queries:
            raise RuntimeError("node accounting mismatch")
        return [dict(tokens=r["legal"], probabilities=probability[i, mask[i]].copy(),
                     legal_prior=legal[i, mask[i]].copy(), values=q[i, mask[i]].copy(),
                     root_wdl=softmax(root[i, 2413:2416].astype(float)),
                     nodes=int(nodes[i]), simulations=int(allocated[i]), prefill_tokens=len(r["prefix"]),
                     batch_prefill_tokens=int(prefill), method=method)
                for i, r in enumerate(rows)]

    @staticmethod
    def _advance(tree, bridge):
        while not tree.done:
            h = tree.select()
            if len(h):
                tree.update(bridge(h))
