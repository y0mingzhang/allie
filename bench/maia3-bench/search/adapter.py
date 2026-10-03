"""Run main's frozen search (search/) on the Maia-3 benchmark positions.

Main's search/ is imported, never modified. Two adapters:
- load(): search.model.load_checkpoint with one extra arch key allowed, fp32_small_masters, which
  only moves optimizer masters (modded_medium.create_model keeps the model weights BF16).
- Bench.predict(): Search._batch at any fixed budget; 5/8/25 use the frozen 128-simulation policy,
  460 the frozen 512 one (calibration.json has no others).
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/yimingz3/src/allie")
sys.path.insert(0, str(REPO))
import search.model as sm  # noqa: E402
from search import Search  # noqa: E402
from search.native import from_prefix  # noqa: E402

G = Path("/data/group_data/dei-group/yimingz3/allie/strat-eval-v1")
DATA = Path("/data/group_data/dei-group/yimingz3/allie/maia3-bench")
POLICY = {5: "128", 8: "128", 25: "128", 128: "128", 460: "512"}


def load(path, device="cpu"):
    """The sweep source also dropped the clock/Elo token tables, unused at feats=3 (the port
    reads them only under config clock/elo): zero tables fill their slots."""
    real = torch.load

    def patched(*a, **k):
        state = real(*a, **k)
        if "model" in state:
            state["config"]["arch"].pop("fp32_small_masters", None)
            assert not state["config"].get("clock") and not state["config"].get("elo")
            w = state["model"]["embed.weight"]
            for key in ("clock_embed.weight", "elo_embed.weight"):
                state["model"].setdefault(key, w.new_zeros(64, w.shape[1]))
        return state

    sm.torch.load = patched
    try:
        return sm.load_checkpoint(path, device)
    finally:
        sm.torch.load = real


def export(checkpoint, out):
    """search.export with load() in place of load_checkpoint."""
    from safetensors.torch import save_file

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    model, source = load(checkpoint)
    config = dict(
        model_type="allie_chess",
        allie=vars(model.config),
        auto_map={"AutoConfig": "configuration_allie.AllieConfig"},
        architectures=["AllieShipForCausalLM"],
        torch_dtype="bfloat16",
    )
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy(REPO / "search/configuration_allie.py", out / "configuration_allie.py")
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()},
        str(out / "model.safetensors"),
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (out / "provenance.json").write_text(
        json.dumps(dict(checkpoint=str(source), sha256=digest), indent=2) + "\n"
    )


def positions():
    """Benchmark positions: prefix tokens, causal features, cell, target token, game, band."""
    with np.load(G / "strat.npz") as z:
        rows = z["rows"].astype(np.int64)
    with np.load(G / "feats.npz") as z:
        feats = z["feats"]
    with np.load(DATA / "legal.npz") as z:
        pos, target = z["pos"], z["target"]
    with np.load(DATA / "games.npz") as z:
        s = z["sel"][z["keep"]]
    out = []
    for (r, c), t, (g, ply, cell, *_) in zip(pos, target, s):
        bos = np.flatnonzero(rows[r, : c + 1] == 2348)[-1]
        assert c - bos == 10 + ply
        out.append(
            dict(
                prefix=rows[r, bos : c + 1].tolist(),
                features=feats[r, bos : c + 1].astype(np.float32),
                cell=int(cell),
                target=int(t) + 378,
                game=int(g),
            )
        )
    return out


def dev_positions():
    """build_dev.py's calibration positions, in the same form as positions()."""
    z = dict(np.load(DATA / "search" / "dev-positions.npz"))
    o, tokens, feats = z["offsets"], z["tokens"], z["feats"]
    return [dict(prefix=tokens[a:b].tolist(), features=feats[a:b].copy(), cell=int(c), target=int(t), game=int(g))
            for a, b, c, t, g in zip(o[:-1], o[1:], z["cell"], z["target"], z["game"])]


def subsample(n_per_band, seed=20260924):
    """Stratified: n_per_band positions per blitz band, indices into the 80K sample (sorted)."""
    with np.load(DATA / "games.npz") as z:
        cell = z["sel"][z["keep"]][:, 2]
    rng = np.random.default_rng(seed)
    return np.sort(
        np.concatenate(
            [
                rng.choice(np.flatnonzero(cell == c), n_per_band, replace=False)
                for c in (4, 5, 6, 7)
            ]
        )
    )


class Bench(Search):
    def __init__(self, oracle, **kw):
        super().__init__(oracle, **kw)
        bp = self.parameters["budget_policies"]
        for b, key in POLICY.items():
            bp.setdefault(str(b), bp[key])

    def predict(self, queries, budget, clock_rule="predicted"):
        if budget in ("legal", "adaptive") or budget in (64, 1000):
            method = "legal" if budget == "legal" else "coverage"
            return super().predict(
                queries, method=method, budget=budget, clock_rule=clock_rule
            )
        assert budget in POLICY, budget
        rows, feats = [], []
        for q in queries:
            board = from_prefix(np.asarray(q["prefix"]))
            assert board.outcome() < 0
            rows.append(
                dict(prefix=list(q["prefix"]), cell=q["cell"], legal=board.legal())
            )
            feats.append(np.asarray(q["features"], np.float32))
        out = []
        for lo in range(0, len(rows), self.batch_size):
            out += self._batch(
                rows[lo : lo + self.batch_size],
                feats[lo : lo + self.batch_size],
                "coverage",
                budget,
                clock_rule,
                False,
                0.9,
                2.0,
                1.25,
            )
        return out
