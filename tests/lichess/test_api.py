import os
from pathlib import Path

import chess
import pytest
import torch

from allie.lichess.api import REPO, Allie, parse_moves, parse_time_control
from allie.lichess.model import Cache, Model, step

from .test_model import inputs
from .test_tokens import random_game


def test_parse_moves():
    uci = ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert parse_moves(uci)[0] == uci
    assert parse_moves(["e4", "e5", "Nf3", "Nc6"])[0] == uci
    assert parse_moves("e4 e5 g1f3 Nc6")[0] == uci
    pgn = (
        "1. e4 { [%clk 0:03:00] } e5 { [%clk 0:02:58] } 2. Nf3 { [%clk 0:02:59] } Nc6 *"
    )
    assert parse_moves(pgn) == (uci, [180, 178, 179, None])
    assert parse_time_control("180+2") == (180, 2) and parse_time_control(None) == (
        None,
        None,
    )
    with pytest.raises(ValueError):
        parse_moves(["e2e5"])


def test_predict_matches_model_and_reuses_cache(tiny):
    allie = Allie(tiny)
    moves = random_game(2, 30)
    for n in (0, 1, 12, 13, 30, 7):  # extends, then rewinds to a shorter prefix
        p = allie.predict(
            moves[:n], 1500, 1700, "180+2", [180 - k // 2 for k in range(n)]
        )
        x = inputs(moves[:n])
        z = step(tiny, [(Cache(tiny), *x)])[0].double()
        b = chess.Board()
        for m in moves[:n]:
            b.push_uci(m)
        legal = [m.uci() for m in b.legal_moves]
        from allie.lichess.tokens import MOVE_ID

        ref = torch.softmax(z[[MOVE_ID[m] for m in legal]], 0)
        assert list(p) == sorted(legal, key=lambda m: -ref[legal.index(m)].item())
        for m, q in p.items():
            assert abs(q - ref[legal.index(m)].item()) < 1e-6
    assert len(allie.sessions) == 1  # one game: every call reused its cache
    assert allie.play(moves[:5], elo=900, temperature=0) == next(
        iter(allie.predict(moves[:5], 900, 900))
    )
    out = allie.analyze(moves[:5], 1500, 1500, "60+0")
    assert (
        len(out["wdl"]) == 3
        and abs(sum(out["wdl"]) - 1) < 1e-6
        and out["think_time"] > 0
    )


def test_transformers_remote_code(tiny_path, tmp_path):
    transformers = pytest.importorskip("transformers")
    from allie.lichess.hub import build

    (tmp_path / "export").mkdir()
    for f in ("config.json", "model.safetensors"):
        (tmp_path / "export" / f).symlink_to(tiny_path / f)
    out = build(tmp_path / "export", tmp_path / "hf")
    model = transformers.AutoModel.from_pretrained(
        str(out), trust_remote_code=True, device="cpu", torch_dtype=torch.float32
    )
    ref = Allie(Model(tiny_path, dtype=torch.float32))
    game = "1. e4 e5 2. Nf3 Nc6 3. Bb5"
    assert model.predict(game, 1800, 1750, "180+2") == ref.predict(
        game, 1800, 1750, "180+2"
    )


@pytest.mark.skipif(
    not os.environ.get("ALLIE_EXPORT"), reason="set ALLIE_EXPORT to a local export to compare"
)
def test_hub_weights_match_local_export():
    """Downloads the 11 GB release and compares every tensor with a local export."""
    from huggingface_hub import hf_hub_download

    from allie.lichess.model import read

    local = Path(os.environ["ALLIE_EXPORT"]) / "model.safetensors"
    hub = Path(hf_hub_download(os.environ.get("ALLIE_REPO", REPO), "model.safetensors"))
    for (a, x), (b, y) in zip(read(local), read(hub), strict=True):
        assert a == b and torch.equal(x, y), a
