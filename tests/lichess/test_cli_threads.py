import sys

import pytest
import torch

from allie.lichess import cli
from allie.lichess.config import Config
from allie.lichess.model import Model


def test_fast_backend_leaves_torch_one_thread(tiny_path):
    """The kernels' pool holds `threads` CPUs; torch's own small ops must not add threads that fight it."""
    pytest.importorskip("allie_fast")
    from allie.lichess import fastrs

    before = torch.get_num_threads()
    try:
        m = cli.model(Config(model=str(tiny_path), threads=2, backend="rust"))
        assert m.fast is not None and m.fast.threads == min(2, len(fastrs.cpus())) and torch.get_num_threads() == 1
        cli.model(Config(model=str(tiny_path), threads=2, backend="torch"))
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(before)


def test_without_the_engine(tiny_path, monkeypatch):
    """No Rust engine: backend "rust" refuses to start the bot; the default runs the reference, with a warning."""
    monkeypatch.setitem(sys.modules, "allie_fast", None)  # import allie_fast raises ImportError
    with pytest.raises(ImportError, match="uv sync --extra fast"):
        cli.model(Config(model=str(tiny_path), backend="rust"))
    with pytest.warns(UserWarning, match="using PyTorch"):
        m = Model(tiny_path, dtype=torch.bfloat16)
    assert m.fast is None
