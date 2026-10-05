import pytest
import torch

from allie.lichess import cli
from allie.lichess.config import Config


def test_fast_backend_leaves_torch_one_thread(tiny_path):
    """The kernels' pool holds `threads` CPUs; torch's own small ops must not add threads that fight it."""
    from allie.lichess import fast

    try:
        fast.library()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no fast kernels: {e}")
    before = torch.get_num_threads()
    try:
        m = cli.model(Config(model=str(tiny_path), threads=2, backend="fast"))
        assert m.fast is not None and m.fast.threads == min(2, len(fast.cpus())) and torch.get_num_threads() == 1
        cli.model(Config(model=str(tiny_path), threads=2, backend="torch"))
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(before)
