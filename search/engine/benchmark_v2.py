"""Reload CPU-only helpers on the resident runner; model/KV math is unchanged."""
import importlib
from . import direct,policy,native_mcts,benchmark_nodes,validate_mcts_gpu


def run(oracle,spec):
    importlib.reload(direct);importlib.reload(policy);importlib.reload(native_mcts)
    importlib.reload(benchmark_nodes);importlib.reload(validate_mcts_gpu)
    oracle.reset();oracle.__class__=direct.DirectOracle
    parity=validate_mcts_gpu.run(oracle,spec)
    return dict(parity=parity,benchmark=benchmark_nodes.run(oracle,spec))
