"""New explicitly named sweep; reload the driver only after its prior task ended."""
import importlib
from . import selection_pilot


def run(oracle,spec):
    return importlib.reload(selection_pilot).run(oracle,spec)
