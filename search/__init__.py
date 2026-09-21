"""Human chess inference. Importing search never starts a worker or loads a model."""
from .algorithm import Search

__all__ = ["Search"]
