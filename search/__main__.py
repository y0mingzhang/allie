"""Resident JSONL runner; never allocates GPUs or starts background jobs."""
import argparse
import json
import sys
import numpy as np
from .algorithm import Search


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", help="SGLang export directory, or original checkpoint with --backend cpu")
    p.add_argument("--backend", choices=["sglang", "cpu"], default="sglang")
    p.add_argument("--method", choices=["coverage", "legal", "allie"], default="coverage")
    p.add_argument("--budget", default="adaptive", help="adaptive or fixed simulation count")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--capacity", type=int, default=262144)
    p.add_argument("--memory-fraction", type=float, default=.7)
    a = p.parse_args()
    # Library startup messages must not corrupt the JSONL output stream.
    output = sys.stdout
    sys.stdout = sys.stderr
    if a.backend == "cpu":
        from .cpu import CPUOracle
        oracle = CPUOracle(a.model)
    else:
        from .runtime import ShipOracle
        oracle = ShipOracle(a.model, capacity=a.capacity, mem_fraction_static=a.memory_fraction)
    search = Search(oracle, batch_size=a.batch_size, threads=a.threads)
    budget = a.budget if a.budget == "adaptive" else int(a.budget)
    def encode(x):
        return x.tolist() if isinstance(x, np.ndarray) else x
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        result = search.predict(request["positions"], method=a.method, budget=budget)
        print(json.dumps({"predictions": result}, default=encode, allow_nan=False), file=output, flush=True)


if __name__ == "__main__":
    main()
