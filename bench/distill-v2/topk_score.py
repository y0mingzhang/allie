"""score_moe.py with fewer routed experts per token at inference: TOPK=K MODE=trunc|renorm torchrun ... topk_score.py
<score_moe arguments>. Every MoE layer selects its usual top-k by score + bias and keeps the first K of them.
trunc: the K kept experts keep the gates they have in the full top-k (the dropped ones' share is simply missing);
renorm: their gates are renormalised as a K-expert layer's (K**0.5 / sum of the K scores). Shared expert unchanged.
Eval only (the patched forward has no training path)."""

import os
import runpy
import sys
from pathlib import Path

K, MODE = int(os.environ["TOPK"]), os.environ["MODE"]
assert MODE in ("trunc", "renorm")
source = Path(sys.argv[sys.argv.index("--source") + 1]).resolve()
sys.path.insert(0, str(source))
import modded_moe  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import topk_moe  # noqa: E402

assert 0 < K <= 16
modded_moe.MoE.forward = topk_moe.forward
modded_moe.MoE.eval_topk = K, MODE
sys.argv[0] = "/home/yimingz3/src/allie/results/recipe10x/maia3-bench/score_moe.py"
runpy.run_path(sys.argv[0], run_name="__main__")
