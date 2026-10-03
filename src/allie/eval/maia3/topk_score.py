"""eval.maia3.score_moe with fewer routed experts per token at inference:
TOPK=K MODE=trunc|renorm torchrun ... -m allie.eval.maia3.topk_score <score_moe arguments>. Every MoE layer selects
its usual top-k by score + bias and keeps the first K of them. trunc: the K kept experts keep the gates they have in
the full top-k (the dropped ones' share is simply missing); renorm: their gates are renormalised as a K-expert
layer's (K**0.5 / sum of the K scores). Shared expert unchanged. Eval only (the patched forward has no training
path). With --source (a pre-package checkpoint), the frozen flat source's MoE is patched, else this package's."""

import os
import runpy
import sys
from pathlib import Path

from allie.eval.maia3 import topk_moe


def main():
    K, MODE = int(os.environ["TOPK"]), os.environ["MODE"]
    assert MODE in ("trunc", "renorm") and 0 < K <= 16
    if "--source" in sys.argv:
        sys.path.insert(
            0, str(Path(sys.argv[sys.argv.index("--source") + 1]).resolve())
        )
        import modded_moe as moe_layer
        import modded_smoe as kernels
    else:
        from allie.model import moe as moe_layer
        from allie.model import moe_kernels as kernels

    topk_moe.counts, topk_moe.routed = moe_layer.counts, kernels.routed
    moe_layer.MoE.forward = topk_moe.forward
    moe_layer.MoE.eval_topk = K, MODE
    runpy.run_module("allie.eval.maia3.score_moe", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
