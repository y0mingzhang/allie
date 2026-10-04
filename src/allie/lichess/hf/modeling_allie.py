"""Allie 2.0 for transformers (trust_remote_code): plain PyTorch, CPU or GPU.

    from transformers import AutoModel
    model = AutoModel.from_pretrained("yimingzhang/allie-2.0", trust_remote_code=True)
    model.predict(["e2e4", "e7e5", "g1f3"], white_elo=1800, black_elo=1750, time_control="180+2")

The network, its key-value cache and the chess inputs are api.py, model.py and tokens.py, the
same files as the allie package's allie.lichess (GITHUB_URL).
"""

from pathlib import Path

import torch
from transformers import PreTrainedModel

from .api import Allie
from .configuration_allie import AllieConfig
from .model import Model


class AllieModel(PreTrainedModel):
    """predict(), analyze() and play() as allie.lichess.api.Allie's. The weights live in
    self.allie.model (BF16 by default, or int8 on CPU), not in torch parameters."""

    config_class = AllieConfig

    def __init__(self, config, allie=None):
        super().__init__(config)
        self.allie = allie

    @classmethod
    def from_pretrained(cls, name, *args, config=None, device=None, int8=False, experts=None,
                        torch_dtype=None, **kwargs):  # fmt: skip
        """device: default CUDA if available. int8: int8 weights on CPU (half the memory,
        faster). experts: route through only this many of the 16 experts (faster)."""
        path = Path(name)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            hub = {k: kwargs[k] for k in ("revision", "cache_dir", "token", "local_files_only",
                                          "force_download") if k in kwargs}  # fmt: skip
            path = Path(snapshot_download(name, allow_patterns=["*.json", "*.safetensors"], **hub))
        if config is None:
            config = AllieConfig.from_pretrained(path)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch_dtype if isinstance(torch_dtype, torch.dtype) else torch.bfloat16
        return cls(config, Allie(Model(path, device, dtype, experts, int8)))

    def analyze(self, *args, **kwargs):
        return self.allie.analyze(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.allie.predict(*args, **kwargs)

    def play(self, *args, **kwargs):
        return self.allie.play(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("use predict(), analyze() or play()")
