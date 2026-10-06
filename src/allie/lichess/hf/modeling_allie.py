"""Allie 2.0 for transformers (trust_remote_code), on CPU or GPU.

    from transformers import AutoModel
    model = AutoModel.from_pretrained("yimingzhang/allie-2.0", trust_remote_code=True)
    model.predict(["e2e4", "e7e5", "g1f3"], white_elo=1800, black_elo=1750, time_control="180+2")

The network, its key-value cache, its fast paths and the chess inputs are api.py, model.py, fast.py,
fastrs.py and tokens.py, the same files as the allie package's allie.lichess (GITHUB_URL). On CPU it runs
the PyTorch reference unless the Rust engine is installed (pip install "allie[fast] @ GITHUB_URL"), which
is several times faster.
"""

import torch
from transformers import PreTrainedModel

from .api import Allie, resolve
from .configuration_allie import AllieConfig
from .fast import Graphs  # noqa: F401 - transformers copies only directly imported files
from .fastrs import RustFast  # noqa: F401
from .model import Model
from .tokens import (
    HEADER,  # noqa: F401 - transformers copies only directly imported files
)

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


class AllieModel(PreTrainedModel):
    """predict(), analyze() and play() as allie.lichess.api.Allie's. The weights live in
    self.allie.model (int8 on CPU and BF16 on GPU by default), not in torch parameters, so
    choose the device and precision when loading."""

    config_class = AllieConfig

    def __init__(self, config, allie=None):
        super().__init__(config)
        self.allie = allie

    @classmethod
    def from_pretrained(cls, name, *args, config=None, device=None, device_map=None,
                        int8=None, active_experts=None, backend=None, threads=None,
                        **kwargs):  # fmt: skip
        """device (or a single-device device_map): default CUDA if available. dtype /
        torch_dtype: bfloat16 (default) or float32. int8: int8 weights, the CPU default with
        bfloat16 (half the memory, faster). active_experts: route each token through
        only this many of its 16 experts (faster, slightly less accurate). backend: "rust"
        (the Rust engine, the CPU default when installed; "fast" too) or "torch" (the
        PyTorch reference). threads: the Rust engine's threads."""
        hub = {k: kwargs[k] for k in ("revision", "cache_dir", "token", "local_files_only",
                                      "force_download") if k in kwargs}  # fmt: skip
        path = resolve(name, **hub)
        config = config or AllieConfig.from_pretrained(path)
        if isinstance(device_map, dict):
            raise ValueError(
                "Allie runs on one device: pass device= or a device string"
            )
        device = device or (None if device_map in (None, "auto") else device_map)
        device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype = kwargs.get("dtype", kwargs.get("torch_dtype"))
        for name in ("dtype", "torch_dtype"):  # set on the config when AutoConfig took it
            dtype = dtype or config.__dict__.get(name)
        dtype = (
            torch.bfloat16
            if dtype in (None, "auto")
            else DTYPES.get(str(dtype).removeprefix("torch."))
        )
        if dtype is None:
            raise ValueError("dtype must be bfloat16 or float32")
        if int8 is None:
            int8 = device.type == "cpu" and dtype == torch.bfloat16
        model = Model(path, device, dtype, active_experts, int8, backend, threads)
        return cls(config, Allie(model))

    def analyze(self, *args, **kwargs):
        return self.allie.analyze(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.allie.predict(*args, **kwargs)

    def play(self, *args, **kwargs):
        return self.allie.play(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("use predict(), analyze() or play()")

    def to(self, *args, **kwargs):
        raise NotImplementedError("choose device=, dtype= and int8= in from_pretrained")

    cuda = cpu = half = float = bfloat16 = to

    @property
    def device(self):
        allie = self.__dict__.get("allie")
        return allie.model.device if allie else torch.device("cpu")

    @property
    def dtype(self):
        allie = self.__dict__.get("allie")
        if not allie:
            return torch.bfloat16
        return torch.int8 if allie.model.scales else allie.model.dtype

