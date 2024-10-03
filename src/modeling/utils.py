import logging
import os
import random
import sys
from collections.abc import Iterator

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm

Batch = dict[str, torch.Tensor]

logger = logging.getLogger(__name__)


def setup_logging(output_dir: str, file: bool = False) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "%m-%d %H:%M",
    )
    stdouth = logging.StreamHandler(sys.stdout)
    stdouth.setLevel(logging.INFO)
    stdouth.setFormatter(formatter)
    logger.addHandler(stdouth)
    if file:
        log_path = os.path.join(output_dir, "log.txt")
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        log_path = os.path.join(output_dir, "debug.txt")
        dfh = logging.FileHandler(log_path)
        dfh.setLevel(logging.DEBUG)
        dfh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(dfh)


def seed_everything(seed: int) -> None:
    torch.backends.cudnn.benchmark = True
    logger.info(
        f"not setting manual seed to {seed} due to dataloader behavior after requeue"
    )
    return

    logger.info(f"setting manual seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def partition_model_parameters(
    model: PreTrainedModel,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, transformers.Conv1D)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, LlamaRMSNorm)
    param_dict = {pn: p for pn, p in model.named_parameters()}

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn
            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif fpn.endswith("lm_head.weight"):
                if fpn in param_dict:
                    # no weight tying
                    no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay set!" % (
        str(param_dict.keys() - union_params),
    )

    return [param_dict[pn] for pn in sorted(list(decay))], [
        param_dict[pn] for pn in sorted(list(no_decay))
    ]


def cycle_dataloader(dataloader: DataLoader, device: str) -> Iterator[Batch]:
    epoch = 0
    while True:
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        for batch in dataloader:
            yield to_device(batch, device)
        epoch += 1


def to_device(batch: Batch, device: str, blocking: bool = False) -> Batch:
    assert isinstance(batch, dict)
    if device.startswith("cuda"):
        return {
            k: v.pin_memory().to(device, non_blocking=not blocking)
            for k, v in batch.items()
        }
    return batch


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
