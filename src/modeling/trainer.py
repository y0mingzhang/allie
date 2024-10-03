"""
Implementation based on https://github.com/karpathy/nanoGPT.
"""

import inspect
import logging
import math
import os
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from os.path import join
from typing import Iterable, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

import wandb
from modeling.data import UCITokenizer, json_to_game
from modeling.utils import (
    Batch,
    count_params,
    partition_model_parameters,
    seed_everything,
    setup_logging,
    to_device,
)

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        # training setup
        config: DictConfig,
        output_dir: str = "output/debug",
        seed: int = 42,
        backend: str = "nccl",
        device: str = "cuda",
        dtype: str = "bfloat16",
        torch_compile: bool = True,
        wandb_run_name: Optional[str] = None,
        wandb_run_id: Optional[str] = None,
        # hyperparameters
        gradient_accumulation_steps: int = 1,
        batch_size: int = 16,
        lr: float = 6e-4,
        min_lr: float = 6e-5,
        decay_lr: bool = True,
        weight_decay: float = 1e-1,
        beta1: float = 0.9,
        beta2: float = 0.95,
        grad_clip: float = 1.0,
        warmup_iters: int = 2000,
        decay_iters: int = 200000,
        max_iters: int = 600000,
        eval_interval: int = 2000,
        log_interval: int = 10,
        save_interval: int = 500,
    ):
        # setup
        self.output_dir = output_dir
        self.backend = backend
        self.dtype = dtype
        self.torch_compile = torch_compile
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.batch_size = batch_size
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            init_process_group(backend=backend)
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
            seed += self.ddp_rank
        else:
            self.device = device
            self.master_process = True

        if self.master_process:
            try:
                wandb.init(
                    project="chess-lm",
                    name=wandb_run_name,
                    id=wandb_run_id,
                    resume="allow",
                    config=OmegaConf.to_container(config),
                )
            except:
                logger.warning("wandb is offline")
                wandb.init(
                    project="chess-lm",
                    name=wandb_run_name,
                    id=wandb_run_id,
                    resume="allow",
                    config=OmegaConf.to_container(config),
                    mode="offline",
                )
            os.makedirs(output_dir, exist_ok=True)
            setup_logging(output_dir, file=True)

        seed_everything(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
        self.device_type = "cuda" if "cuda" in device else "cpu"
        self.ctx = (
            nullcontext()
            if device == "cpu"
            else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

        self.lr = lr
        self.min_lr = min_lr
        self.decay_lr = decay_lr
        self.weight_decay = weight_decay
        self.betas = (beta1, beta2)
        self.grad_clip = grad_clip
        self.warmup_iters = warmup_iters
        self.decay_iters = decay_iters
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.save_interval = save_interval

        self.iter_num = 0
        self.best_val_loss = float("inf")

    def save_config(self, config: DictConfig) -> None:
        logger.info("saving experiment configuration")
        if self.master_process:
            OmegaConf.save(config, join(self.output_dir, "config.yaml"))

    def setup_dataloaders(
        self,
        tokenizer: UCITokenizer,
        train_dataset: np.memmap,
        val_dataset: Iterable[Batch],
        test_dataset: Iterable[Batch],
    ) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        def train_iter():
            data_len = len(train_dataset)
            seq_len = tokenizer.max_length
            while True:
                ix = torch.randint(data_len - seq_len, (self.batch_size,))
                data = np.stack([train_dataset[i : i + seq_len] for i in ix])

                batch = tokenizer.pad_and_collate(data)
                yield to_device(batch, self.device)

        def collate_evaluate(data: list[list[dict]]):
            games = [json_to_game(d) for d in data]
            batch = tokenizer.pad_and_collate(games)
            return batch

        self.train_dataiter = train_iter()
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            collate_fn=collate_evaluate,
            pin_memory=True,
            prefetch_factor=10,
        )

        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            collate_fn=collate_evaluate,
            pin_memory=True,
            prefetch_factor=10,
        )

    def setup_model(self, model: PreTrainedModel) -> None:
        self.model = model.to(self.device)
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay_params, no_decay_params = partition_model_parameters(self.model)
        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": decay_params,
                "weight_decay": self.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]
        # PyTorch nightly has a new 'fused' option for AdamW that is much faster
        assert (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        ), "use torch 2.x for fused AdamW"

        logger.info(f"model parameters: {count_params(model)/1e9:.2f}B")

        use_fused = self.device_type == "cuda"
        if use_fused:
            logger.info("using fused AdamW optimizer")
        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.lr,
            betas=self.betas,
            fused=self.device_type == "cuda",
        )
        logger.info("optimizer initialized")

        if self.torch_compile:
            self.model_fn = torch.compile(self.model)
            logger.info("model compiled")
        else:
            self.model_fn = self.model

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

        self.load_ckpt("last")

    def train(self) -> None:
        self.model.train()
        batch = next(self.train_dataiter)
        t_start = t0 = time.time()
        losses: deque[float] = deque(maxlen=100)

        while True:
            lr = self.get_lr()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            if (
                self.eval_interval > 0
                and self.iter_num % self.eval_interval == 0
                and self.master_process
            ):
                val_log = self.validate()
                wandb.log(
                    {
                        "iter": self.iter_num,
                        "lr": lr,
                        "train/loss": sum(losses) / len(losses) if losses else None,
                    }
                    | val_log
                )

            if (
                self.save_interval > 0
                and self.iter_num % self.save_interval == 0
                and self.master_process
            ):
                self.save_ckpt("last")

            for micro_step in range(self.gradient_accumulation_steps):
                if self.ddp:
                    self.model.require_backward_grad_sync = (
                        micro_step == self.gradient_accumulation_steps - 1
                    )
                with self.ctx:
                    outputs = self.model_fn(**batch)

                batch = next(self.train_dataiter)
                loss = outputs["loss"]
                self.scaler.scale(loss).backward()

            if self.grad_clip > 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

            if self.log_interval >= 0 and self.iter_num % self.log_interval == 0:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                loss_mean = sum(losses) / len(losses)
                logger.info(
                    f"train - iter {self.iter_num}: loss {loss_mean:.4f}, time {dt:.2f}s"
                )
            self.iter_num += 1
            if self.iter_num > self.max_iters:
                break

        t_end = time.time()
        logger.info(f"training complete! total time {t_end - t_start:.2f}s")

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        assert self.master_process
        self.model.eval()
        t0 = time.time()
        losses = defaultdict(list)
        keys = ["lm_loss", "value_loss", "time_loss", "loss"]
        for batch in dataloader:
            batch = to_device(batch, self.device)
            with self.ctx:
                outputs = self.model_fn(**batch)
                assert "loss" in outputs
                for k in keys:
                    if k in outputs:
                        losses[k].append(outputs[k].item())

        t1 = time.time()
        self.model.train()

        return {k: sum(losses[k]) / len(losses[k]) for k in keys if k in losses} | {
            "time": t1 - t0
        }

    def generate_log_string(self, eval_data: dict[str, float]):
        loss_keys = ["lm_loss", "value_loss", "time_loss", "loss"]
        dt = eval_data["time"]
        s = ", ".join(
            [f"{k} {eval_data[k]:.4f}" for k in loss_keys if k in eval_data]
            + [f"time {dt:.2f}s"]
        )
        return s

    def validate(self) -> dict[str, float]:
        eval_outputs = self.evaluate(self.val_dataloader)
        loss = eval_outputs["loss"]
        log_string = self.generate_log_string(eval_outputs)
        logger.info(f"val - iter {self.iter_num}: {log_string}")

        if loss < self.best_val_loss:
            logger.info(f"new best val loss {loss:.4f}")
            self.best_val_loss = loss
            self.save_ckpt("best")

        return {f"val/{k}": v for k, v in eval_outputs.items() if k.endswith("loss")}

    def test(self) -> None:
        self.load_ckpt("best")
        if self.master_process:
            eval_outputs = self.evaluate(self.test_dataloader)
            loss = eval_outputs["loss"]
            dt = eval_outputs["time"]
            logger.info(f"test: loss {loss:.4f}, time {dt:.2f}s")
            wandb.log({"iter": self.iter_num, "test/loss": loss})

    def get_lr(self) -> float:
        it = self.iter_num
        if not self.decay_lr:
            return self.lr
        assert self.lr >= self.min_lr
        lr_decay_iters = self.decay_iters

        if it < self.warmup_iters:
            return self.lr * it / self.warmup_iters
        if it > lr_decay_iters:
            return self.min_lr

        decay_ratio = (it - self.warmup_iters) / (lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.lr - self.min_lr)

    def cleanup(self) -> None:
        if self.ddp:
            destroy_process_group()
        logger.info("all done! exiting gracefully...")

    def load_ckpt(self, ckpt_type: str) -> None:
        ckpt_path = join(self.output_dir, f"{ckpt_type}.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.iter_num = ckpt["iter_num"]
            self.best_val_loss = ckpt["best_val_loss"]
            model = self.model.module if self.ddp else self.model
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            logger.info(
                f"loading {ckpt_type} checkpoint from iter {self.iter_num}: best_val_loss {self.best_val_loss}"
            )
        else:
            logger.info(f"no {ckpt_type} checkpoint found, start training from scratch")

    def save_ckpt(self, ckpt_type: str) -> None:
        ckpt_path = join(self.output_dir, f"{ckpt_type}.pt")
        model = self.model.module if self.ddp else self.model
        torch.save(
            {
                "iter_num": self.iter_num,
                "best_val_loss": self.best_val_loss,
                "model": model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            ckpt_path,
        )
        logger.info(f"saved checkpoint to {ckpt_path}")
