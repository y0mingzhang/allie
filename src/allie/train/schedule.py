"""Absolute-step warmup-stable-decay schedule (model.nanogpt.TrainingManager): learning rate,
Muon momentum and multi-token prediction weights; no dependency on the total training horizon."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Schedule:
    warmup_steps: int = 32
    mtp_steps: int = 64
    split_step: int = 65
    batch_rows: int = 512
    plateau: float = 2.0
    final_lr: float = 0.1
    decay_shape: str = "linear"
    momentum_warmup: bool = True  # Muon momentum 0.85 -> 0.95 over warmup; else 0.95

    def validate(self):
        assert self.decay_shape in ("linear", "cosine")
        assert self.warmup_steps > 0 and self.mtp_steps >= 0  # 0: no multi-token loss
        assert self.split_step >= self.mtp_steps and self.split_step % 2 == 1
        assert self.batch_rows > 0 and 0 < self.final_lr < self.plateau

    def lr(self, step, decay_start=-1, end_step=None):
        if decay_start >= 0 and step >= decay_start:
            assert end_step > decay_start
            # First decay update starts decreasing; last update reaches floor.
            fraction = (step - decay_start + 1) / (end_step - decay_start)
            if self.decay_shape == "cosine":
                return self.final_lr + (self.plateau - self.final_lr) * 0.5 * (
                    1 + math.cos(math.pi * min(1.0, fraction))
                )
            return self.plateau + (self.final_lr - self.plateau) * min(1.0, fraction)
        return self.plateau * min(1.0, (step + 1) / self.warmup_steps)

    def momentum(self, step):
        if not self.momentum_warmup:
            return 0.95
        return 0.85 + 0.10 * min(1.0, step / self.warmup_steps)

    def mtp(self, step):
        if self.mtp_steps == 0:
            return [1.0]
        x = step / self.mtp_steps
        if x < 0.5:
            return [1.0, 0.5, 0.25 * (1 - 2 * x)]
        if x < 1:
            return [1.0, 0.5 * (2 - 2 * x)]
        return [1.0]
