import math

import torch
from torch import nn

from chessgpt.train.config import TrainConfig


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.AdamW:
    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        decay_parameters.append(parameter) if parameter.dim() >= 2 else no_decay_parameters.append(parameter)

    return torch.optim.AdamW(
        params=[
            {"params": decay_parameters, "weight_decay": config.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=config.max_learning_rate,
        betas=(config.beta1, config.beta2),
    )


def learning_rate_at(step: int, total_steps: int, config: TrainConfig) -> float:
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))

    if step < warmup_steps:
        return config.max_learning_rate * (step + 1) / warmup_steps

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + cosine * (config.max_learning_rate - config.min_learning_rate)
