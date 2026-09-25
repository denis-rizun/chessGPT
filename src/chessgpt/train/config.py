from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path
    output_dir: Path

    batch_size: int = 32
    epochs: int = 5

    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_ratio: float = 0.02
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    eval_every: int = 200
    eval_batches: int = 20
    log_every: int = 50
    num_workers: int = 2
    seed: int = 1337
