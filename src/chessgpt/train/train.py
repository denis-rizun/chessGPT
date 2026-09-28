import time
from pathlib import Path

import structlog
import torch
from torch import nn

from chessgpt.data.dataset import build_dataloader
from chessgpt.logger import configure_logging
from chessgpt.model.config import ModelConfig
from chessgpt.model.transformer import Transformer
from chessgpt.train.config import TrainConfig
from chessgpt.train.evaluate import estimate_validation_loss
from chessgpt.train.optimizer import build_optimizer, learning_rate_at

logger = structlog.get_logger(__name__)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(model_config: ModelConfig, config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    device = select_device()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_dataloader(
        config.data_dir, "train", config.batch_size,
        shuffle=True, num_workers=config.num_workers,
    )
    validation_loader = build_dataloader(
        config.data_dir, "validation", config.batch_size,
        shuffle=False, num_workers=0,
    )

    model = Transformer(model_config).to(device)
    optimizer = build_optimizer(model, config)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.epochs
    logger.info(
        "training_started",
        device=str(device),
        parameters=model.parameter_count(),
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
    )

    best_validation_loss = float("inf")
    step = 0
    started = time.time()

    for epoch in range(config.epochs):
        for inputs, targets in train_loader:
            learning_rate = learning_rate_at(step, total_steps, config)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            _, loss = model(inputs, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            if step % config.log_every == 0:
                logger.info(
                    "train_step",
                    epoch=epoch,
                    step=step,
                    total_steps=total_steps,
                    loss=round(loss.item(), 4),
                    learning_rate=learning_rate,
                    elapsed_s=round(time.time() - started),
                )

            if step > 0 and step % config.eval_every == 0:
                validation_loss = estimate_validation_loss(model, validation_loader, config.eval_batches, device)
                logger.info("validation", step=step, validation_loss=round(validation_loss, 4))

                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    save_checkpoint(model, model_config, step, validation_loss, config.output_dir)
                    logger.info("checkpoint_saved", step=step, path=str(config.output_dir / "best.pt"))

            step += 1

    logger.info(
        "training_finished",
        elapsed_min=round((time.time() - started) / 60, 1),
        best_validation_loss=round(best_validation_loss, 4),
    )


def save_checkpoint(
    model: nn.Module,
    model_config: ModelConfig,
    step: int,
    validation_loss: float,
    output_dir: Path,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config.__dict__,
            "step": step,
            "validation_loss": validation_loss,
        },
        output_dir / "best.pt",
    )


if __name__ == "__main__":
    model_config = ModelConfig(
        vocabulary_size=1981,
        context_length=256,
        layer_count=4,
        head_count=4,
        model_dim=256,
        feedforward_dim=1024,
        dropout=0.1,
    )
    train_config = TrainConfig(
        data_dir=Path("/Users/d.rizun/Documents/work/pets/chessGPT/src/chessgpt/data/processed/2013-01"),
        output_dir=Path("/Users/d.rizun/Documents/work/pets/chessGPT/src/chessgpt/data/checkpoints/debug-3m"),
        batch_size=32,
        epochs=5,
    )
    configure_logging(log_file=train_config.output_dir / "train.log")
    train(model_config, train_config)
