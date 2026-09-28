from pathlib import Path

import structlog
import torch

from chessgpt.model.config import ModelConfig
from chessgpt.model.transformer import Transformer

logger = structlog.get_logger(__name__)


def load_checkpoint(path: Path, device: torch.device) -> tuple[Transformer, ModelConfig, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = Transformer(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    logger.info(
        "checkpoint_loaded",
        path=str(path),
        step=checkpoint["step"],
        validation_loss=round(checkpoint["validation_loss"], 4),
    )
    return model, config, checkpoint
