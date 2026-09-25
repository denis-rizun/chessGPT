from pathlib import Path

import torch

from chessgpt.model.config import ModelConfig
from chessgpt.model.transformer import Transformer


def load_checkpoint(path: Path, device: torch.device) -> tuple[Transformer, ModelConfig, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = Transformer(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    print(f"checkpoint: step {checkpoint['step']}, val loss {checkpoint['validation_loss']:.4f}")
    return model, config, checkpoint