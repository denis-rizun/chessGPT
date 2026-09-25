import torch
from torch import nn

from chessgpt.model.attention import MultiHeadAttention
from chessgpt.model.config import ModelConfig
from chessgpt.model.feed_forward import FeedForward
from chessgpt.model.normalization import RMSNorm


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.attention_norm = RMSNorm(config.model_dim)
        self.attention = MultiHeadAttention(config.model_dim, config.head_count, config.context_length)
        self.feedforward_norm = RMSNorm(config.model_dim)
        self.feedforward = FeedForward(config.model_dim, config.feedforward_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feedforward(self.feedforward_norm(x))
        return x
