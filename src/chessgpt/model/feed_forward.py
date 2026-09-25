import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, feedforward_dim: int) -> None:
        super().__init__()

        self.expand = nn.Linear(model_dim, feedforward_dim)
        self.contract = nn.Linear(feedforward_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = nn.functional.gelu(x)
        return self.contract(x)
