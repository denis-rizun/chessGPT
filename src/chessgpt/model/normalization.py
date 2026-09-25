import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, model_dim: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(model_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(mean_square + self.epsilon)
        return normalized * self.scale
