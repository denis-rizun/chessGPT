import torch
from torch import nn


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_length: int, base: float = 10_000.0) -> None:
        super().__init__()

        pair_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
        frequencies = 1.0 / (base ** (pair_indices / head_dim))

        positions = torch.arange(max_length, dtype=torch.float32)
        angles = positions[:, None] * frequencies[None, :]

        self.register_buffer("cosines", angles.cos(), persistent=False)
        self.register_buffer("sines", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-2]

        cosines = self.cosines[:length]
        sines = self.sines[:length]

        even = x[..., 0::2]
        odd = x[..., 1::2]

        rotated_even = even * cosines - odd * sines
        rotated_odd = even * sines + odd * cosines

        result = torch.empty_like(x)
        result[..., 0::2] = rotated_even
        result[..., 1::2] = rotated_odd
        return result
