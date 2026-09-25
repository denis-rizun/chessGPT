import math

import torch
from torch import nn

from chessgpt.model.rope import RotaryPositionEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int, head_count: int, context_length: int) -> None:
        super().__init__()

        self.head_count = head_count
        self.head_dim = model_dim // head_count
        self.rope = RotaryPositionEmbedding(self.head_dim, context_length)
        self.query_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.key_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.value_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.output_projection = nn.Linear(model_dim, model_dim, bias=False)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = x.shape
        x = x.view(batch_size, length, self.head_count, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length, model_dim = x.shape

        query = self.split_heads(self.query_projection(x))
        key = self.split_heads(self.key_projection(x))
        value = self.split_heads(self.value_projection(x))

        query = self.rope(query)
        key = self.rope(key)

        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        causal_mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        attended = weights @ value

        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, length, model_dim)
        return self.output_projection(attended)