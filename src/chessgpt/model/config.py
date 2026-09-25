from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocabulary_size: int
    context_length: int = 256
    layer_count: int = 8
    head_count: int = 8
    model_dim: int = 512
    feedforward_dim: int = 2048
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.model_dim // self.head_count
