import torch
from torch import nn

from chessgpt.model.block import TransformerBlock
from chessgpt.model.config import ModelConfig
from chessgpt.model.normalization import RMSNorm

IGNORE_INDEX = -100


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config
        self.embedding = nn.Embedding(config.vocabulary_size, config.model_dim)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.layer_count)])
        self.final_norm = RMSNorm(config.model_dim)
        self.output_head = nn.Linear(config.model_dim, config.vocabulary_size, bias=False)
        self.output_head.weight = self.embedding.weight
        self.apply(self._init_weights)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.output_head(x)

        if targets is None:
            return logits, None

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=IGNORE_INDEX,
        )
        return logits, loss

    def parameter_count(self, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (p for p in parameters if p.requires_grad)
        return sum(p.numel() for p in parameters)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
