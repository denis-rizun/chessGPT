import torch
from torch import nn


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative = probabilities.cumsum(dim=-1)

    remove_sorted = cumulative - probabilities > top_p

    remove = torch.zeros_like(remove_sorted)
    remove.scatter_(dim=-1, index=sorted_indices, src=remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


@torch.no_grad()
def generate(
    model: nn.Module,
    prefix: torch.Tensor,
    max_new_tokens: int,
    end_token: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> torch.Tensor:
    model.eval()
    tokens = prefix
    context_length = model.config.context_length

    for _ in range(max_new_tokens):
        window = tokens[:, -context_length:]
        logits, _ = model(window)
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)
        next_logits = apply_top_p(next_logits, top_p)

        probabilities = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

        if (next_token == end_token).all():
            break

    return tokens