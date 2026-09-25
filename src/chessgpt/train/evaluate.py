import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def estimate_validation_loss(
    model: nn.Module,
    loader: DataLoader,
    batches: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for index, (inputs, targets) in enumerate(loader):
        if index >= batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        _, loss = model(inputs, targets)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)
