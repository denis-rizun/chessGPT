from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chessgpt.data.dataset import ChessDataset

PAD_INDEX = 0
IGNORE_INDEX = -100


def collate_games(games: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(game) for game in games)

    inputs = torch.full((len(games), max_length - 1), PAD_INDEX, dtype=torch.long)
    targets = torch.full((len(games), max_length - 1), IGNORE_INDEX, dtype=torch.long)

    for row, game in enumerate(games):
        length = len(game) - 1
        inputs[row, :length] = game[:-1]
        targets[row, :length] = game[1:]

    return inputs, targets



def build_dataloader(
    data_dir: Path,
    split: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    return DataLoader(
        dataset=ChessDataset(data_dir, split=split),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_games,
        num_workers=num_workers,
        drop_last=True,
    )

