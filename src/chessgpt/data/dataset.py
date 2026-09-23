import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PAD_INDEX = 0
IGNORE_INDEX = -100


class ChessDataset(Dataset):
    def __init__(self, data_dir: Path, split: str = "train") -> None:
        self.tokens_path = data_dir / f"{split}.bin"
        self.boundaries = np.load(data_dir / f"{split}_boundaries.npy")
        self.index_to_token = json.loads((data_dir / "vocabulary.json").read_text())
        self._tokens: np.memmap | None = None

    @property
    def tokens(self) -> np.memmap:
        if not self._tokens:
            self._tokens = np.memmap(self.tokens_path, dtype=np.uint16, mode="r")
        return self._tokens

    def __len__(self) -> int:
        return len(self.boundaries) - 1

    def __getitem__(self, index: int) -> torch.Tensor:
        start = int(self.boundaries[index])
        end = int(self.boundaries[index + 1])
        game = np.asarray(self.tokens[start:end], dtype=np.int64)
        return torch.from_numpy(game)


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

