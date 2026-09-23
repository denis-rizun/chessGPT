from pathlib import Path

import numpy as np

TOKEN_DTYPE = np.uint16
OFFSET_DTYPE = np.uint64


class Writer:
    def __init__(self, tokens_path: Path) -> None:
        self.tokens_path = tokens_path
        self.file = tokens_path.open("wb")
        self.offset = []
        self.position = 0

    def add(self, tokens: list[int]) -> None:
        self.offset.append(self.position)
        array = np.asarray(tokens, dtype=TOKEN_DTYPE)
        self.file.write(array.tobytes())
        self.position += len(array)

    @property
    def game_count(self) -> int:
        return len(self.offset)

    def close(self, offsets_path: Path) -> None:
        self.file.close()
        boundaries = self.offset + [self.position]
        np.save(offsets_path, np.asarray(boundaries, dtype=OFFSET_DTYPE))
