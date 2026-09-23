import json
from pathlib import Path

import chess
import numpy as np


def verify_dataset(output_dir: Path, sample_size: int = 300, seed: int = 0) -> None:
    index_to_token = json.loads((output_dir / "vocabulary.json").read_text())
    tokens = np.memmap(output_dir / "train.bin", dtype=np.uint16, mode="r")
    boundaries = np.load(output_dir / "train_offsets.npy")

    game_count = len(boundaries) - 1
    print(f"game count: {game_count}, tokens: {len(tokens)}")
    rng = np.random.default_rng(seed)
    checked_moves = 0

    for game_index in rng.choice(game_count, size=min(sample_size, game_count), replace=False):
        start, end = int(boundaries[game_index]), int(boundaries[game_index + 1])
        game_tokens = tokens[start:end]

        assert index_to_token[game_tokens[0]] == "<bos>", f"game {game_index} without <bos>"
        assert index_to_token[game_tokens[-1]] == "<eos>", f"game {game_index} without <eos>"

        board = chess.Board()
        for token in game_tokens[3:-1]:
            move = chess.Move.from_uci(index_to_token[token])
            assert move in board.legal_moves, (
                f"game {game_index}: illegal {index_to_token[token]} in position {board.fen()}"
            )
            board.push(move)
            checked_moves += 1

    print(f"verified {sample_size} games, {checked_moves} moves - all legal")