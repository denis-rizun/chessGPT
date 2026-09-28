import json
from pathlib import Path

import structlog
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from chessgpt.data.encode import encode_game
from chessgpt.data.filters import GameMetadata, MIN_ELO, MIN_PLIES, MAX_PLIES
from chessgpt.data.stream import stream_blocks
from chessgpt.data.writer import Writer
from chessgpt.model.tokenizer import Vocabulary

VALIDATION_EVERY = 20

logger = structlog.get_logger(__name__)


def prepare_dataset(source: Path, output_dir: Path, max_games: int = 50_000, validation_games: int = 2_000) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    vocabulary = Vocabulary.build_vocabulary()
    (output_dir / "vocabulary.json").write_text(json.dumps(vocabulary.index_to_token))

    train = Writer(output_dir / "train.bin")
    validation = Writer(output_dir / "validation.bin")

    scanned = 0
    accepted = 0
    progress = tqdm(total=max_games, unit="game")
    with logging_redirect_tqdm():
        for headers, move_text in stream_blocks(source):
            scanned += 1
            metadata = GameMetadata.parse(headers)
            if not metadata:
                continue

            tokens = encode_game(metadata, move_text, vocabulary)
            if not tokens:
                continue

            goes_to_validation = validation.game_count < validation_games and accepted % VALIDATION_EVERY == 0
            if goes_to_validation:
                validation.add(tokens)
            else:
                train.add(tokens)
                progress.update(1)

            accepted += 1
            if train.game_count >= max_games:
                break

    progress.close()
    train_games, train_tokens = train.game_count, train.position
    validation_games_written, validation_tokens = validation.game_count, validation.position

    train.close(output_dir / "train_offsets.npy")
    validation.close(output_dir / "validation_offsets.npy")
    (output_dir / "meta.json").write_text(json.dumps({
        "vocabulary_size": len(vocabulary.index_to_token),
        "train_games": train_games,
        "train_tokens": train_tokens,
        "validation_games": validation_games_written,
        "validation_tokens": validation_tokens,
        "games_scanned": scanned,
        "source": source.name,
        "filters": {
            "min_elo": MIN_ELO,
            "plies": [MIN_PLIES, MAX_PLIES],
        },
    }, indent=2, ensure_ascii=False))

    logger.info(
        "dataset_prepared",
        scanned=scanned,
        accepted=accepted,
        acceptance_rate=round(accepted / scanned, 4),
        train_games=train_games,
        train_tokens=train_tokens,
        validation_games=validation_games_written,
        validation_tokens=validation_tokens,
    )
