import re

import chess
import structlog

from chessgpt.data.filters import GameMetadata, MIN_PLIES, MAX_PLIES
from chessgpt.model.tokenizer import ELO_BUCKETS, Vocabulary

COMMENT_PATTERN = re.compile(r"\{[^}]*\}")
RESULT_TOKENS = frozenset({"1-0", "0-1", "1/2-1/2", "*"})
ANNOTATION_CHARACTERS = "?!"

logger = structlog.get_logger(__name__)


def extract_san_moves(move_text: str) -> list[str]:
    move_text = COMMENT_PATTERN.sub(" ", move_text)

    san_moves = []
    for token in move_text.split():
        if token in RESULT_TOKENS or token.startswith("$"):
            continue

        if "." in token:
            token = token.rpartition(".")[2]

        token = token.rstrip(ANNOTATION_CHARACTERS)
        if not token:
            continue


        san_moves.append(token)
    return san_moves


def map_elo_to_bucket(elo: int) -> int:
    bucket = ELO_BUCKETS[0]
    for lower_bound in ELO_BUCKETS:
        if elo >= lower_bound:
            bucket = lower_bound
    return bucket


def encode_game(game_metadata: GameMetadata, move_text: str, vocabulary: Vocabulary) -> list[int] | None:
    san_moves = extract_san_moves(move_text)
    if not MIN_PLIES <= len(san_moves) <= MAX_PLIES:
        return None

    token_to_index = vocabulary.token_to_index
    tokens = [
        token_to_index["<bos>"],
        token_to_index[f"<w_elo_{map_elo_to_bucket(game_metadata.white_elo)}>"],
        token_to_index[f"<b_elo_{map_elo_to_bucket(game_metadata.black_elo)}>"],
    ]
    board = chess.Board()
    for san in san_moves:
        try:
            move = board.parse_san(san)
        except (ValueError, AssertionError) as e:
            logger.warning("illegal_san", san=san, error=str(e))
            return None

        board.push(move)
        tokens.append(token_to_index[move.uci()])

    tokens.append(token_to_index["<eos>"])
    return tokens
