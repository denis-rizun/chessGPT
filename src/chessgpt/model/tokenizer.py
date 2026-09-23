from dataclasses import dataclass

import chess

ELO_BUCKETS = [1600, 1800, 2000, 2200, 2400]
BOARD_SIZE = 8
PROMOTION_PIECES = "qrbn"
PROMOTION_RANKS = ((6, 7), (1, 0))


@dataclass
class Vocabulary:
    index_to_token: list[str]
    token_to_index: dict[str, int]

    @classmethod
    def build_vocabulary(cls) -> Vocabulary:
        move_tokens = cls.calculate_by_square() + cls.calculate_by_file()
        special_tokens = cls.calculate_special_tokens()

        index_to_token = special_tokens + move_tokens
        token_to_index = {token: index for index, token in enumerate(index_to_token)}
        return cls(index_to_token=index_to_token, token_to_index=token_to_index)

    @classmethod
    def calculate_by_square(cls) -> list[str]:
        moves = []
        for from_square in chess.SQUARES:
            for to_square in chess.SQUARES:
                if from_square == to_square:
                    continue

                if cls.is_reachable_on_empty_board(from_square, to_square):
                    move_name = chess.square_name(from_square) + chess.square_name(to_square)
                    moves.append(move_name)
        return moves

    @classmethod
    def calculate_by_file(cls) -> list[str]:
        moves = []
        for from_file in range(BOARD_SIZE):
            for to_file in (from_file - 1, from_file, from_file + 1):
                if not 0 <= to_file < BOARD_SIZE:
                    continue

                for from_rank, to_rank in PROMOTION_RANKS:
                    from_square = chess.square(from_file, from_rank)
                    to_square = chess.square(to_file, to_rank)
                    for promotion_piece in PROMOTION_PIECES:
                        move_name = chess.square_name(from_square) + chess.square_name(to_square) + promotion_piece
                        moves.append(move_name)
        return moves

    @classmethod
    def calculate_special_tokens(cls) -> list[str]:
        special_tokens = ["<pad>", "<bos>", "<eos>"]
        for color in ("w", "b"):
            for bucket_lower_bound in ELO_BUCKETS:
                special_tokens.append(f"<{color}_elo_{bucket_lower_bound}>")

        return special_tokens

    @classmethod
    def is_reachable_on_empty_board(cls, from_square: chess.Square, to_square: chess.Square) -> bool:
        rank_delta = chess.square_rank(to_square) - chess.square_rank(from_square)
        file_delta = chess.square_file(to_square) - chess.square_file(from_square)

        is_queen_move = rank_delta == 0 or file_delta == 0 or abs(rank_delta) == abs(file_delta)
        if is_queen_move:
            return True

        return {abs(rank_delta), abs(file_delta)} == {1, 2}
