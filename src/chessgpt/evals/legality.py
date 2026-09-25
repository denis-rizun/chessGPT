from dataclasses import dataclass, field

import chess
import torch
from torch import nn

from chessgpt.model.sampling import generate
from chessgpt.model.tokenizer import Vocabulary

SPECIAL_PREFIX_LENGTH = 3


@dataclass(frozen=True)
class GenerationSettings:
    games: int = 500
    batch_size: int = 50
    max_new_tokens: int = 160
    temperature: float = 1.0
    top_p: float = 0.95
    elo_bucket: int = 1800
    seed: int = 0


@dataclass(frozen=True)
class GameOutcome:
    legal_moves: int
    failed: bool
    failure_token: str | None
    reached_end: bool


@dataclass
class LegalityReport:
    games: int = 0
    total_legal_moves: int = 0
    failures: list[str] = field(default_factory=list)
    lengths_before_failure: list[int] = field(default_factory=list)
    clean_games: int = 0
    finished_games: int = 0

    @property
    def legal_move_rate(self) -> float:
        total_attempted = self.total_legal_moves + len(self.failures)
        return self.total_legal_moves / max(1, total_attempted)

    @property
    def clean_game_rate(self) -> float:
        return self.clean_games / max(1, self.games)

    @property
    def mean_length_before_failure(self) -> float:
        if not self.lengths_before_failure:
            return 0.0
        return sum(self.lengths_before_failure) / len(self.lengths_before_failure)

    def add(self, outcome: GameOutcome) -> None:
        self.games += 1
        self.total_legal_moves += outcome.legal_moves

        if outcome.failed:
            self.failures.append(outcome.failure_token)
            self.lengths_before_failure.append(outcome.legal_moves)
            return

        self.clean_games += 1
        if outcome.reached_end:
            self.finished_games += 1


def replay_game(tokens: list[int], vocabulary: Vocabulary) -> GameOutcome:
    """Replays a generated game on the board up to the first illegal move."""
    board = chess.Board()
    legal_moves = 0

    for index in tokens[SPECIAL_PREFIX_LENGTH:]:
        token = vocabulary.index_to_token[index]

        if token == "<eos>":
            return GameOutcome(legal_moves, failed=False, failure_token=None, reached_end=True)
        if token.startswith("<"):
            return GameOutcome(legal_moves, failed=True, failure_token=token, reached_end=False)

        move = chess.Move.from_uci(token)
        if move not in board.legal_moves:
            return GameOutcome(legal_moves, failed=True, failure_token=token, reached_end=False)

        board.push(move)
        legal_moves += 1

    return GameOutcome(legal_moves, failed=False, failure_token=None, reached_end=False)


def build_prefix(vocabulary: Vocabulary, elo_bucket: int) -> list[int]:
    return [
        vocabulary.token_to_index["<bos>"],
        vocabulary.token_to_index[f"<w_elo_{elo_bucket}>"],
        vocabulary.token_to_index[f"<b_elo_{elo_bucket}>"],
    ]


@torch.no_grad()
def evaluate_legality(
    model: nn.Module,
    vocabulary: Vocabulary,
    device: torch.device,
    settings: GenerationSettings,
) -> LegalityReport:
    torch.manual_seed(settings.seed)

    prefix_tokens = build_prefix(vocabulary, settings.elo_bucket)
    end_token = vocabulary.token_to_index["<eos>"]
    report = LegalityReport()

    for batch_start in range(0, settings.games, settings.batch_size):
        current_batch = min(settings.batch_size, settings.games - batch_start)
        prefix = torch.tensor([prefix_tokens] * current_batch, device=device)

        generated = generate(
            model,
            prefix,
            max_new_tokens=settings.max_new_tokens,
            end_token=end_token,
            temperature=settings.temperature,
            top_p=settings.top_p,
        )

        for row in generated.tolist():
            report.add(replay_game(row, vocabulary))

    return report
