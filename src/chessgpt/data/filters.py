from dataclasses import dataclass

MIN_ELO = 1600
MIN_PLIES = 10
MAX_PLIES = 150
MIN_DURATION_SECONDS = 180
ALLOWED_TERMINATIONS = frozenset({"Normal", "Time forfeit"})
ALLOWED_VARIANTS = frozenset({"Standard"})


@dataclass
class GameMetadata:
    white_elo: int
    black_elo: int

    @classmethod
    def parse(cls, headers: dict[str, str]) -> GameMetadata | None:
        if headers.get("Termination", "Normal") not in ALLOWED_TERMINATIONS:
            return None

        if headers.get("Variant", "Standard") not in ALLOWED_VARIANTS:
            return None

        duration = cls.estimate_duration_seconds(headers.get("TimeControl"))
        if not duration or duration < MIN_DURATION_SECONDS:
            return None

        white_elo = headers.get("WhiteElo", "")
        black_elo = headers.get("BlackElo", "")
        if not white_elo.isdigit() or not black_elo.isdigit():
            return None

        white_elo, black_elo = int(white_elo), int(black_elo)
        if white_elo < MIN_ELO and black_elo < MIN_ELO:
            return None

        return GameMetadata(white_elo=white_elo, black_elo=black_elo)

    @staticmethod
    def estimate_duration_seconds(time_control: str) -> int | None:
        if "+" not in time_control:
            return None

        base, _, increment = time_control.partition("+")
        if not base.isdigit() or not increment.isdigit():
            return None

        return int(base) + 40 * int(increment)
