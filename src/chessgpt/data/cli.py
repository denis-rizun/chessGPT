import argparse
from pathlib import Path

from chessgpt.data.prepare import prepare_dataset
from chessgpt.data.verify import verify_dataset
from chessgpt.logger import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset dump")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", type=bool, default=True)
    parser.add_argument("--max-games", type=int, default=50_000)
    parser.add_argument("--validation-games", type=int, default=2_000)

    args = parser.parse_args()
    configure_logging()
    prepare_dataset(
        source=args.source,
        output_dir=args.output,
        max_games=args.max_games,
        validation_games=args.validation_games,
    )
    if args.verify:
        print("------")
        verify_dataset(output_dir=args.output)


if __name__ == "__main__":
    main()