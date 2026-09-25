import argparse
from datetime import UTC, datetime
from pathlib import Path

from chessgpt.evals.legality import GenerationSettings, evaluate_legality
from chessgpt.evals.report import build_report_payload, print_report, save_report
from chessgpt.model.tokenizer import Vocabulary
from chessgpt.train.checkpoint import load_checkpoint
from chessgpt.train.train import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure move legality")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data_dir", type=Path, help="dataset directory the model was trained on")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.7, 1.0])
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    device = select_device()
    model, model_config, checkpoint = load_checkpoint(arguments.checkpoint, device)
    vocabulary = Vocabulary.build_vocabulary()
    reports_dir = arguments.checkpoint.parent / "reports"
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")

    for temperature in arguments.temperatures:
        settings = GenerationSettings(
            games=arguments.games,
            temperature=temperature,
            seed=arguments.seed,
        )

        print(f"\n=== temperature {temperature} ===")
        report = evaluate_legality(model, vocabulary, device, settings)
        print_report(report)

        payload = build_report_payload(
            report=report,
            settings=settings,
            model_config=model_config,
            parameter_count=model.parameter_count(),
            checkpoint_path=arguments.checkpoint,
            checkpoint=checkpoint,
            data_dir=arguments.data_dir,
            device=str(device),
        )
        save_report(payload, reports_dir / f"{stamp}_legality_t{temperature}.json")


if __name__ == "__main__":
    main()