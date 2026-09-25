import json
import platform
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from chessgpt.evals.legality import GenerationSettings, LegalityReport
from chessgpt.model.config import ModelConfig


def print_report(report: LegalityReport, top_failures: int = 5) -> None:
    print(f"games:                       {report.games}")
    print(f"legal moves:                 {report.legal_move_rate:.2%}")
    print(f"clean games:                 {report.clean_game_rate:.2%}")
    print(f"finished to <eos>:           {report.finished_games}")
    print(f"mean length before failure:  {report.mean_length_before_failure:.1f} plies")

    if report.failures:
        print("\nmost common illegal moves:")
        for token, count in Counter(report.failures).most_common(top_failures):
            print(f"  {token:8} {count}")


def build_report_payload(
    report: LegalityReport,
    settings: GenerationSettings,
    model_config: ModelConfig,
    parameter_count: int,
    checkpoint_path: Path,
    checkpoint: dict,
    data_dir: Path,
    device: str,
) -> dict:
    dataset_meta = json.loads((data_dir / "meta.json").read_text())

    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "checkpoint": {
            "path": str(checkpoint_path),
            "step": checkpoint["step"],
            "validation_loss": checkpoint["validation_loss"],
        },
        "model": {**asdict(model_config), "parameter_count": parameter_count},
        "dataset": {
            "source": dataset_meta.get("source"),
            "train_games": dataset_meta.get("train_games"),
            "train_tokens": dataset_meta.get("train_tokens"),
        },
        "generation": asdict(settings),
        "metrics": {
            "legal_move_rate": round(report.legal_move_rate, 4),
            "clean_game_rate": round(report.clean_game_rate, 4),
            "mean_length_before_failure": round(report.mean_length_before_failure, 2),
            "finished_games": report.finished_games,
            "games": report.games,
            "total_legal_moves": report.total_legal_moves,
            "failure_count": len(report.failures),
        },
        "top_failures": [
            {"token": token, "count": count}
            for token, count in Counter(report.failures).most_common(10)
        ],
        "environment": {"device": device, "platform": platform.platform()},
    }


def save_report(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"report saved: {path}")
