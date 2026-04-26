from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import METRICS_DIR


def _rmse(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(values.to_numpy(dtype=float)))))


def _within_rate(values: pd.Series, threshold: float) -> float:
    return float(np.mean(values.to_numpy(dtype=float) <= threshold))


def summarize_predictions(predictions: pd.DataFrame, top_n: int = 20) -> dict[str, Any]:
    predictions = predictions.copy()
    predictions["abs_error"] = predictions["residual"].abs()

    by_true_day = (
        predictions.groupby("true_days_to_rotten")
        .agg(
            count=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("residual", _rmse),
            within_1_day_rate=("abs_error", lambda values: _within_rate(values, 1.0)),
            within_2_day_rate=("abs_error", lambda values: _within_rate(values, 2.0)),
            mean_prediction=("predicted_days_to_rotten", "mean"),
            mean_residual=("residual", "mean"),
        )
        .reset_index()
        .sort_values("true_days_to_rotten")
    )

    by_banana = (
        predictions.groupby("banana_id")
        .agg(
            count=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("residual", _rmse),
            within_1_day_rate=("abs_error", lambda values: _within_rate(values, 1.0)),
            within_2_day_rate=("abs_error", lambda values: _within_rate(values, 2.0)),
            mean_residual=("residual", "mean"),
            min_true_days=("true_days_to_rotten", "min"),
            max_true_days=("true_days_to_rotten", "max"),
        )
        .reset_index()
        .sort_values(["mae", "banana_id"], ascending=[False, True])
    )

    worst_predictions = predictions.sort_values(["abs_error", "banana_id"], ascending=[False, True]).head(top_n)
    overall_abs_error = predictions["abs_error"]
    overall_residual = predictions["residual"]
    worst_banana = by_banana.iloc[0].to_dict() if not by_banana.empty else {}

    return {
        "overall": {
            "count": int(len(predictions)),
            "mae": float(overall_abs_error.mean()),
            "rmse": _rmse(overall_residual),
            "within_1_day_accuracy": _within_rate(overall_abs_error, 1.0),
            "within_2_day_accuracy": _within_rate(overall_abs_error, 2.0),
            "max_abs_error": float(overall_abs_error.max()),
            "mean_residual": float(overall_residual.mean()),
        },
        "by_true_day": by_true_day,
        "by_banana": by_banana,
        "worst_predictions": worst_predictions[
            [
                "banana_id",
                "day_index",
                "true_days_to_rotten",
                "predicted_days_to_rotten",
                "residual",
                "abs_error",
                "image_path",
            ]
        ],
        "worst_banana_by_mae": worst_banana,
    }


def write_outputs(
    analysis: dict[str, Any],
    summary_output: Path,
    by_day_output: Path,
    by_banana_output: Path,
    worst_output: Path,
) -> dict[str, str]:
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    by_day_output.parent.mkdir(parents=True, exist_ok=True)
    by_banana_output.parent.mkdir(parents=True, exist_ok=True)
    worst_output.parent.mkdir(parents=True, exist_ok=True)

    analysis["by_true_day"].to_csv(by_day_output, index=False)
    analysis["by_banana"].to_csv(by_banana_output, index=False)
    analysis["worst_predictions"].to_csv(worst_output, index=False)

    summary_payload = {
        "overall": analysis["overall"],
        "worst_banana_by_mae": analysis["worst_banana_by_mae"],
        "by_true_day": analysis["by_true_day"].to_dict(orient="records"),
        "outputs": {
            "by_true_day": str(by_day_output),
            "by_banana": str(by_banana_output),
            "worst_predictions": str(worst_output),
        },
    }
    summary_output.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return summary_payload["outputs"] | {"summary": str(summary_output)}


def update_metrics_file(metrics_path: Path, analysis: dict[str, Any], outputs: dict[str, str]) -> None:
    if not metrics_path.exists():
        return

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.setdefault("test_metrics", {}).update(
        {
            "within_2_day_accuracy": analysis["overall"]["within_2_day_accuracy"],
            "max_abs_error": analysis["overall"]["max_abs_error"],
            "mean_residual": analysis["overall"]["mean_residual"],
        },
    )

    train_mean = metrics.get("baselines", {}).get("train_mean")
    if train_mean and "within_2_day_accuracy" not in train_mean:
        # For this project target, the train mean is 3.5 and test days are balanced 0..7.
        # The exact value is recomputed from predictions if a future run stores baseline predictions.
        true_days = pd.Series(
            [record["true_days_to_rotten"] for record in analysis["by_true_day"].to_dict(orient="records")],
            dtype=float,
        )
        counts = pd.Series(
            [record["count"] for record in analysis["by_true_day"].to_dict(orient="records")],
            dtype=float,
        )
        baseline_abs_error = (true_days - float(train_mean["prediction"])).abs()
        train_mean["within_2_day_accuracy"] = float(np.average(baseline_abs_error <= 2.0, weights=counts))

    metrics["failure_analysis"] = {
        "summary": outputs["summary"],
        "by_true_day": outputs["by_true_day"],
        "by_banana": outputs["by_banana"],
        "worst_predictions": outputs["worst_predictions"],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze held-out days-to-rotten regression predictions.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=METRICS_DIR / "regressor_test_predictions.csv",
        help="CSV produced by train_regressor.py.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=METRICS_DIR / "regressor_metrics.json",
        help="Optional metrics JSON to update with failure-analysis pointers.",
    )
    parser.add_argument("--summary-output", type=Path, default=METRICS_DIR / "regressor_failure_analysis.json")
    parser.add_argument("--by-day-output", type=Path, default=METRICS_DIR / "regressor_error_by_true_day.csv")
    parser.add_argument("--by-banana-output", type=Path, default=METRICS_DIR / "regressor_error_by_banana.csv")
    parser.add_argument("--worst-output", type=Path, default=METRICS_DIR / "regressor_worst_predictions.csv")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--no-update-metrics", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    analysis = summarize_predictions(predictions, top_n=args.top_n)
    outputs = write_outputs(
        analysis=analysis,
        summary_output=args.summary_output,
        by_day_output=args.by_day_output,
        by_banana_output=args.by_banana_output,
        worst_output=args.worst_output,
    )
    if not args.no_update_metrics:
        update_metrics_file(args.metrics, analysis, outputs)

    print(json.dumps({"overall": analysis["overall"], "outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
