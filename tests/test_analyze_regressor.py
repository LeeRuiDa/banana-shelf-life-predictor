from __future__ import annotations

import pandas as pd

from src.analyze_regressor import summarize_predictions


def test_summarize_predictions_reports_by_day_and_within_two_days() -> None:
    predictions = pd.DataFrame(
        [
            {
                "image_path": "a.jpg",
                "banana_id": "Banana_ID_001",
                "day_index": 0,
                "true_days_to_rotten": 7.0,
                "predicted_days_to_rotten": 6.5,
                "residual": -0.5,
            },
            {
                "image_path": "b.jpg",
                "banana_id": "Banana_ID_001",
                "day_index": 7,
                "true_days_to_rotten": 0.0,
                "predicted_days_to_rotten": 2.5,
                "residual": 2.5,
            },
        ],
    )

    analysis = summarize_predictions(predictions, top_n=1)

    assert analysis["overall"]["within_1_day_accuracy"] == 0.5
    assert analysis["overall"]["within_2_day_accuracy"] == 0.5
    assert list(analysis["by_true_day"]["true_days_to_rotten"]) == [0.0, 7.0]
    assert analysis["worst_predictions"].iloc[0]["image_path"] == "b.jpg"
