from __future__ import annotations

from src.explain import build_prediction_interpretation, summarize_explanation


def test_summarize_explanation_for_classification() -> None:
    summary = summarize_explanation(
        {
            "task": "classification",
            "predicted_stage": "ripe",
            "confidence": 0.8123,
        }
    )

    assert "ripe" in summary
    assert "81.2%" in summary


def test_summarize_explanation_for_regression() -> None:
    summary = summarize_explanation(
        {
            "task": "regression",
            "predicted_days_to_rotten": 2.3456,
        }
    )

    assert summary == "2.35 predicted days to rotten"


def test_build_prediction_interpretation_for_combined_prediction() -> None:
    interpretation = build_prediction_interpretation(
        {
            "stage": {
                "task": "classification",
                "predicted_stage": "overripe",
                "confidence": 0.9,
            },
            "days_left": {
                "task": "regression",
                "predicted_days_to_rotten": 1.4,
            },
        }
    )

    assert interpretation == "Late-stage banana that should be eaten soon."


def test_build_prediction_interpretation_for_regression_only() -> None:
    interpretation = build_prediction_interpretation(
        {
            "days_left": {
                "task": "regression",
                "predicted_days_to_rotten": 5.8,
            },
        }
    )

    assert interpretation == "Regressor estimates a long remaining shelf life."
