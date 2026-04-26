from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
EXTERNAL_METADATA_DIR = METADATA_DIR / "external"
GENERATED_METADATA_DIR = METADATA_DIR / "generated"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"

DEFAULT_CLASSIFIER_CHECKPOINT = MODELS_DIR / "classifier_best.pt"
DEFAULT_REGRESSOR_CHECKPOINT = MODELS_DIR / "regressor_best.pt"

TARGET_DEFINITION = "days until the banana reaches the rotten stage"
REQUIRED_METADATA_COLUMNS = (
    "image_path",
    "banana_id",
    "day_index",
    "days_to_rotten",
    "ripeness_stage",
    "source_dataset",
    "split_group",
    "notes",
)
DEFAULT_STAGE_ORDER = ("unripe", "ripe", "overripe", "rotten")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def sort_stage_labels(labels: list[str] | tuple[str, ...]) -> list[str]:
    unique_labels = list(dict.fromkeys(str(label) for label in labels))
    known = [label for label in DEFAULT_STAGE_ORDER if label in unique_labels]
    extras = sorted(label for label in unique_labels if label not in DEFAULT_STAGE_ORDER)
    return known + extras
