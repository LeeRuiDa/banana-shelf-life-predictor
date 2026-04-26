from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from .config import REQUIRED_METADATA_COLUMNS, sort_stage_labels


class BananaMetadataError(ValueError):
    """Raised when the metadata CSV does not satisfy the project contract."""


def _normalize_required_string_column(metadata: pd.DataFrame, column: str) -> pd.Series:
    normalized = metadata[column].fillna("").astype(str).str.strip()
    normalized = normalized.replace({"nan": "", "None": ""})
    if normalized.str.len().eq(0).any():
        raise BananaMetadataError(f"Every row must include a non-empty {column}.")
    return normalized


def _normalize_optional_string_column(metadata: pd.DataFrame, column: str) -> pd.Series:
    normalized = metadata[column].fillna("").astype(str).str.strip()
    return normalized.replace({"nan": "", "None": ""})


def validate_metadata_frame(metadata: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_METADATA_COLUMNS if column not in metadata.columns]
    if missing_columns:
        raise BananaMetadataError(f"Metadata is missing required columns: {missing_columns}")

    validated = metadata.copy()
    validated["image_path"] = _normalize_required_string_column(validated, "image_path")
    validated["banana_id"] = _normalize_required_string_column(validated, "banana_id")
    validated["ripeness_stage"] = _normalize_optional_string_column(validated, "ripeness_stage")
    validated["source_dataset"] = _normalize_required_string_column(validated, "source_dataset")
    validated["split_group"] = _normalize_required_string_column(validated, "split_group")
    validated["notes"] = validated["notes"].fillna("").astype(str)

    for column in ("day_index", "days_to_rotten"):
        validated[column] = pd.to_numeric(validated[column], errors="coerce")

    if validated["days_to_rotten"].dropna().lt(0).any():
        raise BananaMetadataError("days_to_rotten must be zero, positive, or missing.")
    if validated["day_index"].dropna().lt(0).any():
        raise BananaMetadataError("day_index must be zero, positive, or missing.")

    return validated.reset_index(drop=True)


def load_metadata(metadata_path: str | Path) -> pd.DataFrame:
    return validate_metadata_frame(pd.read_csv(metadata_path))


def filter_metadata_for_target(metadata: pd.DataFrame, target_column: str) -> pd.DataFrame:
    validated = validate_metadata_frame(metadata)
    if target_column == "ripeness_stage":
        filtered = validated.loc[validated["ripeness_stage"].str.len() > 0]
    elif target_column == "days_to_rotten":
        filtered = validated.loc[validated["days_to_rotten"].notna()]
    else:
        raise ValueError("target_column must be 'ripeness_stage' or 'days_to_rotten'.")
    return filtered.reset_index(drop=True)


def split_metadata_by_group(
    metadata: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    group_column: str = "split_group",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not math.isclose(train_size + val_size + test_size, 1.0, abs_tol=1e-6):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    if metadata[group_column].nunique() < 3:
        raise ValueError("At least three distinct groups are required for train/val/test splitting.")

    primary_splitter = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_indices, temp_indices = next(primary_splitter.split(metadata, groups=metadata[group_column]))

    train_df = metadata.iloc[train_indices].reset_index(drop=True)
    temp_df = metadata.iloc[temp_indices].reset_index(drop=True)

    val_fraction_of_temp = val_size / (val_size + test_size)
    secondary_splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=val_fraction_of_temp,
        random_state=seed,
    )
    val_indices, test_indices = next(
        secondary_splitter.split(temp_df, groups=temp_df[group_column]),
    )

    val_df = temp_df.iloc[val_indices].reset_index(drop=True)
    test_df = temp_df.iloc[test_indices].reset_index(drop=True)
    return train_df, val_df, test_df


def summarize_metadata(metadata: pd.DataFrame) -> dict[str, Any]:
    validated = validate_metadata_frame(metadata)
    staged_rows = validated.loc[validated["ripeness_stage"].str.len() > 0, "ripeness_stage"]
    day_series = validated["day_index"].dropna()
    days_to_rotten_series = validated["days_to_rotten"].dropna()
    return {
        "n_rows": int(len(validated)),
        "n_bananas": int(validated["banana_id"].nunique()),
        "n_split_groups": int(validated["split_group"].nunique()),
        "sources": validated["source_dataset"].value_counts().to_dict(),
        "stages": staged_rows.value_counts().to_dict(),
        "day_range": [int(day_series.min()), int(day_series.max())] if not day_series.empty else None,
        "days_to_rotten_range": [
            float(days_to_rotten_series.min()),
            float(days_to_rotten_series.max()),
        ]
        if not days_to_rotten_series.empty
        else None,
    }


class BananaImageDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        images_root: str | Path | None = None,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        target_column: str = "ripeness_stage",
        class_names: list[str] | None = None,
    ) -> None:
        if target_column not in {"ripeness_stage", "days_to_rotten"}:
            raise ValueError("target_column must be 'ripeness_stage' or 'days_to_rotten'.")

        self.metadata = validate_metadata_frame(metadata)
        self.images_root = Path(images_root) if images_root is not None else None
        self.transform = transform
        self.target_column = target_column
        self.class_names = sort_stage_labels(class_names or self.metadata["ripeness_stage"].unique().tolist())
        self.stage_to_index = {stage: index for index, stage in enumerate(self.class_names)}

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.metadata.iloc[index]
        image_path = self._resolve_image_path(record["image_path"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image) if self.transform is not None else F.pil_to_tensor(image).float() / 255.0

        if self.target_column == "ripeness_stage":
            stage_name = str(record["ripeness_stage"]).strip()
            if not stage_name:
                raise BananaMetadataError("ripeness_stage is missing for a classification sample.")
            target_value = self.stage_to_index[stage_name]
            target_tensor = torch.tensor(target_value, dtype=torch.long)
        else:
            if pd.isna(record["days_to_rotten"]):
                raise BananaMetadataError("days_to_rotten is missing for a regression sample.")
            target_tensor = torch.tensor(float(record["days_to_rotten"]), dtype=torch.float32)

        day_index = -1 if pd.isna(record["day_index"]) else int(record["day_index"])
        days_to_rotten = float("nan") if pd.isna(record["days_to_rotten"]) else float(record["days_to_rotten"])

        return {
            "image": image_tensor,
            "target": target_tensor,
            "image_path": str(image_path),
            "banana_id": str(record["banana_id"]),
            "day_index": day_index,
            "days_to_rotten": days_to_rotten,
            "ripeness_stage": str(record["ripeness_stage"]),
            "source_dataset": str(record["source_dataset"]),
            "split_group": str(record["split_group"]),
            "notes": str(record["notes"]),
        }

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        if self.images_root is None:
            return path
        return (self.images_root / path).resolve()
