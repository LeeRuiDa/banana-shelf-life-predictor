from __future__ import annotations

import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.datasets import (
    BananaImageDataset,
    BananaMetadataError,
    filter_metadata_for_target,
    split_metadata_by_group,
    validate_metadata_frame,
)
from src.transforms import build_eval_transforms


def make_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_path": "banana_1_day0.png",
                "banana_id": "B001",
                "day_index": 0,
                "days_to_rotten": 7,
                "ripeness_stage": "unripe",
                "source_dataset": "synthetic",
                "split_group": "B001",
                "notes": "",
            },
            {
                "image_path": "banana_1_day1.png",
                "banana_id": "B001",
                "day_index": 1,
                "days_to_rotten": 6,
                "ripeness_stage": "ripe",
                "source_dataset": "synthetic",
                "split_group": "B001",
                "notes": "",
            },
            {
                "image_path": "banana_2_day0.png",
                "banana_id": "B002",
                "day_index": 0,
                "days_to_rotten": 7,
                "ripeness_stage": "unripe",
                "source_dataset": "synthetic",
                "split_group": "B002",
                "notes": "",
            },
            {
                "image_path": "banana_3_day0.png",
                "banana_id": "B003",
                "day_index": 0,
                "days_to_rotten": 7,
                "ripeness_stage": "unripe",
                "source_dataset": "synthetic",
                "split_group": "B003",
                "notes": "",
            },
            {
                "image_path": "banana_4_day0.png",
                "banana_id": "B004",
                "day_index": 0,
                "days_to_rotten": 7,
                "ripeness_stage": "rotten",
                "source_dataset": "synthetic",
                "split_group": "B004",
                "notes": "",
            },
        ],
    )


def test_validate_metadata_rejects_missing_columns() -> None:
    metadata = pd.DataFrame([{"image_path": "banana.png"}])
    with pytest.raises(BananaMetadataError):
        validate_metadata_frame(metadata)


def test_group_split_keeps_groups_disjoint() -> None:
    metadata = make_metadata()
    train_df, val_df, test_df = split_metadata_by_group(metadata, train_size=0.6, val_size=0.2, test_size=0.2, seed=7)

    train_groups = set(train_df["split_group"])
    val_groups = set(val_df["split_group"])
    test_groups = set(test_df["split_group"])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_dataset_returns_tensor_and_target(tmp_path) -> None:
    image_path = tmp_path / "banana_1_day0.png"
    Image.new("RGB", (64, 64), color=(255, 255, 0)).save(image_path)

    metadata = pd.DataFrame(
        [
            {
                "image_path": image_path.name,
                "banana_id": "B001",
                "day_index": 0,
                "days_to_rotten": 7,
                "ripeness_stage": "unripe",
                "source_dataset": "synthetic",
                "split_group": "B001",
                "notes": "",
            },
        ],
    )
    dataset = BananaImageDataset(
        metadata=metadata,
        images_root=tmp_path,
        transform=build_eval_transforms(128),
        target_column="ripeness_stage",
    )

    sample = dataset[0]
    assert isinstance(sample["image"], torch.Tensor)
    assert tuple(sample["image"].shape) == (3, 128, 128)
    assert sample["target"].dtype == torch.long


def test_validate_metadata_allows_missing_regression_fields_for_classification_only_rows() -> None:
    metadata = pd.DataFrame(
        [
            {
                "image_path": "class_a/example.png",
                "banana_id": "IMG001",
                "day_index": None,
                "days_to_rotten": None,
                "ripeness_stage": "class_a",
                "source_dataset": "classification_repo",
                "split_group": "IMG001",
                "notes": "",
            },
        ],
    )

    validated = validate_metadata_frame(metadata)
    assert validated["day_index"].isna().all()
    assert validated["days_to_rotten"].isna().all()


def test_classification_only_rows_collate_with_missing_regression_fields(tmp_path) -> None:
    image_path = tmp_path / "class_a" / "example.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), color=(255, 255, 0)).save(image_path)

    metadata = pd.DataFrame(
        [
            {
                "image_path": "class_a/example.png",
                "banana_id": "IMG001",
                "day_index": None,
                "days_to_rotten": None,
                "ripeness_stage": "class_a",
                "source_dataset": "classification_repo",
                "split_group": "IMG001",
                "notes": "",
            },
            {
                "image_path": "class_a/example.png",
                "banana_id": "IMG002",
                "day_index": None,
                "days_to_rotten": None,
                "ripeness_stage": "class_a",
                "source_dataset": "classification_repo",
                "split_group": "IMG002",
                "notes": "",
            },
        ],
    )

    dataset = BananaImageDataset(
        metadata=metadata,
        images_root=tmp_path,
        transform=build_eval_transforms(64),
        target_column="ripeness_stage",
    )
    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert batch["day_index"].tolist() == [-1, -1]
    assert torch.isnan(batch["days_to_rotten"]).all()


def test_filter_metadata_for_target_drops_rows_without_requested_target() -> None:
    metadata = pd.DataFrame(
        [
            {
                "image_path": "img_a.png",
                "banana_id": "A",
                "day_index": 0,
                "days_to_rotten": 7,
                "ripeness_stage": "unripe",
                "source_dataset": "mixed",
                "split_group": "A",
                "notes": "",
            },
            {
                "image_path": "img_b.png",
                "banana_id": "B",
                "day_index": None,
                "days_to_rotten": None,
                "ripeness_stage": "class_a",
                "source_dataset": "mixed",
                "split_group": "B",
                "notes": "",
            },
        ],
    )

    regression_only = filter_metadata_for_target(metadata, "days_to_rotten")
    assert list(regression_only["banana_id"]) == ["A"]
