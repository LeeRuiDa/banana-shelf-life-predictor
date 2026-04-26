from __future__ import annotations

import pandas as pd
from PIL import Image

from src.prepare_metadata import (
    build_flat_folder_template,
    prepare_classification_folders,
    prepare_timeseries_csv,
    prepare_timeseries_folders,
)


def test_prepare_timeseries_csv_zero_bases_days(tmp_path) -> None:
    csv_path = tmp_path / "banana_reference.csv"
    csv_path.write_text(
        "\n".join(
            [
                "banana_id,day_number,days_remaining,split,file_path",
                '1,1,7,train,"Banana Images Dataset\\Banana1,1.jpeg"',
                '1,8,0,train,"Banana Images Dataset\\Banana1,8.jpeg"',
            ],
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "normalized.csv"
    prepare_timeseries_csv(
        csv_path=csv_path,
        output_path=output_path,
        source_name="banana_images_reference",
        path_prefix="kaggle_reference",
    )

    metadata = pd.read_csv(output_path)
    assert metadata.loc[0, "day_index"] == 0
    assert metadata.loc[1, "day_index"] == 7
    assert metadata.loc[0, "days_to_rotten"] == 7
    assert metadata.loc[1, "ripeness_stage"] == "rotten"
    assert metadata.loc[0, "image_path"] == "kaggle_reference/Banana Images Dataset/Banana1,1.jpeg"


def test_prepare_classification_folders_uses_folder_labels(tmp_path) -> None:
    class_a = tmp_path / "train" / "Class A"
    class_b = tmp_path / "test" / "Class B"
    class_a.mkdir(parents=True)
    class_b.mkdir(parents=True)

    Image.new("RGB", (32, 32), color=(255, 255, 0)).save(class_a / "sample_a.png")
    Image.new("RGB", (32, 32), color=(220, 220, 0)).save(class_b / "sample_b.png")

    output_path = tmp_path / "classification.csv"
    prepare_classification_folders(
        dataset_root=tmp_path,
        output_path=output_path,
        source_name="banana_ripeness_repo",
    )

    metadata = pd.read_csv(output_path)
    assert set(metadata["ripeness_stage"]) == {"class_a", "class_b"}
    assert metadata["days_to_rotten"].isna().all()
    assert metadata["notes"].str.contains("source_split=").all()


def test_prepare_timeseries_folders_infers_days_and_groups_by_banana(tmp_path) -> None:
    banana_dir = tmp_path / "Banana_ID_001"
    banana_dir.mkdir()
    Image.new("RGB", (32, 32), color=(255, 255, 0)).save(banana_dir / "Day_0.jpg")
    Image.new("RGB", (32, 32), color=(120, 90, 20)).save(banana_dir / "Day_7.jpg")

    output_path = tmp_path / "regression.csv"
    summary = prepare_timeseries_folders(
        dataset_root=tmp_path,
        output_path=output_path,
        source_name="day0_day7",
    )

    metadata = pd.read_csv(output_path)
    assert metadata.loc[0, "banana_id"] == "Banana_ID_001"
    assert metadata.loc[0, "split_group"] == "Banana_ID_001"
    assert metadata.loc[0, "day_index"] == 0
    assert metadata.loc[0, "days_to_rotten"] == 7
    assert metadata.loc[1, "day_index"] == 7
    assert metadata.loc[1, "ripeness_stage"] == "rotten"
    assert summary["missing_day_indices_by_banana"] == {"Banana_ID_001": [1, 2, 3, 4, 5, 6]}


def test_build_flat_folder_template_marks_rows_for_manual_annotation(tmp_path) -> None:
    root = tmp_path / "flat_images"
    root.mkdir()
    Image.new("RGB", (32, 32), color=(255, 255, 0)).save(root / "2.jpg")
    Image.new("RGB", (32, 32), color=(255, 255, 0)).save(root / "10.jpg")

    output_path = tmp_path / "template.csv"
    build_flat_folder_template(
        images_root=root,
        output_path=output_path,
        source_name="mendeley_flat_archive",
    )

    metadata = pd.read_csv(output_path)
    assert metadata["ripeness_stage"].fillna("").eq("").all()
    assert metadata["notes"].eq("manual_annotation_required").all()
    assert metadata.loc[0, "image_path"] == "2.jpg"
    assert metadata.loc[1, "image_path"] == "10.jpg"
