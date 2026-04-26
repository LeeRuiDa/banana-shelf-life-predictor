from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError

from .config import GENERATED_METADATA_DIR, IMAGE_EXTENSIONS
from .datasets import summarize_metadata, validate_metadata_frame

SPLIT_NAMES = {"train", "training", "val", "valid", "validation", "test"}
CLASS_LABEL_PATTERNS = (
    re.compile(r"^class\s+.+$", re.IGNORECASE),
    re.compile(r"^ripeness\s+.+$", re.IGNORECASE),
)
DEFAULT_DAY_PATTERN = re.compile(r"day[_\s-]*(\d+)", re.IGNORECASE)


def slugify(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return normalized or "unknown"


def natural_sort_key(text: str) -> list[Any]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", text)]


def normalize_relative_path(raw_path: str) -> str:
    return PurePosixPath(str(raw_path).replace("\\", "/")).as_posix()


def infer_stage_from_days_to_rotten(days_to_rotten: float | int | None) -> str:
    if days_to_rotten is None or pd.isna(days_to_rotten):
        return ""
    if float(days_to_rotten) <= 0:
        return "rotten"
    if float(days_to_rotten) <= 2:
        return "overripe"
    if float(days_to_rotten) <= 4:
        return "ripe"
    return "unripe"


def _write_metadata_output(
    metadata: pd.DataFrame,
    output_path: Path,
    summary_output: Path | None = None,
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validated = validate_metadata_frame(metadata)
    validated = validated.sort_values(
        by=["source_dataset", "banana_id", "day_index", "image_path"],
        na_position="last",
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(output_path, index=False)

    summary = summarize_metadata(validated)
    summary["output_path"] = str(output_path)
    if extra_summary:
        summary.update(extra_summary)

    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def prepare_timeseries_csv(
    csv_path: str | Path,
    output_path: str | Path,
    source_name: str,
    image_path_column: str = "file_path",
    banana_id_column: str = "banana_id",
    day_column: str = "day_number",
    days_left_column: str = "days_remaining",
    split_column: str = "split",
    day_number_offset: int = 1,
    path_prefix: str | None = None,
    summary_output: str | Path | None = None,
) -> dict[str, Any]:
    raw = pd.read_csv(csv_path)
    prefix = normalize_relative_path(path_prefix) if path_prefix else ""

    normalized_paths = raw[image_path_column].map(normalize_relative_path)
    if prefix:
        normalized_paths = normalized_paths.map(lambda value: normalize_relative_path(f"{prefix}/{value}"))

    metadata = pd.DataFrame(
        {
            "image_path": normalized_paths,
            "banana_id": raw[banana_id_column].map(lambda value: f"banana_{int(value):03d}"),
            "day_index": raw[day_column].astype(int) - int(day_number_offset),
            "days_to_rotten": raw[days_left_column].astype(float),
            "ripeness_stage": raw[days_left_column].map(infer_stage_from_days_to_rotten),
            "source_dataset": source_name,
            "split_group": raw[banana_id_column].map(lambda value: f"banana_{int(value):03d}"),
            "notes": raw[split_column].map(lambda value: f"source_split={value}; reference_csv"),
        },
    )

    metadata["source_split"] = raw[split_column].astype(str)
    metadata["original_file_path"] = raw[image_path_column].astype(str)

    return _write_metadata_output(
        metadata=metadata,
        output_path=Path(output_path),
        summary_output=Path(summary_output) if summary_output is not None else None,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_duplicate_safe_split_groups(
    banana_ids: list[str],
    duplicate_paths_by_hash: dict[str, list[str]],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    parent = {banana_id: banana_id for banana_id in banana_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for paths in duplicate_paths_by_hash.values():
        duplicate_banana_ids = sorted(
            {PurePosixPath(path).parts[0] for path in paths},
            key=natural_sort_key,
        )
        if len(duplicate_banana_ids) < 2:
            continue
        first_id = duplicate_banana_ids[0]
        for banana_id in duplicate_banana_ids[1:]:
            union(first_id, banana_id)

    components: dict[str, list[str]] = defaultdict(list)
    for banana_id in banana_ids:
        components[find(banana_id)].append(banana_id)

    split_group_by_banana: dict[str, str] = {}
    duplicate_split_groups: dict[str, list[str]] = {}
    duplicate_group_index = 1
    for members in sorted(components.values(), key=lambda values: natural_sort_key(values[0])):
        members = sorted(members, key=natural_sort_key)
        if len(members) == 1:
            split_group_by_banana[members[0]] = members[0]
            continue

        split_group = f"duplicate_group_{duplicate_group_index:03d}"
        duplicate_group_index += 1
        duplicate_split_groups[split_group] = members
        for banana_id in members:
            split_group_by_banana[banana_id] = split_group

    return split_group_by_banana, duplicate_split_groups


def prepare_timeseries_folders(
    dataset_root: str | Path,
    output_path: str | Path,
    source_name: str,
    rotten_day_index: int = 7,
    day_pattern: re.Pattern[str] = DEFAULT_DAY_PATTERN,
    summary_output: str | Path | None = None,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    expected_days = set(range(0, rotten_day_index + 1))
    rows: list[dict[str, Any]] = []
    days_by_banana: dict[str, set[int]] = defaultdict(set)
    image_count_by_banana: Counter[str] = Counter()
    image_sizes: Counter[str] = Counter()
    file_hashes: dict[str, list[str]] = defaultdict(list)
    unreadable_files: list[str] = []
    inferred_day_files: dict[str, dict[str, int]] = defaultdict(dict)

    banana_dirs = [path for path in dataset_root.iterdir() if path.is_dir()]
    for banana_dir in sorted(banana_dirs, key=lambda item: natural_sort_key(item.name)):
        banana_id = banana_dir.name
        image_paths = [
            path
            for path in sorted(banana_dir.rglob("*"), key=lambda item: natural_sort_key(item.as_posix()))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        parsed_paths: list[tuple[int, Path, bool]] = []
        unparsed_paths: list[Path] = []

        for image_path in image_paths:
            day_match = day_pattern.search(image_path.stem)
            if day_match is None:
                unparsed_paths.append(image_path)
            else:
                parsed_paths.append((int(day_match.group(1)), image_path, False))

        observed_days = {day_index for day_index, _, _ in parsed_paths}
        missing_for_folder = sorted(expected_days - observed_days)
        if unparsed_paths:
            if len(unparsed_paths) != len(missing_for_folder):
                raise ValueError(
                    f"Could not infer day_index for {len(unparsed_paths)} file(s) in {banana_id}; "
                    f"missing expected days are {missing_for_folder}."
                )
            for day_index, image_path in zip(missing_for_folder, unparsed_paths):
                parsed_paths.append((day_index, image_path, True))
                inferred_day_files[banana_id][image_path.name] = day_index

        for day_index, image_path, inferred_day in sorted(parsed_paths, key=lambda item: (item[0], natural_sort_key(item[1].name))):
            days_to_rotten = rotten_day_index - day_index
            if days_to_rotten < 0:
                raise ValueError(
                    f"day_index {day_index} in {image_path} exceeds rotten_day_index {rotten_day_index}."
                )

            relative_path = image_path.relative_to(dataset_root).as_posix()
            days_by_banana[banana_id].add(day_index)
            image_count_by_banana[banana_id] += 1
            try:
                with Image.open(image_path) as image:
                    image_sizes[f"{image.width}x{image.height}"] += 1
            except (OSError, UnidentifiedImageError):
                unreadable_files.append(relative_path)

            file_hashes[_file_sha256(image_path)].append(relative_path)
            rows.append(
                {
                    "image_path": relative_path,
                    "banana_id": banana_id,
                    "day_index": day_index,
                    "days_to_rotten": float(days_to_rotten),
                    "ripeness_stage": infer_stage_from_days_to_rotten(days_to_rotten),
                    "source_dataset": source_name,
                    "split_group": banana_id,
                    "notes": f"source_folder={banana_id}; rotten_day_index={rotten_day_index}; inferred_day={inferred_day}",
                    "raw_day_label": image_path.stem,
                },
            )

    missing_days = {
        banana_id: sorted(expected_days - observed_days)
        for banana_id, observed_days in sorted(days_by_banana.items(), key=lambda item: natural_sort_key(item[0]))
        if expected_days - observed_days
    }
    unexpected_days = {
        banana_id: sorted(observed_days - expected_days)
        for banana_id, observed_days in sorted(days_by_banana.items(), key=lambda item: natural_sort_key(item[0]))
        if observed_days - expected_days
    }
    duplicate_files = [
        {"sha256": file_hash, "paths": paths}
        for file_hash, paths in sorted(file_hashes.items())
        if len(paths) > 1
    ]
    images_per_banana = {
        banana_id: image_count
        for banana_id, image_count in sorted(image_count_by_banana.items(), key=lambda item: natural_sort_key(item[0]))
    }

    metadata = pd.DataFrame(rows)
    banana_ids = sorted(days_by_banana.keys(), key=natural_sort_key)
    split_group_by_banana, duplicate_split_groups = _build_duplicate_safe_split_groups(
        banana_ids=banana_ids,
        duplicate_paths_by_hash=file_hashes,
    )
    metadata["split_group"] = metadata["banana_id"].map(split_group_by_banana)
    return _write_metadata_output(
        metadata=metadata,
        output_path=Path(output_path),
        summary_output=Path(summary_output) if summary_output is not None else None,
        extra_summary={
            "expected_day_indices": sorted(expected_days),
            "images_per_banana": images_per_banana,
            "missing_day_indices_by_banana": missing_days,
            "unexpected_day_indices_by_banana": unexpected_days,
            "inferred_day_files": {
                banana_id: dict(files)
                for banana_id, files in sorted(inferred_day_files.items(), key=lambda item: natural_sort_key(item[0]))
            },
            "duplicate_file_groups": duplicate_files,
            "duplicate_safe_split_groups": duplicate_split_groups,
            "image_size_distribution": dict(sorted(image_sizes.items())),
            "unreadable_files": unreadable_files,
        },
    )


def prepare_classification_folders(
    dataset_root: str | Path,
    output_path: str | Path,
    source_name: str,
    stage_map: dict[str, str] | None = None,
    summary_output: str | Path | None = None,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    stage_map = {key.lower(): value for key, value in (stage_map or {}).items()}
    rows: list[dict[str, Any]] = []

    for path in sorted(dataset_root.rglob("*"), key=lambda item: natural_sort_key(item.as_posix())):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = path.relative_to(dataset_root)
        parts = list(relative_path.parts[:-1])
        lower_parts = [part.lower() for part in parts]

        split_name = next((part for part in lower_parts if part in SPLIT_NAMES), "")
        raw_label = next(
            (
                part
                for part in parts
                if any(pattern.match(part) for pattern in CLASS_LABEL_PATTERNS)
            ),
            parts[-1] if parts else "unlabeled",
        )

        normalized_label = stage_map.get(raw_label.lower(), slugify(raw_label))
        unique_id = f"{slugify(source_name)}__{slugify(relative_path.with_suffix('').as_posix())}"
        note_parts = [f"raw_label={raw_label}"]
        if split_name:
            note_parts.append(f"source_split={split_name}")

        rows.append(
            {
                "image_path": relative_path.as_posix(),
                "banana_id": unique_id,
                "day_index": pd.NA,
                "days_to_rotten": pd.NA,
                "ripeness_stage": normalized_label,
                "source_dataset": source_name,
                "split_group": unique_id,
                "notes": "; ".join(note_parts),
                "raw_label": raw_label,
                "source_split": split_name,
            },
        )

    metadata = pd.DataFrame(rows)
    return _write_metadata_output(
        metadata=metadata,
        output_path=Path(output_path),
        summary_output=Path(summary_output) if summary_output is not None else None,
    )


def build_flat_folder_template(
    images_root: str | Path,
    output_path: str | Path,
    source_name: str,
    note: str = "manual_annotation_required",
    summary_output: str | Path | None = None,
) -> dict[str, Any]:
    images_root = Path(images_root)
    rows: list[dict[str, Any]] = []

    image_paths = [
        path
        for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for sequence_index, path in enumerate(sorted(image_paths, key=lambda item: natural_sort_key(item.as_posix())), start=1):
        relative_path = path.relative_to(images_root)
        unique_id = f"{slugify(source_name)}_{sequence_index:06d}"
        rows.append(
            {
                "image_path": relative_path.as_posix(),
                "banana_id": unique_id,
                "day_index": pd.NA,
                "days_to_rotten": pd.NA,
                "ripeness_stage": "",
                "source_dataset": source_name,
                "split_group": unique_id,
                "notes": note,
                "sequence_index": sequence_index,
                "filename_stem": path.stem,
            },
        )

    metadata = pd.DataFrame(rows)
    return _write_metadata_output(
        metadata=metadata,
        output_path=Path(output_path),
        summary_output=Path(summary_output) if summary_output is not None else None,
    )


def _load_stage_map(stage_map_file: str | Path | None) -> dict[str, str] | None:
    if stage_map_file is None:
        return None
    return json.loads(Path(stage_map_file).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize banana datasets into the shared metadata schema.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    timeseries_parser = subparsers.add_parser("timeseries-csv", help="Normalize a day-by-day CSV reference file.")
    timeseries_parser.add_argument("--csv", type=Path, required=True)
    timeseries_parser.add_argument("--output", type=Path, default=GENERATED_METADATA_DIR / "banana_metadata_timeseries.csv")
    timeseries_parser.add_argument("--source-name", type=str, required=True)
    timeseries_parser.add_argument("--image-path-column", type=str, default="file_path")
    timeseries_parser.add_argument("--banana-id-column", type=str, default="banana_id")
    timeseries_parser.add_argument("--day-column", type=str, default="day_number")
    timeseries_parser.add_argument("--days-left-column", type=str, default="days_remaining")
    timeseries_parser.add_argument("--split-column", type=str, default="split")
    timeseries_parser.add_argument("--day-number-offset", type=int, default=1)
    timeseries_parser.add_argument("--path-prefix", type=str, default="")
    timeseries_parser.add_argument("--summary-output", type=Path, default=None)

    folder_timeseries_parser = subparsers.add_parser(
        "timeseries-folders",
        help="Normalize Banana_ID_xxx folders containing Day_0..Day_N images.",
    )
    folder_timeseries_parser.add_argument("--dataset-root", type=Path, required=True)
    folder_timeseries_parser.add_argument("--output", type=Path, default=GENERATED_METADATA_DIR / "banana_metadata_regression.csv")
    folder_timeseries_parser.add_argument("--source-name", type=str, required=True)
    folder_timeseries_parser.add_argument("--rotten-day-index", type=int, default=7)
    folder_timeseries_parser.add_argument("--summary-output", type=Path, default=None)

    classification_parser = subparsers.add_parser("classification-folders", help="Build metadata from class-organized image folders.")
    classification_parser.add_argument("--dataset-root", type=Path, required=True)
    classification_parser.add_argument("--output", type=Path, default=GENERATED_METADATA_DIR / "banana_metadata_classification.csv")
    classification_parser.add_argument("--source-name", type=str, required=True)
    classification_parser.add_argument("--stage-map-file", type=Path, default=None)
    classification_parser.add_argument("--summary-output", type=Path, default=None)

    template_parser = subparsers.add_parser("flat-template", help="Create a manual-labeling template from a flat image folder.")
    template_parser.add_argument("--images-root", type=Path, required=True)
    template_parser.add_argument("--output", type=Path, default=GENERATED_METADATA_DIR / "banana_metadata_template.csv")
    template_parser.add_argument("--source-name", type=str, required=True)
    template_parser.add_argument("--note", type=str, default="manual_annotation_required")
    template_parser.add_argument("--summary-output", type=Path, default=None)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "timeseries-csv":
        summary = prepare_timeseries_csv(
            csv_path=args.csv,
            output_path=args.output,
            source_name=args.source_name,
            image_path_column=args.image_path_column,
            banana_id_column=args.banana_id_column,
            day_column=args.day_column,
            days_left_column=args.days_left_column,
            split_column=args.split_column,
            day_number_offset=args.day_number_offset,
            path_prefix=args.path_prefix,
            summary_output=args.summary_output,
        )
    elif args.command == "timeseries-folders":
        summary = prepare_timeseries_folders(
            dataset_root=args.dataset_root,
            output_path=args.output,
            source_name=args.source_name,
            rotten_day_index=args.rotten_day_index,
            summary_output=args.summary_output,
        )
    elif args.command == "classification-folders":
        summary = prepare_classification_folders(
            dataset_root=args.dataset_root,
            output_path=args.output,
            source_name=args.source_name,
            stage_map=_load_stage_map(args.stage_map_file),
            summary_output=args.summary_output,
        )
    elif args.command == "flat-template":
        summary = build_flat_folder_template(
            images_root=args.images_root,
            output_path=args.output,
            source_name=args.source_name,
            note=args.note,
            summary_output=args.summary_output,
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
