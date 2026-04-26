from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import zipfile

import requests


MENDELEY_PUBLIC_API_BASE = "https://data.mendeley.com/public-api"


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    display_name: str
    provider: str
    role: str
    homepage: str
    notes: str
    dataset_id: str | None = None
    version: int | None = None
    static_artifacts: tuple[dict[str, Any], ...] = ()
    default_artifact_names: tuple[str, ...] = ()


KNOWN_SOURCES: dict[str, SourceSpec] = {
    "mendeley_ripen_banana_2025": SourceSpec(
        source_id="mendeley_ripen_banana_2025",
        display_name="Ripen Banana Dataset",
        provider="mendeley",
        role="time_series",
        homepage="https://data.mendeley.com/datasets/j9sp322drp/1",
        notes=(
            "Verified on 2026-04-24. The natural-ripening subset is the best match for shelf-life work, "
            "but the archive is a flat numbered image folder and likely needs a manual labeling pass for "
            "banana_id/day_index before regression training."
        ),
        dataset_id="j9sp322drp",
        version=1,
        default_artifact_names=("Without Carbide.zip",),
    ),
    "mendeley_prata_catarina_2023": SourceSpec(
        source_id="mendeley_prata_catarina_2023",
        display_name="Dataset of Banana Prata Catarina Images Labeled in Eight Ripeness Stages",
        provider="mendeley",
        role="classification",
        homepage="https://data.mendeley.com/datasets/7vb4djkbrc/1",
        notes=(
            "Verified on 2026-04-24. This is an eight-stage classification dataset packaged as a RAR archive. "
            "Download is scriptable, but extraction may require 7-Zip or manual extraction outside Python."
        ),
        dataset_id="7vb4djkbrc",
        version=1,
    ),
    "mendeley_bananaimagebd_2024": SourceSpec(
        source_id="mendeley_bananaimagebd_2024",
        display_name="BananaImageBD",
        provider="mendeley",
        role="classification",
        homepage="https://data.mendeley.com/datasets/ptfscwtnyz/2",
        notes=(
            "Verified on 2026-04-24. Four-stage banana ripeness dataset with Green, Semi-ripe, "
            "Ripe, and Overripe folders. The original ripeness archive is the default for honest V1 metrics; "
            "the augmented archive is available for later experiments."
        ),
        dataset_id="ptfscwtnyz",
        version=2,
        default_artifact_names=("Banana Ripeness Detection Dataset.zip",),
    ),
    "mendeley_banana_day0_day7_2026": SourceSpec(
        source_id="mendeley_banana_day0_day7_2026",
        display_name="banana_ripening_dataset_day0_to_day7",
        provider="mendeley",
        role="time_series_regression",
        homepage="https://data.mendeley.com/datasets/d5tczj7fs7/1",
        notes=(
            "Verified on 2026-04-24. Structured day0-to-day7 time-series image dataset with "
            "individual Banana_ID_xxx folders, suitable for grouped days-to-rotten regression."
        ),
        dataset_id="d5tczj7fs7",
        version=1,
        default_artifact_names=("banana_ripening_dataset_day0_to_day7.zip",),
    ),
    "hf_banana_images_reference_csv": SourceSpec(
        source_id="hf_banana_images_reference_csv",
        display_name="Banana Images Dataset reference CSV",
        provider="huggingface",
        role="time_series_reference",
        homepage="https://huggingface.co/datasets/Saeraj/Banana_Images_Dataset",
        notes=(
            "Verified on 2026-04-24. This mirror contains the 12-banana day-by-day CSV with banana_id, "
            "day_number, days_remaining, split, and file_path, but it does not host the actual image files."
        ),
        static_artifacts=(
            {
                "name": "banana_dataset.csv",
                "filename": "banana_dataset.csv",
                "url": "https://huggingface.co/datasets/Saeraj/Banana_Images_Dataset/resolve/main/banana_dataset.csv?download=true",
                "size_bytes": None,
                "sha256": None,
                "content_type": "text/csv",
                "archive_format": "csv",
            },
        ),
        default_artifact_names=("banana_dataset.csv",),
    ),
    "github_bananaripeness": SourceSpec(
        source_id="github_bananaripeness",
        display_name="BananaRipeness",
        provider="github",
        role="classification",
        homepage="https://github.com/luischuquim/BananaRipeness",
        notes=(
            "Verified on 2026-04-24. Public GitHub repository with real and synthetic Cavendish banana images "
            "organized by ripeness folders. Use the classification-folders metadata builder after checkout."
        ),
    ),
}


def list_known_sources() -> list[dict[str, Any]]:
    return [
        {
            "source_id": source.source_id,
            "display_name": source.display_name,
            "provider": source.provider,
            "role": source.role,
            "homepage": source.homepage,
            "notes": source.notes,
            "default_artifact_names": list(source.default_artifact_names),
        }
        for source in KNOWN_SOURCES.values()
    ]


def get_source_spec(source_id: str) -> SourceSpec:
    try:
        return KNOWN_SOURCES[source_id]
    except KeyError as error:
        available = ", ".join(sorted(KNOWN_SOURCES))
        raise KeyError(f"Unknown source_id '{source_id}'. Available sources: {available}") from error


def fetch_source_manifest(source_id: str) -> dict[str, Any]:
    spec = get_source_spec(source_id)
    if spec.provider == "mendeley":
        return _fetch_mendeley_manifest(spec)
    if spec.static_artifacts:
        return {
            "source_id": spec.source_id,
            "display_name": spec.display_name,
            "provider": spec.provider,
            "role": spec.role,
            "homepage": spec.homepage,
            "notes": spec.notes,
            "artifacts": list(spec.static_artifacts),
            "default_artifact_names": list(spec.default_artifact_names),
        }
    return {
        "source_id": spec.source_id,
        "display_name": spec.display_name,
        "provider": spec.provider,
        "role": spec.role,
        "homepage": spec.homepage,
        "notes": spec.notes,
        "artifacts": [],
        "default_artifact_names": [],
    }


def _fetch_mendeley_manifest(spec: SourceSpec) -> dict[str, Any]:
    if spec.dataset_id is None or spec.version is None:
        raise ValueError(f"Mendeley source '{spec.source_id}' is missing dataset_id/version metadata.")

    response = requests.get(
        f"{MENDELEY_PUBLIC_API_BASE}/datasets/{spec.dataset_id}",
        params={"version": spec.version, "fields": "*"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()

    artifacts = []
    for file_entry in payload.get("files", []):
        content_details = file_entry.get("content_details", {})
        filename = file_entry["filename"]
        artifacts.append(
            {
                "name": filename,
                "filename": filename,
                "url": content_details.get("download_url"),
                "size_bytes": content_details.get("size"),
                "sha256": content_details.get("sha256_hash"),
                "content_type": content_details.get("content_type"),
                "archive_format": Path(filename).suffix.lower().lstrip("."),
            },
        )

    licence = payload.get("data_licence") or {}
    return {
        "source_id": spec.source_id,
        "display_name": spec.display_name,
        "provider": spec.provider,
        "role": spec.role,
        "homepage": spec.homepage,
        "notes": spec.notes,
        "dataset_id": payload.get("id"),
        "dataset_version": payload.get("version"),
        "doi": (payload.get("doi") or {}).get("id"),
        "license_name": licence.get("short_name"),
        "license_url": licence.get("url"),
        "description": payload.get("description"),
        "artifacts": artifacts,
        "default_artifact_names": list(spec.default_artifact_names),
    }


def save_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def download_artifact(
    artifact: dict[str, Any],
    output_path: str | Path,
    overwrite: bool = False,
    chunk_size: int = 1024 * 1024 * 4,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        return output_path

    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    expected_size = artifact.get("size_bytes")

    with requests.get(artifact["url"], stream=True, timeout=60) as response:
        response.raise_for_status()
        downloaded_bytes = 0
        with temporary_path.open("wb") as file_handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                file_handle.write(chunk)
                downloaded_bytes += len(chunk)

    if expected_size is not None and downloaded_bytes != int(expected_size):
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded size mismatch for {artifact['name']}: expected {expected_size}, got {downloaded_bytes}"
        )

    temporary_path.replace(output_path)
    return output_path


def extract_archive(archive_path: str | Path, output_dir: str | Path, overwrite: bool = False) -> Path:
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = archive_path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(output_dir)
        return output_dir

    raise ValueError(
        f"Extraction for '{archive_path.name}' is not implemented automatically. "
        "ZIP is supported; RAR archives should be extracted manually or with 7-Zip."
    )
