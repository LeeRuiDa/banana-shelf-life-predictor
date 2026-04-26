from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DATA_DIR, EXTERNAL_METADATA_DIR
from .data_sources import (
    download_artifact,
    extract_archive,
    fetch_source_manifest,
    list_known_sources,
    save_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch verified banana dataset manifests and downloads.")
    parser.add_argument("--list-sources", action="store_true", help="Print the known dataset sources and exit.")
    parser.add_argument("--source", type=str, help="Known source identifier.")
    parser.add_argument("--manifest-out", type=Path, default=None, help="Optional JSON path for the fetched manifest.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "raw" / "downloads",
        help="Directory where downloaded artifacts should be stored.",
    )
    parser.add_argument(
        "--artifact-name",
        action="append",
        default=[],
        help="Specific artifact name to download. Repeat to select multiple artifacts.",
    )
    parser.add_argument("--download", action="store_true", help="Download the selected artifact(s).")
    parser.add_argument("--extract", action="store_true", help="Extract downloaded ZIP archives.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_sources:
        print(json.dumps(list_known_sources(), indent=2))
        return 0

    if not args.source:
        raise SystemExit("--source is required unless --list-sources is used.")

    manifest = fetch_source_manifest(args.source)
    if args.manifest_out is not None:
        manifest_path = save_manifest(manifest, args.manifest_out)
        print(json.dumps({"manifest": str(manifest_path)}, indent=2))
    else:
        default_manifest_path = EXTERNAL_METADATA_DIR / f"{args.source}_manifest.json"
        manifest_path = save_manifest(manifest, default_manifest_path)
        print(json.dumps({"manifest": str(manifest_path)}, indent=2))

    print(json.dumps(manifest, indent=2))

    if not args.download:
        return 0

    artifacts = manifest.get("artifacts", [])
    if not artifacts:
        raise SystemExit(f"Source '{args.source}' has no directly downloadable artifacts in the registry.")

    selected_names = set(args.artifact_name)
    if not selected_names:
        selected_names = set(manifest.get("default_artifact_names") or [artifact["name"] for artifact in artifacts])

    selected_artifacts = [artifact for artifact in artifacts if artifact["name"] in selected_names]
    if not selected_artifacts:
        raise SystemExit(f"No artifacts matched {sorted(selected_names)} for source '{args.source}'.")

    download_root = args.output_dir / args.source
    results: list[dict[str, str]] = []
    for artifact in selected_artifacts:
        local_path = download_artifact(
            artifact=artifact,
            output_path=download_root / artifact["filename"],
            overwrite=args.overwrite,
        )
        result = {"artifact": artifact["name"], "downloaded_to": str(local_path)}

        if args.extract:
            extraction_root = download_root / "extracted" / local_path.stem
            extracted_path = extract_archive(local_path, extraction_root, overwrite=args.overwrite)
            result["extracted_to"] = str(extracted_path)

        results.append(result)

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

