from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader

from .datasets import BananaImageDataset, filter_metadata_for_target, load_metadata
from .predict import load_model_bundle
from .training import evaluate_classifier, evaluate_regressor
from .transforms import build_eval_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on a metadata file.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to the metadata CSV.")
    parser.add_argument("--images-root", type=Path, default=None, help="Root directory for image paths.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    bundle = load_model_bundle(args.checkpoint, args.device)

    target_column = "ripeness_stage" if bundle.task == "classification" else "days_to_rotten"
    metadata = filter_metadata_for_target(metadata, target_column)
    dataset = BananaImageDataset(
        metadata=metadata,
        images_root=args.images_root,
        transform=build_eval_transforms(bundle.image_size),
        target_column=target_column,
        class_names=bundle.class_names or None,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if bundle.task == "classification":
        metrics = evaluate_classifier(bundle.model, dataloader, nn.CrossEntropyLoss(), bundle.device, bundle.class_names)
    else:
        metrics = evaluate_regressor(bundle.model, dataloader, nn.L1Loss(), bundle.device)

    print(json.dumps(metrics, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
