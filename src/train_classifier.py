from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import DEFAULT_CLASSIFIER_CHECKPOINT, METRICS_DIR, sort_stage_labels
from .datasets import BananaImageDataset, filter_metadata_for_target, load_metadata, split_metadata_by_group
from .models import build_classifier
from .training import evaluate_classifier, get_device, load_checkpoint, save_checkpoint, set_seed, train_one_epoch
from .transforms import build_eval_transforms, build_train_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the banana ripeness classifier.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to the metadata CSV.")
    parser.add_argument("--images-root", type=Path, default=None, help="Root directory for image paths.")
    parser.add_argument("--output", type=Path, default=DEFAULT_CLASSIFIER_CHECKPOINT, help="Checkpoint output path.")
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=METRICS_DIR / "classifier_metrics.json",
        help="Path for the training summary JSON.",
    )
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", choices=("efficientnet_b0", "efficientnet_b1", "resnet50"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet initialization.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    metadata = filter_metadata_for_target(load_metadata(args.metadata), "ripeness_stage")
    train_size = 1.0 - args.val_size - args.test_size
    train_df, val_df, test_df = split_metadata_by_group(
        metadata,
        train_size=train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    class_names = sort_stage_labels(train_df["ripeness_stage"].unique().tolist())
    train_dataset = BananaImageDataset(
        metadata=train_df,
        images_root=args.images_root,
        transform=build_train_transforms(args.image_size),
        target_column="ripeness_stage",
        class_names=class_names,
    )
    val_dataset = BananaImageDataset(
        metadata=val_df,
        images_root=args.images_root,
        transform=build_eval_transforms(args.image_size),
        target_column="ripeness_stage",
        class_names=class_names,
    )
    test_dataset = BananaImageDataset(
        metadata=test_df,
        images_root=args.images_root,
        transform=build_eval_transforms(args.image_size),
        target_column="ripeness_stage",
        class_names=class_names,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = get_device(args.device)
    model = build_classifier(
        backbone=args.backbone,
        num_classes=len(class_names),
        pretrained=not args.no_pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_macro_f1 = float("-inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, task="classification")
        val_metrics = evaluate_classifier(model, val_loader, criterion, device, class_names)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, indent=2))

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                model,
                args.output,
                task="classification",
                backbone=args.backbone,
                class_names=class_names,
                image_size=args.image_size,
                best_val_macro_f1=best_macro_f1,
            )

    load_checkpoint(model, args.output, device)
    test_metrics = evaluate_classifier(model, test_loader, criterion, device, class_names)
    summary = {
        "task": "classification",
        "backbone": args.backbone,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "class_names": class_names,
        "history": history,
        "test_metrics": test_metrics,
    }

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output), "metrics": str(args.metrics_output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
