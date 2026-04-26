from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import DEFAULT_REGRESSOR_CHECKPOINT, METRICS_DIR, REPORTS_DIR, TARGET_DEFINITION
from .datasets import BananaImageDataset, filter_metadata_for_target, load_metadata, split_metadata_by_group
from .models import build_regressor
from .training import evaluate_regressor, get_device, load_checkpoint, save_checkpoint, set_seed, train_one_epoch
from .transforms import build_eval_transforms, build_train_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the banana shelf-life regressor.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to the metadata CSV.")
    parser.add_argument("--images-root", type=Path, default=None, help="Root directory for image paths.")
    parser.add_argument("--output", type=Path, default=DEFAULT_REGRESSOR_CHECKPOINT, help="Checkpoint output path.")
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=METRICS_DIR / "regressor_metrics.json",
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
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet initialization.")
    parser.add_argument(
        "--init-classifier-checkpoint",
        type=Path,
        default=None,
        help="Optional classifier checkpoint to use for backbone initialization.",
    )
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the regression head unless last blocks are unfrozen.")
    parser.add_argument("--unfreeze-last-blocks", type=int, default=0, help="With --freeze-backbone, unfreeze the last N EfficientNet feature blocks.")
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=METRICS_DIR / "regressor_test_predictions.csv",
        help="CSV output for held-out test predictions.",
    )
    parser.add_argument(
        "--plots-output-dir",
        type=Path,
        default=REPORTS_DIR / "figures",
        help="Directory for regression diagnostic plots.",
    )
    return parser.parse_args()


def load_matching_backbone_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model_state = model.state_dict()
    compatible_state = {
        name: value
        for name, value in state_dict.items()
        if name in model_state and tuple(model_state[name].shape) == tuple(value.shape)
    }
    skipped = len(state_dict) - len(compatible_state)
    model_state.update(compatible_state)
    model.load_state_dict(model_state)
    return {"loaded_tensors": len(compatible_state), "skipped_tensors": skipped}


def configure_trainable_layers(model: nn.Module, backbone: str, freeze_backbone: bool, unfreeze_last_blocks: int) -> int:
    if not freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    for parameter in model.parameters():
        parameter.requires_grad = False

    if backbone.startswith("efficientnet"):
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        if unfreeze_last_blocks > 0:
            for block in list(model.features.children())[-unfreeze_last_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
    elif backbone == "resnet50":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        if unfreeze_last_blocks > 0:
            for block in [model.layer4, model.layer3, model.layer2, model.layer1][:unfreeze_last_blocks]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@torch.inference_mode()
def collect_regression_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device).float()
        predictions = model(images).squeeze(1)

        for index, prediction in enumerate(predictions.cpu().tolist()):
            target = float(targets[index].cpu().item())
            rows.append(
                {
                    "image_path": batch["image_path"][index],
                    "banana_id": batch["banana_id"][index],
                    "day_index": int(batch["day_index"][index]),
                    "true_days_to_rotten": target,
                    "predicted_days_to_rotten": float(prediction),
                    "residual": float(prediction - target),
                },
            )
    return pd.DataFrame(rows)


def write_regression_plots(predictions: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scatter_path = output_dir / "regressor_predicted_vs_true.png"
    residual_path = output_dir / "regressor_residuals_by_true_day.png"

    plt.figure(figsize=(6.5, 6))
    plt.scatter(predictions["true_days_to_rotten"], predictions["predicted_days_to_rotten"], alpha=0.75)
    min_value = min(predictions["true_days_to_rotten"].min(), predictions["predicted_days_to_rotten"].min())
    max_value = max(predictions["true_days_to_rotten"].max(), predictions["predicted_days_to_rotten"].max())
    plt.plot([min_value, max_value], [min_value, max_value], color="black", linestyle="--", linewidth=1)
    plt.xlabel("True days to rotten")
    plt.ylabel("Predicted days to rotten")
    plt.title("Regressor Predicted vs True Days")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.scatter(predictions["true_days_to_rotten"], predictions["residual"], alpha=0.75)
    plt.xlabel("True days to rotten")
    plt.ylabel("Prediction residual")
    plt.title("Regressor Residuals by True Day")
    plt.tight_layout()
    plt.savefig(residual_path, dpi=160)
    plt.close()

    return {
        "predicted_vs_true_plot": str(scatter_path),
        "residual_plot": str(residual_path),
    }


def mean_baseline_metrics(train_df, test_df) -> dict[str, float]:
    mean_prediction = float(train_df["days_to_rotten"].mean())
    targets = test_df["days_to_rotten"].astype(float).to_numpy()
    predictions = np.full_like(targets, fill_value=mean_prediction, dtype=float)
    residuals = predictions - targets
    return {
        "strategy": "predict_train_mean_days_to_rotten",
        "prediction": mean_prediction,
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "within_1_day_accuracy": float(np.mean(np.abs(residuals) <= 1.0)),
        "within_2_day_accuracy": float(np.mean(np.abs(residuals) <= 2.0)),
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    metadata = filter_metadata_for_target(load_metadata(args.metadata), "days_to_rotten")
    train_size = 1.0 - args.val_size - args.test_size
    train_df, val_df, test_df = split_metadata_by_group(
        metadata,
        train_size=train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    train_dataset = BananaImageDataset(
        metadata=train_df,
        images_root=args.images_root,
        transform=build_train_transforms(args.image_size),
        target_column="days_to_rotten",
    )
    val_dataset = BananaImageDataset(
        metadata=val_df,
        images_root=args.images_root,
        transform=build_eval_transforms(args.image_size),
        target_column="days_to_rotten",
    )
    test_dataset = BananaImageDataset(
        metadata=test_df,
        images_root=args.images_root,
        transform=build_eval_transforms(args.image_size),
        target_column="days_to_rotten",
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = get_device(args.device)
    model = build_regressor(
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
    ).to(device)
    initialization: dict[str, object] = {"pretrained": not args.no_pretrained}
    if args.init_classifier_checkpoint is not None:
        initialization["classifier_checkpoint"] = str(args.init_classifier_checkpoint)
        initialization.update(load_matching_backbone_weights(model, args.init_classifier_checkpoint, device))

    trainable_parameters = configure_trainable_layers(
        model=model,
        backbone=args.backbone,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
    )

    criterion = nn.HuberLoss(delta=args.huber_delta)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_mae = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, task="regression")
        val_metrics = evaluate_regressor(model, val_loader, criterion, device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, indent=2))

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            save_checkpoint(
                model,
                args.output,
                task="regression",
                backbone=args.backbone,
                image_size=args.image_size,
                target_definition=TARGET_DEFINITION,
                best_val_mae=best_mae,
                initialization=initialization,
                freeze_backbone=args.freeze_backbone,
                unfreeze_last_blocks=args.unfreeze_last_blocks,
            )

    load_checkpoint(model, args.output, device)
    test_metrics = evaluate_regressor(model, test_loader, criterion, device)
    test_predictions = collect_regression_predictions(model, test_loader, device)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    test_predictions.to_csv(args.predictions_output, index=False)
    plot_outputs = write_regression_plots(test_predictions, args.plots_output_dir)
    summary = {
        "task": "regression",
        "backbone": args.backbone,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "target_definition": TARGET_DEFINITION,
        "initialization": initialization,
        "freeze_backbone": args.freeze_backbone,
        "unfreeze_last_blocks": args.unfreeze_last_blocks,
        "trainable_parameters": trainable_parameters,
        "split_counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "split_group_counts": {
            "train": int(train_df["split_group"].nunique()),
            "val": int(val_df["split_group"].nunique()),
            "test": int(test_df["split_group"].nunique()),
        },
        "split_banana_counts": {
            "train": int(train_df["banana_id"].nunique()),
            "val": int(val_df["banana_id"].nunique()),
            "test": int(test_df["banana_id"].nunique()),
        },
        "baselines": {"train_mean": mean_baseline_metrics(train_df, test_df)},
        "history": history,
        "test_metrics": test_metrics,
        "predictions_output": str(args.predictions_output),
        "plots": plot_outputs,
    }

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output), "metrics": str(args.metrics_output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
