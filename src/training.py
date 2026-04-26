from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, recall_score, root_mean_squared_error


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(preferred)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    task: str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        if task == "regression":
            outputs = outputs.squeeze(1)
            targets = targets.float()

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return {"loss": total_loss / max(total_examples, 1)}


@torch.inference_mode()
def evaluate_classifier(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)
        predictions = outputs.argmax(dim=1)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": accuracy_score(all_targets, all_predictions),
        "macro_f1": f1_score(all_targets, all_predictions, average="macro"),
        "per_class_recall": recall_score(
            all_targets,
            all_predictions,
            average=None,
            labels=list(range(len(class_names))),
        ).tolist(),
        "confusion_matrix": confusion_matrix(
            all_targets,
            all_predictions,
            labels=list(range(len(class_names))),
        ).tolist(),
    }


@torch.inference_mode()
def evaluate_regressor(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_targets: list[float] = []
    all_predictions: list[float] = []

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device).float()

        outputs = model(images).squeeze(1)
        loss = criterion(outputs, targets)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(outputs.cpu().tolist())

    return {
        "loss": total_loss / max(total_examples, 1),
        "mae": mean_absolute_error(all_targets, all_predictions),
        "rmse": root_mean_squared_error(all_targets, all_predictions),
        "within_1_day_accuracy": float(
            np.mean(np.abs(np.asarray(all_predictions) - np.asarray(all_targets)) <= 1.0)
        ),
        "within_2_day_accuracy": float(
            np.mean(np.abs(np.asarray(all_predictions) - np.asarray(all_targets)) <= 2.0)
        ),
    }


def save_checkpoint(
    model: torch.nn.Module,
    output_path: str | Path,
    **metadata: Any,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), **metadata}
    torch.save(payload, output_path)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}
