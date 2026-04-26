from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .config import DEFAULT_CLASSIFIER_CHECKPOINT, DEFAULT_REGRESSOR_CHECKPOINT
from .models import build_classifier, build_regressor
from .training import get_device
from .transforms import build_eval_transforms


@dataclass
class ModelBundle:
    model: torch.nn.Module
    task: str
    backbone: str
    image_size: int
    device: torch.device
    class_names: list[str]
    checkpoint_path: Path


def load_model_bundle(checkpoint_path: str | Path, device: str = "auto") -> ModelBundle:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    resolved_device = get_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary payload with model metadata.")

    task = checkpoint.get("task")
    backbone = checkpoint.get("backbone", "efficientnet_b0")
    image_size = int(checkpoint.get("image_size", 224))
    class_names = checkpoint.get("class_names", [])

    if task == "classification":
        if not class_names:
            raise ValueError("Classification checkpoints must include class_names metadata.")
        model = build_classifier(backbone=backbone, num_classes=len(class_names), pretrained=False)
    elif task == "regression":
        model = build_regressor(backbone=backbone, pretrained=False)
    else:
        raise ValueError(f"Unsupported checkpoint task: {task}")

    model.load_state_dict(checkpoint["state_dict"])
    model.to(resolved_device)
    model.eval()

    return ModelBundle(
        model=model,
        task=task,
        backbone=backbone,
        image_size=image_size,
        device=resolved_device,
        class_names=list(class_names),
        checkpoint_path=checkpoint_path,
    )


def predict_image(image: Image.Image, bundle: ModelBundle) -> dict[str, Any]:
    processed = build_eval_transforms(bundle.image_size)(image.convert("RGB")).unsqueeze(0).to(bundle.device)

    with torch.inference_mode():
        outputs = bundle.model(processed)

    if bundle.task == "classification":
        probabilities = torch.softmax(outputs, dim=1)[0]
        index = int(probabilities.argmax().item())
        stage_name = bundle.class_names[index] if bundle.class_names else str(index)
        return {
            "task": "classification",
            "predicted_class_index": index,
            "predicted_stage": stage_name,
            "confidence": float(probabilities[index].item()),
            "probabilities": {
                class_name: float(probabilities[class_index].item())
                for class_index, class_name in enumerate(bundle.class_names)
            },
        }

    predicted_days = min(max(float(outputs.squeeze().item()), 0.0), 7.0)
    return {
        "task": "regression",
        "predicted_days_to_rotten": predicted_days,
    }


def predict_combined(
    image: Image.Image,
    classifier_bundle: ModelBundle | None = None,
    regressor_bundle: ModelBundle | None = None,
) -> dict[str, Any]:
    predictions: dict[str, Any] = {}
    if classifier_bundle is not None:
        predictions["stage"] = predict_image(image, classifier_bundle)
    if regressor_bundle is not None:
        predictions["days_left"] = predict_image(image, regressor_bundle)
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one banana image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to an input image.")
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_CLASSIFIER_CHECKPOINT)
    parser.add_argument("--regressor-checkpoint", type=Path, default=DEFAULT_REGRESSOR_CHECKPOINT)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = Image.open(args.image).convert("RGB")

    classifier_bundle = load_model_bundle(args.classifier_checkpoint, args.device) if args.classifier_checkpoint.exists() else None
    regressor_bundle = load_model_bundle(args.regressor_checkpoint, args.device) if args.regressor_checkpoint.exists() else None

    if classifier_bundle is None and regressor_bundle is None:
        raise FileNotFoundError("No classifier or regressor checkpoint was found.")

    predictions = predict_combined(image, classifier_bundle, regressor_bundle)
    print(json.dumps(predictions, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
