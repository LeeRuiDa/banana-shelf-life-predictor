from __future__ import annotations

from typing import Any

import matplotlib.cm as cm
import numpy as np
from captum.attr import IntegratedGradients
from PIL import Image

from .predict import ModelBundle, predict_image
from .transforms import build_eval_transforms


def compute_integrated_gradients(
    bundle: ModelBundle,
    image: Image.Image,
    target: int | None = None,
    n_steps: int = 32,
) -> np.ndarray:
    model_input = build_eval_transforms(bundle.image_size)(image.convert("RGB")).unsqueeze(0).to(bundle.device)
    explainer = IntegratedGradients(bundle.model)

    if bundle.task == "classification" and target is None:
        target = predict_image(image, bundle)["predicted_class_index"]
    elif bundle.task == "regression" and target is None:
        target = 0

    attributions = explainer.attribute(model_input, target=target, n_steps=n_steps)
    attribution_map = attributions.squeeze(0).detach().abs().sum(dim=0).cpu().numpy()
    attribution_map -= attribution_map.min()
    attribution_map /= attribution_map.max() + 1e-8
    return attribution_map


def overlay_attribution(image: Image.Image, attribution_map: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heatmap = (cm.inferno(attribution_map)[..., :3] * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap).resize(image.size)
    return Image.blend(image.convert("RGB"), heatmap_image, alpha=alpha)


def explain_prediction(bundle: ModelBundle, image: Image.Image, n_steps: int = 32) -> dict[str, Any]:
    prediction = predict_image(image, bundle)
    target = prediction.get("predicted_class_index") if bundle.task == "classification" else None
    attribution_map = compute_integrated_gradients(bundle, image, target=target, n_steps=n_steps)
    overlay = overlay_attribution(image, attribution_map)
    return {
        "prediction": prediction,
        "attribution_map": attribution_map,
        "overlay": overlay,
        "method": "Integrated Gradients",
    }


def summarize_explanation(prediction: dict[str, Any]) -> str:
    if prediction["task"] == "classification":
        return (
            f"{prediction['predicted_stage']} "
            f"({prediction['confidence']:.1%} classifier confidence)"
        )
    return f"{prediction['predicted_days_to_rotten']:.2f} predicted days to rotten"


def build_prediction_interpretation(predictions: dict[str, Any]) -> str:
    stage_prediction = predictions.get("stage")
    days_prediction = predictions.get("days_left")

    stage_name = stage_prediction.get("predicted_stage") if stage_prediction else None
    predicted_days = days_prediction.get("predicted_days_to_rotten") if days_prediction else None

    if stage_name and predicted_days is not None:
        stage_phrases = {
            "unripe": "Mostly early-stage banana",
            "semi_ripe": "Transition-stage banana",
            "ripe": "Ready-to-eat banana",
            "overripe": "Late-stage banana",
            "rotten": "End-stage banana",
        }
        days_value = float(predicted_days)
        if days_value >= 5:
            shelf_life_phrase = "with high remaining shelf life."
        elif days_value >= 3:
            shelf_life_phrase = "with several days remaining before the rotten stage."
        elif days_value >= 1:
            shelf_life_phrase = "that should be eaten soon."
        elif days_value >= 0.25:
            shelf_life_phrase = "with very little shelf life remaining."
        else:
            shelf_life_phrase = "that is effectively at the rotten endpoint."
        return f"{stage_phrases.get(stage_name, stage_name.replace('_', ' '))} {shelf_life_phrase}"

    if stage_name:
        return f"Classifier sees a {stage_name.replace('_', ' ')} banana."

    if predicted_days is not None:
        days_value = float(predicted_days)
        if days_value >= 5:
            return "Regressor estimates a long remaining shelf life."
        if days_value >= 3:
            return "Regressor estimates a mid-window banana with several days left."
        if days_value >= 1:
            return "Regressor estimates a short remaining shelf life."
        return "Regressor estimates the banana is near the rotten stage."

    return "No prediction available."
