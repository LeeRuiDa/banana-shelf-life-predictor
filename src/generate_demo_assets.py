from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps

from .config import DEFAULT_CLASSIFIER_CHECKPOINT, DEFAULT_REGRESSOR_CHECKPOINT, PROJECT_ROOT, REPORTS_DIR
from .explain import explain_prediction
from .predict import load_model_bundle, predict_combined


DEFAULT_DAY0_DAY7_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "downloads"
    / "mendeley_banana_day0_day7_2026"
    / "extracted"
    / "banana_ripening_dataset_day0_to_day7"
    / "banana_ripening_dataset_day0_to_day7"
)


@dataclass(frozen=True)
class DemoSample:
    key: str
    title: str
    relative_path: Path
    true_days_to_rotten: int
    interpretation: str


DEFAULT_DEMO_SAMPLES = (
    DemoSample(
        key="green_unripe",
        title="Green / Unripe",
        relative_path=Path("Banana_ID_005") / "Day_0.jpg",
        true_days_to_rotten=7,
        interpretation="Early-stage banana; expected to have the most remaining shelf life.",
    ),
    DemoSample(
        key="ripe",
        title="Ripe",
        relative_path=Path("Banana_ID_036") / "Day_2.png",
        true_days_to_rotten=5,
        interpretation="Yellow banana with visible speckling; selected because the classifier predicts the ripe class.",
    ),
    DemoSample(
        key="overripe",
        title="Overripe",
        relative_path=Path("Banana_ID_005") / "Day_5.jpg",
        true_days_to_rotten=2,
        interpretation="Late-stage banana where darkening and bruise-like regions should matter.",
    ),
    DemoSample(
        key="near_rotten",
        title="Near Rotten",
        relative_path=Path("Banana_ID_005") / "Day_7.jpg",
        true_days_to_rotten=0,
        interpretation="Endpoint label for this dataset: zero days to rotten.",
    ),
)


def _fit_square(image: Image.Image, size: int = 320) -> Image.Image:
    return ImageOps.contain(image.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)


def _draw_caption(image: Image.Image, caption: str, height: int = 54) -> Image.Image:
    output = Image.new("RGB", (image.width, image.height + height), color=(255, 255, 255))
    output.paste(image, (0, 0))
    draw = ImageDraw.Draw(output)
    draw.multiline_text((8, image.height + 8), caption, fill=(20, 20, 20), spacing=4)
    return output


def _save_panel(
    sample: DemoSample,
    image: Image.Image,
    classifier_overlay: Image.Image,
    regressor_overlay: Image.Image,
    predictions: dict[str, Any],
    output_path: Path,
) -> None:
    stage = predictions.get("stage", {})
    days_left = predictions.get("days_left", {})
    title = (
        f"{sample.title} | target {sample.true_days_to_rotten} days | "
        f"predicted {days_left.get('predicted_days_to_rotten', 0.0):.2f} days"
    )
    subtitle = (
        f"Classifier: {stage.get('predicted_stage', 'n/a')} "
        f"({stage.get('confidence', 0.0):.1%})"
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for axis in axes:
        axis.axis("off")

    axes[0].imshow(image.convert("RGB"))
    axes[0].set_title("Original")
    axes[1].imshow(classifier_overlay)
    axes[1].set_title("Ripeness heatmap")
    axes[2].imshow(regressor_overlay)
    axes[2].set_title("Days-left heatmap")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.02, f"{subtitle}. {sample.interpretation}", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_grid(panel_paths: list[Path], output_path: Path) -> None:
    panels = [Image.open(path).convert("RGB") for path in panel_paths]
    target_width = max(panel.width for panel in panels)
    resized = []
    for panel in panels:
        if panel.width == target_width:
            resized.append(panel)
            continue
        scale = target_width / panel.width
        resized.append(panel.resize((target_width, int(panel.height * scale)), Image.Resampling.LANCZOS))

    gap = 18
    total_height = sum(panel.height for panel in resized) + gap * (len(resized) - 1)
    grid = Image.new("RGB", (target_width, total_height), color=(245, 242, 235))
    y_offset = 0
    for panel in resized:
        grid.paste(panel, (0, y_offset))
        y_offset += panel.height + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def generate_demo_assets(
    dataset_root: Path,
    output_dir: Path,
    classifier_checkpoint: Path,
    regressor_checkpoint: Path,
    device: str,
    attribution_steps: int,
) -> dict[str, Any]:
    classifier_bundle = load_model_bundle(classifier_checkpoint, device)
    regressor_bundle = load_model_bundle(regressor_checkpoint, device)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"samples": []}
    panel_paths: list[Path] = []
    for sample in DEFAULT_DEMO_SAMPLES:
        image_path = dataset_root / sample.relative_path
        image = Image.open(image_path).convert("RGB")
        predictions = predict_combined(image, classifier_bundle, regressor_bundle)
        classifier_explanation = explain_prediction(classifier_bundle, image, n_steps=attribution_steps)
        regressor_explanation = explain_prediction(regressor_bundle, image, n_steps=attribution_steps)

        original_output = output_dir / f"{sample.key}_original.jpg"
        classifier_output = output_dir / f"{sample.key}_classifier_heatmap.jpg"
        regressor_output = output_dir / f"{sample.key}_regressor_heatmap.jpg"
        panel_output = output_dir / f"{sample.key}_panel.png"

        _fit_square(image).save(original_output, quality=95)
        _draw_caption(
            _fit_square(classifier_explanation["overlay"]),
            "Ripeness attribution",
        ).save(classifier_output, quality=95)
        _draw_caption(
            _fit_square(regressor_explanation["overlay"]),
            "Days-left attribution",
        ).save(regressor_output, quality=95)
        _save_panel(
            sample=sample,
            image=image,
            classifier_overlay=classifier_explanation["overlay"],
            regressor_overlay=regressor_explanation["overlay"],
            predictions=predictions,
            output_path=panel_output,
        )
        panel_paths.append(panel_output)

        manifest["samples"].append(
            {
                "key": sample.key,
                "title": sample.title,
                "source_image": str(image_path),
                "true_days_to_rotten": sample.true_days_to_rotten,
                "predictions": predictions,
                "outputs": {
                    "original": str(original_output),
                    "classifier_heatmap": str(classifier_output),
                    "regressor_heatmap": str(regressor_output),
                    "panel": str(panel_output),
                },
            },
        )

    grid_output = REPORTS_DIR / "figures" / "demo_explainability_grid.png"
    _save_grid(panel_paths, grid_output)
    manifest["grid"] = str(grid_output)
    manifest_path = output_dir / "demo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate README-ready demo panels with model explanations.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DAY0_DAY7_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR / "demo")
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_CLASSIFIER_CHECKPOINT)
    parser.add_argument("--regressor-checkpoint", type=Path, default=DEFAULT_REGRESSOR_CHECKPOINT)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--attribution-steps", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = generate_demo_assets(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        classifier_checkpoint=args.classifier_checkpoint,
        regressor_checkpoint=args.regressor_checkpoint,
        device=args.device,
        attribution_steps=args.attribution_steps,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
