from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explain import build_prediction_interpretation, explain_prediction, summarize_explanation
from src.predict import load_model_bundle, predict_combined


DEFAULT_CLASSIFIER_PATH = os.getenv("BANANA_CLASSIFIER_CHECKPOINT", str(PROJECT_ROOT / "models" / "classifier_best.pt"))
DEFAULT_REGRESSOR_PATH = os.getenv("BANANA_REGRESSOR_CHECKPOINT", str(PROJECT_ROOT / "models" / "regressor_best.pt"))
DEFAULT_DEMO_MANIFEST_PATH = PROJECT_ROOT / "reports" / "demo" / "demo_manifest.json"


@st.cache_resource(show_spinner=False)
def cached_bundle(checkpoint_path: str):
    path = Path(checkpoint_path)
    if not checkpoint_path or not path.exists():
        return None
    return load_model_bundle(path)


@st.cache_data(show_spinner=False)
def cached_demo_manifest(manifest_path: str) -> list[dict]:
    path = Path(manifest_path)
    if not manifest_path or not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    samples: list[dict] = []
    for sample in payload.get("samples", []):
        source_image = sample.get("source_image")
        if not source_image or not Path(source_image).exists():
            continue
        samples.append(sample)
    return samples


def render_prediction_card(name: str, prediction: dict) -> None:
    st.subheader(name)
    if prediction["task"] == "classification":
        st.metric("Ripeness stage", prediction["predicted_stage"])
        st.metric("Confidence", f"{prediction['confidence']:.1%}")
        if prediction["probabilities"]:
            st.bar_chart(prediction["probabilities"])
    else:
        st.metric("Estimated days left", f"{prediction['predicted_days_to_rotten']:.2f}")


def main() -> None:
    st.set_page_config(page_title="Days to Banana Death", layout="wide")
    st.title("Days to Banana Death")
    st.caption("Single-image banana ripeness and shelf-life estimation.")
    st.info("Best used with single banana, front-lit images and trained checkpoints in the models directory.")

    with st.sidebar:
        st.header("Checkpoints")
        classifier_path = st.text_input("Classifier checkpoint", value=DEFAULT_CLASSIFIER_PATH)
        regressor_path = st.text_input("Regressor checkpoint", value=DEFAULT_REGRESSOR_PATH)
        demo_samples = cached_demo_manifest(str(DEFAULT_DEMO_MANIFEST_PATH))
        input_source = "Upload images"
        selected_demo_sample = None
        if demo_samples:
            st.header("Input")
            input_source = st.radio("Image source", options=["Upload images", "Demo gallery"], index=0)
            if input_source == "Demo gallery":
                demo_options = {
                    f"{sample['title']} ({Path(sample['source_image']).name})": sample
                    for sample in demo_samples
                }
                selected_demo_sample = demo_options[
                    st.selectbox("Curated demo", options=list(demo_options.keys()))
                ]
        show_xai = st.checkbox("Show attribution overlay", value=True)
        attribution_steps = st.slider("Attribution steps", min_value=8, max_value=64, value=24, step=8)

    classifier_bundle = cached_bundle(classifier_path)
    regressor_bundle = cached_bundle(regressor_path)

    if classifier_bundle is None and regressor_bundle is None:
        st.warning("No trained checkpoints were found yet. Train a model first or update the checkpoint paths in the sidebar.")

    inputs: list[tuple[str, Image.Image, dict | None]] = []
    if input_source == "Demo gallery" and selected_demo_sample is not None:
        demo_image = Image.open(selected_demo_sample["source_image"]).convert("RGB")
        inputs.append((selected_demo_sample["title"], demo_image, selected_demo_sample))
    else:
        uploads = st.file_uploader(
            "Upload one or more banana images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        if uploads:
            inputs.extend(
                (upload.name, Image.open(upload).convert("RGB"), None)
                for upload in uploads
            )

    if not inputs:
        st.stop()

    for display_name, image, demo_sample in inputs:
        predictions = predict_combined(image, classifier_bundle, regressor_bundle)

        st.markdown(f"## {display_name}")
        left, right = st.columns([1.1, 1.0])

        with left:
            st.image(image, caption="Input image", use_container_width=True)

        with right:
            if "stage" in predictions:
                render_prediction_card("Ripeness", predictions["stage"])
            if "days_left" in predictions:
                render_prediction_card("Shelf-life", predictions["days_left"])
            st.markdown("**Interpretation**")
            st.write(build_prediction_interpretation(predictions))
            if demo_sample is not None and demo_sample.get("true_days_to_rotten") is not None:
                st.caption(f"Curated demo reference: {demo_sample['true_days_to_rotten']} target days to rotten.")

        if show_xai and (classifier_bundle is not None or regressor_bundle is not None):
            st.subheader("Explanation")
            explainers = []
            if classifier_bundle is not None:
                explainers.append(("Ripeness regions", classifier_bundle))
            if regressor_bundle is not None:
                explainers.append(("Days-left regions", regressor_bundle))

            explanation_columns = st.columns(len(explainers))
            for column, (label, bundle) in zip(explanation_columns, explainers):
                with column:
                    explanation = explain_prediction(bundle, image, n_steps=attribution_steps)
                    st.image(
                        explanation["overlay"],
                        caption=f"{label}: {summarize_explanation(explanation['prediction'])}",
                        use_container_width=True,
                    )

            st.caption("Heatmaps are Integrated Gradients overlays; brighter regions have larger attribution for the displayed output.")

        st.divider()


if __name__ == "__main__":
    main()
