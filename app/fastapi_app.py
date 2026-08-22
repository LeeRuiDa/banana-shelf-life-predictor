from __future__ import annotations

import os
import sys
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_model_bundle, predict_combined


DEFAULT_CLASSIFIER_PATH = Path(os.getenv("BANANA_CLASSIFIER_CHECKPOINT", PROJECT_ROOT / "models" / "classifier_best.pt"))
DEFAULT_REGRESSOR_PATH = Path(os.getenv("BANANA_REGRESSOR_CHECKPOINT", PROJECT_ROOT / "models" / "regressor_best.pt"))

app = FastAPI(title="Days to Banana Death API", version="0.1.0")


@lru_cache(maxsize=1)
def get_classifier_bundle():
    if DEFAULT_CLASSIFIER_PATH.exists():
        return load_model_bundle(DEFAULT_CLASSIFIER_PATH)
    return None


@lru_cache(maxsize=1)
def get_regressor_bundle():
    if DEFAULT_REGRESSOR_PATH.exists():
        return load_model_bundle(DEFAULT_REGRESSOR_PATH)
    return None


@app.get("/health")
def health() -> dict[str, bool]:
    return {
        "classifier_loaded": get_classifier_bundle() is not None,
        "regressor_loaded": get_regressor_bundle() is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    classifier_bundle = get_classifier_bundle()
    regressor_bundle = get_regressor_bundle()
    if classifier_bundle is None and regressor_bundle is None:
        raise HTTPException(status_code=503, detail="No trained checkpoints are available.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Uploaded file is not a valid image: {exc}") from exc

    predictions = predict_combined(image, classifier_bundle, regressor_bundle)
    return {"filename": file.filename, "predictions": predictions}
