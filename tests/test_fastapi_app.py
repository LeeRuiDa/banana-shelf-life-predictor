"""Unit and integration tests for the FastAPI application.

Validates the /health and /predict endpoints, image decoding error handling,
format support (JPEG/PNG), and checkpoint availability states.
"""

from __future__ import annotations

from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.fastapi_app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _make_test_image_bytes(format: str = "JPEG") -> bytes:
    """Create a minimal valid in-memory image for upload testing."""
    image = Image.new("RGB", (64, 64), color=(255, 255, 0))
    buffer = BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    def test_health_returns_status(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert "classifier_loaded" in payload
        assert "regressor_loaded" in payload
        assert isinstance(payload["classifier_loaded"], bool)
        assert isinstance(payload["regressor_loaded"], bool)


# ---------------------------------------------------------------------------
# Predict endpoint tests
# ---------------------------------------------------------------------------
class TestPredictEndpoint:
    def test_predict_with_valid_jpeg(self, client: TestClient) -> None:
        img_bytes = _make_test_image_bytes(format="JPEG")
        response = client.post(
            "/predict",
            files={"file": ("banana.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["filename"] == "banana.jpg"
        assert "predictions" in payload
        assert isinstance(payload["predictions"], dict)

    def test_predict_with_valid_png(self, client: TestClient) -> None:
        img_bytes = _make_test_image_bytes(format="PNG")
        response = client.post(
            "/predict",
            files={"file": ("sample.png", img_bytes, "image/png")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["filename"] == "sample.png"
        assert "predictions" in payload

    def test_predict_with_corrupt_file_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/predict",
            files={"file": ("corrupt.jpg", b"not-a-valid-image-stream", "image/jpeg")},
        )
        assert response.status_code == 400
        assert "not a valid image" in response.json()["detail"]

    def test_predict_with_empty_file_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"]

    @patch("app.fastapi_app.get_classifier_bundle", return_value=None)
    @patch("app.fastapi_app.get_regressor_bundle", return_value=None)
    def test_predict_without_checkpoints_returns_503(
        self, mock_reg: object, mock_cls: object, client: TestClient
    ) -> None:
        img_bytes = _make_test_image_bytes()
        response = client.post(
            "/predict",
            files={"file": ("banana.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 503
        assert "No trained checkpoints" in response.json()["detail"]
