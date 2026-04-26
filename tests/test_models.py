from __future__ import annotations

import torch

from src.models import build_classifier, build_regressor


def test_classifier_output_shape() -> None:
    model = build_classifier(backbone="efficientnet_b0", num_classes=4, pretrained=False)
    outputs = model(torch.randn(2, 3, 224, 224))
    assert tuple(outputs.shape) == (2, 4)


def test_regressor_output_shape() -> None:
    model = build_regressor(backbone="efficientnet_b0", pretrained=False)
    outputs = model(torch.randn(2, 3, 224, 224))
    assert tuple(outputs.shape) == (2, 1)

