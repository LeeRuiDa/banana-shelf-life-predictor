from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    efficientnet_b1,
    resnet50,
)


def build_classifier(
    backbone: str = "efficientnet_b0",
    num_classes: int = 4,
    pretrained: bool = True,
) -> nn.Module:
    model = _build_backbone(backbone=backbone, pretrained=pretrained)

    if backbone.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    elif backbone == "resnet50":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return model


def build_regressor(
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
) -> nn.Module:
    model = _build_backbone(backbone=backbone, pretrained=pretrained)

    if backbone.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 1),
        )
    elif backbone == "resnet50":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return model


def _build_backbone(backbone: str, pretrained: bool) -> nn.Module:
    if backbone == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        return efficientnet_b0(weights=weights)
    if backbone == "efficientnet_b1":
        weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
        return efficientnet_b1(weights=weights)
    if backbone == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        return resnet50(weights=weights)
    raise ValueError(f"Unsupported backbone: {backbone}")

