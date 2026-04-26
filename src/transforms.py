from __future__ import annotations

import torch
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transforms(image_size: int = 224) -> v2.Compose:
    return v2.Compose(
        [
            v2.RandomResizedCrop((image_size, image_size), scale=(0.85, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=10),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
    )


def build_eval_transforms(image_size: int = 224) -> v2.Compose:
    resize_size = int(image_size * 1.15)
    return v2.Compose(
        [
            v2.Resize((resize_size, resize_size)),
            v2.CenterCrop((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
    )


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=image_tensor.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image_tensor.device).view(-1, 1, 1)
    return (image_tensor * std + mean).clamp(0.0, 1.0)

