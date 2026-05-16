"""Display upscaling helpers for extracted WorldCam images."""

from __future__ import annotations

import cv2
import numpy as np


def upscale_for_display(
    image: np.ndarray,
    min_width: int,
    min_height: int,
    max_scale: float = 4.0,
    interpolation: int = cv2.INTER_LANCZOS4,
) -> np.ndarray:
    """Upscale an image for display without requiring an external AI model."""
    image_height, image_width = image.shape[:2]
    if image_width <= 0 or image_height <= 0:
        return image

    scale = max(
        1.0,
        min(
            max_scale,
            min_width / image_width,
            min_height / image_height,
        ),
    )
    if scale <= 1.0:
        return image

    target_width = int(image_width * scale)
    target_height = int(image_height * scale)
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)
