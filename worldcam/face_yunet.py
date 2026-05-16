"""Specialized YuNet face detector helpers for WorldCam."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from worldcam.config import FACE_DETECTION_MODEL_ENGINE, FACE_DETECTION_MODEL_ONNX

FaceBox = tuple[int, int, int, int]

_YUNET_DETECTOR = None
_YUNET_INPUT_SIZE: tuple[int, int] | None = None
_YUNET_WARNING_PRINTED = False
_ENGINE_WARNING_PRINTED = False


def warn_engine_is_documented_only() -> None:
    """Explain once why the optional YuNet TensorRT engine is not used by this OpenCV path."""
    global _ENGINE_WARNING_PRINTED
    engine_path = Path(FACE_DETECTION_MODEL_ENGINE)
    if _ENGINE_WARNING_PRINTED or not engine_path.exists():
        return

    _ENGINE_WARNING_PRINTED = True
    print(
        "YuNet TensorRT engine detecte, mais le scanner visage utilise l'ONNX avec OpenCV FaceDetectorYN. "
        "L'engine demanderait un backend TensorRT dedie avec post-traitement YuNet."
    )


def get_yunet_model_path() -> Path | None:
    """Return the local YuNet ONNX model path when available."""
    global _YUNET_WARNING_PRINTED
    warn_engine_is_documented_only()

    model_path = Path(FACE_DETECTION_MODEL_ONNX)
    if model_path.exists():
        return model_path

    if not _YUNET_WARNING_PRINTED:
        _YUNET_WARNING_PRINTED = True
        print(f"YuNet indisponible: modele ONNX introuvable ({model_path}).")
    return None


def create_yunet_detector(input_size: tuple[int, int]):
    """Create or refresh the OpenCV YuNet detector for the requested input size."""
    global _YUNET_DETECTOR, _YUNET_INPUT_SIZE, _YUNET_WARNING_PRINTED

    if not hasattr(cv2, "FaceDetectorYN_create"):
        if not _YUNET_WARNING_PRINTED:
            _YUNET_WARNING_PRINTED = True
            print("YuNet indisponible: cette version OpenCV ne fournit pas cv2.FaceDetectorYN_create.")
        return None

    model_path = get_yunet_model_path()
    if model_path is None:
        return None

    if _YUNET_DETECTOR is not None and _YUNET_INPUT_SIZE == input_size:
        return _YUNET_DETECTOR

    try:
        _YUNET_DETECTOR = cv2.FaceDetectorYN_create(
            str(model_path),
            "",
            input_size,
            0.60,
            0.30,
            5000,
        )
        _YUNET_INPUT_SIZE = input_size
        return _YUNET_DETECTOR
    except Exception as exc:
        print(f"YuNet indisponible: creation du detecteur impossible: {exc}")
        _YUNET_DETECTOR = None
        _YUNET_INPUT_SIZE = None
        return None


def detect_faces_yunet(image: np.ndarray) -> list[FaceBox]:
    """Detect faces with the specialized YuNet ONNX model."""
    image_height, image_width = image.shape[:2]
    if image_width <= 0 or image_height <= 0:
        return []

    detector = create_yunet_detector((image_width, image_height))
    if detector is None:
        return []

    try:
        _status, detections = detector.detect(image)
    except Exception as exc:
        print(f"Erreur detection visage YuNet: {exc}")
        return []

    if detections is None:
        return []

    faces: list[FaceBox] = []
    for detection in detections:
        x, y, w, h = detection[:4]
        x1 = max(0, min(image_width - 1, int(round(x))))
        y1 = max(0, min(image_height - 1, int(round(y))))
        x2 = max(x1 + 1, min(image_width, int(round(x + w))))
        y2 = max(y1 + 1, min(image_height, int(round(y + h))))
        faces.append((x1, y1, x2 - x1, y2 - y1))

    return faces
