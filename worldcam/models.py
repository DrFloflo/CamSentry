"""YOLO model loading helpers."""

import torch
from ultralytics import YOLO

from worldcam.compat import patch_ultralytics_pose26
from worldcam.config import MODEL_ENGINE, MODEL_PT, POSE_MODEL_ENGINE, POSE_MODEL_PT


def load_yolo_model() -> tuple[YOLO, str]:
    """Load YOLO26L, preferring a TensorRT engine when available."""
    patch_ultralytics_pose26()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Périphérique YOLO utilisé: {device}")

    try:
        print(f"Chargement du modèle TensorRT: {MODEL_ENGINE}")
        model = YOLO(MODEL_ENGINE)
    except Exception as exc:
        print(f"TensorRT indisponible ({exc}); fallback vers le modèle PyTorch: {MODEL_PT}")
        model = YOLO(MODEL_PT)
        if device == "cuda":
            model.half()

    return model, device


def load_pose_model(device: str) -> YOLO:
    """Load the YOLO pose model, preferring a TensorRT engine when available."""
    patch_ultralytics_pose26()

    try:
        print(f"Chargement du modèle pose TensorRT: {POSE_MODEL_ENGINE}")
        pose_model = YOLO(POSE_MODEL_ENGINE)
    except Exception as exc:
        print(f"TensorRT pose indisponible ({exc}); fallback vers le modèle PyTorch: {POSE_MODEL_PT}")
        pose_model = YOLO(POSE_MODEL_PT)
        if device == "cuda":
            pose_model.half()

    return pose_model
