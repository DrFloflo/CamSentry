"""YOLO model loading helpers."""

from dataclasses import dataclass
import importlib
import importlib.metadata
from pathlib import Path
import platform
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from worldcam.core.compat import patch_ultralytics_pose26
from worldcam.core.config import DEFAULT_MODEL_KEY, INFERENCE_WIDTH, MODEL_BACKEND_EXTENSIONS, MODEL_FILE_PREFIX, MODELS_DIR


@dataclass(frozen=True)
class InferenceInput:
    """Prepared resized frame and scaling metadata shared by per-frame YOLO analyses."""

    image: np.ndarray
    scale_x: float
    scale_y: float
    frame_width: int
    frame_height: int
    inference_width: int
    inference_height: int


@dataclass(frozen=True)
class ResizedInferenceResult:
    """YOLO inference output with metadata for original-frame scaling."""

    results: object
    scale_x: float
    scale_y: float
    frame_width: int
    frame_height: int
    inference_width: int
    inference_height: int


def installed_version(package_name: str) -> str:
    """Return an installed package version, or a readable missing marker."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def describe_tensorrt_runtime_error(exc: Exception) -> str:
    """Build a concise TensorRT diagnostic for engine loading failures."""
    diagnostic_lines = [
        f"{type(exc).__name__}: {exc}",
        f"Python: {sys.version.split()[0]} ({platform.platform()})",
        f"CUDA available: {torch.cuda.is_available()}",
        f"torch: {getattr(torch, '__version__', 'unknown')}",
        f"tensorrt: {installed_version('tensorrt')}",
        f"tensorrt-cu12: {installed_version('tensorrt-cu12')}",
        f"tensorrt_cu12_bindings: {installed_version('tensorrt_cu12_bindings')}",
        f"tensorrt_cu12_libs: {installed_version('tensorrt_cu12_libs')}",
        f"legacy tensorrt-bindings: {installed_version('tensorrt-bindings')}",
    ]

    if sys.version_info >= (3, 13):
        diagnostic_lines.append(
            "Hint: NVIDIA TensorRT wheels may be incomplete or unsupported on Python 3.13; "
            "use a Python 3.12 venv with matching TensorRT packages."
        )

    try:
        importlib.import_module("tensorrt")
    except Exception as import_exc:
        diagnostic_lines.append(f"import tensorrt failed: {type(import_exc).__name__}: {import_exc}")

    return "\n  ".join(diagnostic_lines)


def is_tensorrt_model(model: YOLO) -> bool:
    """Return whether this YOLO instance was loaded from a TensorRT engine."""
    return getattr(model, "_worldcam_backend_name", None) == "TensorRT"


def run_model_inference(model: YOLO, image: np.ndarray, device: str):
    """Run inference without forcing device selection for TensorRT engines."""
    if is_tensorrt_model(model):
        return model(image, verbose=False)[0]
    return model(image, verbose=False, device=device)[0]


def prepare_inference_input(frame: np.ndarray) -> InferenceInput:
    """Resize a frame once and return scaling metadata reusable across YOLO models."""
    frame_h, frame_w, _ = frame.shape
    new_width = min(INFERENCE_WIDTH, frame_w)
    new_height = int(frame_h * (new_width / frame_w))
    resized_frame = cv2.resize(frame, (new_width, new_height))
    return InferenceInput(
        image=resized_frame,
        scale_x=frame_w / new_width,
        scale_y=frame_h / new_height,
        frame_width=frame_w,
        frame_height=frame_h,
        inference_width=new_width,
        inference_height=new_height,
    )


def run_prepared_model_inference(model: YOLO, inference_input: InferenceInput, device: str) -> ResizedInferenceResult:
    """Run YOLO on a prepared frame and return the shared scaling metadata."""
    results = run_model_inference(model, inference_input.image, device)
    return ResizedInferenceResult(
        results=results,
        scale_x=inference_input.scale_x,
        scale_y=inference_input.scale_y,
        frame_width=inference_input.frame_width,
        frame_height=inference_input.frame_height,
        inference_width=inference_input.inference_width,
        inference_height=inference_input.inference_height,
    )


def run_resized_model_inference(model: YOLO, frame: np.ndarray, device: str) -> ResizedInferenceResult:
    """Resize a frame to the common inference width, run YOLO, and return scaling metadata."""
    return run_prepared_model_inference(model, prepare_inference_input(frame), device)


def warm_up_model(model: YOLO, label: str, device: str) -> None:
    """Force Ultralytics to initialize metadata and the inference backend during loading."""
    _ = model.names
    dummy_image = np.zeros((INFERENCE_WIDTH, INFERENCE_WIDTH, 3), dtype=np.uint8)
    print(f"Initialisation {label}...")
    run_model_inference(model, dummy_image, device)


def build_model_candidates(model_key: str = DEFAULT_MODEL_KEY, variant: str = "") -> list[tuple[str, str]]:
    """Build backend candidates for a model key and optional YOLO variant suffix."""
    model_stem = f"{MODEL_FILE_PREFIX}{model_key}{variant}"
    return [
        (backend_name, str(Path(MODELS_DIR) / f"{model_stem}.{extension}"))
        for backend_name, extension in MODEL_BACKEND_EXTENSIONS
    ]


def load_backend_model(label: str, candidates: list[tuple[str, str]], device: str) -> YOLO:
    """Load the first existing backend and fall back with diagnostics on failure."""
    last_error = None

    for backend_name, model_path in candidates:
        if not Path(model_path).is_file():
            print(f"Backend {backend_name} introuvable pour {label}: {model_path}")
            continue
        try:
            print(f"Chargement {label} via {backend_name}: {model_path}")
            model = YOLO(model_path)
            setattr(model, "_worldcam_backend_name", backend_name)
            if backend_name == "PyTorch" and device == "cuda":
                model.half()
            warm_up_model(model, label, device)
            return model
        except Exception as exc:
            last_error = exc
            if backend_name == "TensorRT":
                print(f"TensorRT indisponible pour {label}:\n  {describe_tensorrt_runtime_error(exc)}")
            else:
                print(f"Backend {backend_name} indisponible pour {label}: {type(exc).__name__}: {exc}")

    raise RuntimeError(f"Impossible de charger {label}; dernier échec: {last_error}")


def load_yolo_model(model_key: str = DEFAULT_MODEL_KEY) -> tuple[YOLO, str]:
    """Load YOLO for the selected model key, preferring TensorRT, then ONNX, then PyTorch."""
    patch_ultralytics_pose26()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Périphérique YOLO utilisé: {device}")

    model = load_backend_model("YOLO", build_model_candidates(model_key), device)
    return model, device


def load_segmentation_model(device: str, model_key: str = DEFAULT_MODEL_KEY) -> YOLO:
    """Load the YOLO segmentation model for the selected key, preferring TensorRT, then ONNX, then PyTorch."""
    patch_ultralytics_pose26()
    return load_backend_model("YOLO segmentation", build_model_candidates(model_key, "-seg"), device)


def load_pose_model(device: str, model_key: str = DEFAULT_MODEL_KEY) -> YOLO:
    """Load the YOLO pose model for the selected key, preferring TensorRT, then ONNX, then PyTorch."""
    patch_ultralytics_pose26()
    return load_backend_model("YOLO pose", build_model_candidates(model_key, "-pose"), device)
