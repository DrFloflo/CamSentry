"""YOLO model loading helpers."""

import importlib
import importlib.metadata
import platform
import sys

import torch
from ultralytics import YOLO

from worldcam.compat import patch_ultralytics_pose26
from worldcam.config import (
    MODEL_ENGINE,
    MODEL_ONNX,
    MODEL_PT,
    POSE_MODEL_ENGINE,
    POSE_MODEL_ONNX,
    POSE_MODEL_PT,
)


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


def warm_up_model_metadata(model: YOLO) -> None:
    """Force Ultralytics to initialize the backend now, so fallback happens during loading."""
    _ = model.names


def load_backend_model(label: str, candidates: list[tuple[str, str]], device: str) -> YOLO:
    """Load the first available backend and fall back with diagnostics on failure."""
    last_error = None

    for backend_name, model_path in candidates:
        try:
            print(f"Chargement {label} via {backend_name}: {model_path}")
            model = YOLO(model_path)
            warm_up_model_metadata(model)
            if backend_name == "PyTorch" and device == "cuda":
                model.half()
            return model
        except Exception as exc:
            last_error = exc
            if backend_name == "TensorRT":
                print(f"TensorRT indisponible pour {label}:\n  {describe_tensorrt_runtime_error(exc)}")
            else:
                print(f"Backend {backend_name} indisponible pour {label}: {type(exc).__name__}: {exc}")

    raise RuntimeError(f"Impossible de charger {label}; dernier échec: {last_error}")


def load_yolo_model() -> tuple[YOLO, str]:
    """Load YOLO26L, preferring TensorRT, then ONNX, then PyTorch."""
    patch_ultralytics_pose26()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Périphérique YOLO utilisé: {device}")

    model = load_backend_model(
        "YOLO26L",
        [("TensorRT", MODEL_ENGINE), ("ONNX", MODEL_ONNX), ("PyTorch", MODEL_PT)],
        device,
    )
    return model, device


def load_pose_model(device: str) -> YOLO:
    """Load the YOLO pose model, preferring TensorRT, then ONNX, then PyTorch."""
    patch_ultralytics_pose26()
    return load_backend_model(
        "YOLO pose",
        [("TensorRT", POSE_MODEL_ENGINE), ("ONNX", POSE_MODEL_ONNX), ("PyTorch", POSE_MODEL_PT)],
        device,
    )
