"""Object detection analysis and drawing utilities."""

import cv2
import numpy as np
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

from worldcam.config import (
    DETECTION_CLASS_COLORS,
    DETECTION_FALLBACK_COLORS,
    INFERENCE_WIDTH,
    SAHI_OVERLAP_HEIGHT_RATIO,
    SAHI_OVERLAP_WIDTH_RATIO,
    SAHI_SLICE_HEIGHT,
    SAHI_SLICE_WIDTH,
)
from worldcam.models import run_model_inference

Detection = tuple[int, int, int, int, str, float]


def extract_yolo_detections(
    results,
    model: YOLO,
    scale_x: float,
    scale_y: float,
    selected_class_names: set[str],
) -> list[Detection]:
    """Convert selected YOLO results to original-frame coordinates."""
    detections = []

    for box in results.boxes:
        cls_name = model.names[int(box.cls)]
        if cls_name not in selected_class_names:
            continue

        confidence = float(box.conf)
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]
        final_x1 = int(x1 * scale_x)
        final_y1 = int(y1 * scale_y)
        final_x2 = int(x2 * scale_x)
        final_y2 = int(y2 * scale_y)
        label = f"{cls_name} {confidence:.2f}"
        detections.append((final_x1, final_y1, final_x2, final_y2, label, confidence))

    return detections


def run_yolo_analysis(
    frame: np.ndarray,
    model: YOLO,
    device: str,
    selected_class_names: set[str],
) -> list[Detection]:
    """Resize the frame, run YOLO26L inference, and return detections for the original frame."""
    frame_h, frame_w, _ = frame.shape
    new_width = min(INFERENCE_WIDTH, frame_w)
    new_height = int(frame_h * (new_width / frame_w))
    resized_frame = cv2.resize(frame, (new_width, new_height))

    results = run_model_inference(model, resized_frame, device)
    scale_x = frame_w / new_width
    scale_y = frame_h / new_height
    return extract_yolo_detections(results, model, scale_x, scale_y, selected_class_names)


def run_sahi_analysis(
    frame: np.ndarray,
    sahi_model,
    selected_class_names: set[str],
) -> list[Detection]:
    """Run SAHI sliced inference on the original frame and return selected detections."""
    frame_h, frame_w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = get_sliced_prediction(
        frame_rgb,
        sahi_model,
        slice_height=SAHI_SLICE_HEIGHT,
        slice_width=SAHI_SLICE_WIDTH,
        overlap_height_ratio=SAHI_OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=SAHI_OVERLAP_WIDTH_RATIO,
        verbose=0,
    )

    detections = []
    for prediction in result.object_prediction_list:
        cls_name = prediction.category.name
        if cls_name not in selected_class_names:
            continue

        bbox = prediction.bbox
        x1 = max(0, min(frame_w - 1, int(bbox.minx)))
        y1 = max(0, min(frame_h - 1, int(bbox.miny)))
        x2 = max(0, min(frame_w - 1, int(bbox.maxx)))
        y2 = max(0, min(frame_h - 1, int(bbox.maxy)))
        confidence = float(prediction.score.value)
        label = f"{cls_name} {confidence:.2f}"
        detections.append((x1, y1, x2, y2, label, confidence))

    return detections


def get_detection_color(label: str) -> tuple[int, int, int]:
    """Return a stable display color for a detection class label."""
    class_name = label.rsplit(" ", 1)[0]
    if class_name in DETECTION_CLASS_COLORS:
        return DETECTION_CLASS_COLORS[class_name]
    color_index = sum(ord(character) for character in class_name) % len(DETECTION_FALLBACK_COLORS)
    return DETECTION_FALLBACK_COLORS[color_index]


def draw_yolo_detections(frame: np.ndarray, detections: list[Detection], display_threshold: float = 0.5) -> None:
    """Draw the latest YOLO detections that meet the display confidence threshold."""
    for final_x1, final_y1, final_x2, final_y2, label, confidence in detections:
        if confidence < display_threshold:
            continue
        color = get_detection_color(label)
        cv2.rectangle(frame, (final_x1, final_y1), (final_x2, final_y2), color, 2)
        cv2.putText(
            frame,
            label,
            (final_x1, max(final_y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
