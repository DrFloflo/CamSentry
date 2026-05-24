"""Object detection analysis and drawing utilities."""

import cv2
import numpy as np
from sahi.postprocess.combine import NMSPostprocess
from sahi.prediction import ObjectPrediction
from ultralytics import YOLO

from worldcam.config import (
    DETECTION_CLASS_COLORS,
    DETECTION_FALLBACK_COLORS,
    SAHI_CONFIDENCE_THRESHOLD,
    SAHI_OVERLAP_HEIGHT_RATIO,
    SAHI_OVERLAP_WIDTH_RATIO,
    SAHI_SLICE_HEIGHT,
    SAHI_SLICE_WIDTH,
)
from worldcam.models import run_model_inference, run_resized_model_inference

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
    """Resize the frame, run YOLO inference, and return detections for the original frame."""
    inference = run_resized_model_inference(model, frame, device)
    return extract_yolo_detections(
        inference.results,
        model,
        inference.scale_x,
        inference.scale_y,
        selected_class_names,
    )


def get_slice_starts(image_size: int, slice_size: int, overlap_ratio: float) -> list[int]:
    """Return deterministic slice start positions that cover the full image axis."""
    if image_size <= slice_size:
        return [0]

    step = max(1, int(slice_size * (1.0 - overlap_ratio)))
    starts = list(range(0, image_size - slice_size + 1, step))
    last_start = image_size - slice_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def extract_sliced_yolo_predictions(
    frame: np.ndarray,
    model: YOLO,
    device: str,
    selected_class_names: set[str],
) -> list[ObjectPrediction]:
    """Run tiled YOLO inference and convert each tile result into SAHI predictions."""
    frame_h, frame_w, _ = frame.shape
    slice_height = min(SAHI_SLICE_HEIGHT, frame_h)
    slice_width = min(SAHI_SLICE_WIDTH, frame_w)
    y_starts = get_slice_starts(frame_h, slice_height, SAHI_OVERLAP_HEIGHT_RATIO)
    x_starts = get_slice_starts(frame_w, slice_width, SAHI_OVERLAP_WIDTH_RATIO)
    object_predictions = []

    for y_start in y_starts:
        for x_start in x_starts:
            y_end = min(y_start + slice_height, frame_h)
            x_end = min(x_start + slice_width, frame_w)
            tile = frame[y_start:y_end, x_start:x_end]
            results = run_model_inference(model, tile, device)

            for box in results.boxes:
                cls_id = int(box.cls)
                cls_name = model.names[cls_id]
                confidence = float(box.conf)
                if cls_name not in selected_class_names or confidence < SAHI_CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
                object_predictions.append(
                    ObjectPrediction(
                        bbox=[
                            max(0, min(frame_w - 1, int(x1 + x_start))),
                            max(0, min(frame_h - 1, int(y1 + y_start))),
                            max(0, min(frame_w - 1, int(x2 + x_start))),
                            max(0, min(frame_h - 1, int(y2 + y_start))),
                        ],
                        category_id=cls_id,
                        category_name=cls_name,
                        score=confidence,
                        full_shape=[frame_h, frame_w],
                    )
                )

    return object_predictions


def convert_sahi_predictions_to_detections(
    object_predictions: list[ObjectPrediction],
    frame_width: int,
    frame_height: int,
) -> list[Detection]:
    """Convert combined SAHI object predictions to WorldCam detection tuples."""
    detections = []
    for prediction in object_predictions:
        bbox = prediction.bbox
        x1 = max(0, min(frame_width - 1, int(bbox.minx)))
        y1 = max(0, min(frame_height - 1, int(bbox.miny)))
        x2 = max(0, min(frame_width - 1, int(bbox.maxx)))
        y2 = max(0, min(frame_height - 1, int(bbox.maxy)))
        confidence = float(prediction.score.value)
        label = f"{prediction.category.name} {confidence:.2f}"
        detections.append((x1, y1, x2, y2, label, confidence))
    return detections


def run_sahi_analysis(
    frame: np.ndarray,
    model: YOLO,
    device: str,
    selected_class_names: set[str],
) -> list[Detection]:
    """Run SAHI-style sliced inference with the active YOLO backend, including TensorRT engines."""
    frame_h, frame_w, _ = frame.shape
    object_predictions = extract_sliced_yolo_predictions(frame, model, device, selected_class_names)
    if not object_predictions:
        return []
    if len(object_predictions) == 1:
        return convert_sahi_predictions_to_detections(object_predictions, frame_w, frame_h)

    nms_postprocess = NMSPostprocess(match_threshold=0.5, match_metric="IOU", class_agnostic=False)
    combined_predictions = nms_postprocess(object_predictions)
    return convert_sahi_predictions_to_detections(combined_predictions, frame_w, frame_h)


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
        cv2.rectangle(frame, (final_x1, final_y1), (final_x2, final_y2), color, 1)
        cv2.putText(
            frame,
            label,
            (final_x1, max(final_y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
