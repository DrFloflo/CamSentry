"""Object detection analysis and drawing utilities."""

import cv2
import numpy as np
from ultralytics import YOLO

from worldcam.config import DETECTION_COLOR, INFERENCE_WIDTH

Detection = tuple[int, int, int, int, str]


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
        detections.append((final_x1, final_y1, final_x2, final_y2, label))

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

    results = model(resized_frame, verbose=False, device=device)[0]
    scale_x = frame_w / new_width
    scale_y = frame_h / new_height
    return extract_yolo_detections(results, model, scale_x, scale_y, selected_class_names)


def draw_yolo_detections(frame: np.ndarray, detections: list[Detection]) -> None:
    """Draw the latest YOLO detections on the displayed frame."""
    for final_x1, final_y1, final_x2, final_y2, label in detections:
        cv2.rectangle(frame, (final_x1, final_y1), (final_x2, final_y2), DETECTION_COLOR, 2)
        cv2.putText(
            frame,
            label,
            (final_x1, max(final_y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            DETECTION_COLOR,
            2,
        )
