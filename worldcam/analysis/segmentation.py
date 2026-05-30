"""Segmentation analysis and drawing utilities."""

import cv2
import numpy as np
from ultralytics import YOLO

from worldcam.core.config import SEGMENTATION_ALPHA, SEGMENTATION_CONTOUR_THICKNESS, SEGMENTATION_Y_OFFSET
from worldcam.analysis.detection import get_detection_color
from worldcam.core.models import InferenceInput, run_prepared_model_inference, run_resized_model_inference

SegmentationMask = tuple[np.ndarray, str, float]


def extract_segmentation_masks(
    results,
    model: YOLO,
    frame_width: int,
    frame_height: int,
    selected_class_names: set[str],
) -> list[SegmentationMask]:
    """Convert YOLO segmentation results to original-frame masks."""
    segmentations: list[SegmentationMask] = []
    masks = getattr(results, "masks", None)
    boxes = getattr(results, "boxes", None)
    polygons = getattr(masks, "xy", None) if masks is not None else None
    if masks is None or boxes is None or polygons is None:
        return segmentations

    scale_x = frame_width / max(1, results.orig_shape[1])
    scale_y = frame_height / max(1, results.orig_shape[0])
    for mask_index, polygon in enumerate(polygons):
        if mask_index >= len(boxes) or len(polygon) < 3:
            continue

        box = boxes[mask_index]
        class_name = model.names[int(box.cls)]
        if class_name not in selected_class_names:
            continue

        confidence = float(box.conf)
        points = polygon.copy()
        points[:, 0] = np.clip(points[:, 0] * scale_x, 0, frame_width - 1)
        points[:, 1] = np.clip((points[:, 1] * scale_y) + SEGMENTATION_Y_OFFSET, 0, frame_height - 1)
        binary_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.fillPoly(binary_mask, [points.astype(np.int32)], 1)
        label = f"{class_name} seg {confidence:.2f}"
        segmentations.append((binary_mask.astype(bool), label, confidence))

    return segmentations


def run_segmentation_analysis(
    frame: np.ndarray,
    segmentation_model: YOLO,
    device: str,
    selected_class_names: set[str],
    inference_input: InferenceInput | None = None,
) -> list[SegmentationMask]:
    """Resize the frame, run YOLO segmentation inference, and return full-frame masks."""
    inference = (
        run_prepared_model_inference(segmentation_model, inference_input, device)
        if inference_input is not None
        else run_resized_model_inference(segmentation_model, frame, device)
    )
    return extract_segmentation_masks(
        inference.results,
        segmentation_model,
        inference.frame_width,
        inference.frame_height,
        selected_class_names,
    )


def draw_segmentation_masks(frame: np.ndarray, segmentations: list[SegmentationMask], display_threshold: float = 0.5) -> None:
    """Draw translucent segmentation masks and thin contours on the displayed frame."""
    visible_segmentations = [
        (mask, label, confidence)
        for mask, label, confidence in segmentations
        if confidence >= display_threshold
    ]
    if not visible_segmentations:
        return

    overlay = frame.copy()
    for mask, label, confidence in visible_segmentations:
        color = get_detection_color(label.replace(" seg", ""))
        overlay[mask] = color

        contour_mask = mask.astype(np.uint8) * 255
        contours, _hierarchy = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, SEGMENTATION_CONTOUR_THICKNESS)

        moments = cv2.moments(contour_mask)
        if moments["m00"] > 0:
            text_x = int(moments["m10"] / moments["m00"])
            text_y = int(moments["m01"] / moments["m00"])
            cv2.putText(
                frame,
                label,
                (max(0, text_x - 25), max(20, text_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

    cv2.addWeighted(overlay, SEGMENTATION_ALPHA, frame, 1.0 - SEGMENTATION_ALPHA, 0, frame)
