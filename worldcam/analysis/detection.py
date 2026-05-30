"""Object detection analysis and drawing utilities."""

import cv2
import numpy as np
from sahi.postprocess.combine import NMSPostprocess
from sahi.prediction import ObjectPrediction
from ultralytics import YOLO

from worldcam.core.config import (
    DETECTION_CLASS_COLORS,
    DETECTION_FALLBACK_COLORS,
    SAHI_CONFIDENCE_THRESHOLD,
    SAHI_OVERLAP_HEIGHT_RATIO,
    SAHI_OVERLAP_WIDTH_RATIO,
    SAHI_SLICE_HEIGHT,
    SAHI_SLICE_WIDTH,
)
from worldcam.analysis.counting_zone import ZonePoints
from worldcam.core.models import run_model_inference, run_resized_model_inference

Detection = tuple[int, int, int, int, str, float]

DETECTION_DEDUP_IOU_THRESHOLD = 0.45
DETECTION_DEDUP_CONTAINMENT_THRESHOLD = 0.75
DETECTION_DEDUP_CENTER_DISTANCE_FACTOR = 0.35
DETECTION_DEDUP_MIN_IOU_FOR_CENTER_MATCH = 0.10


def extract_yolo_detections(
    results,
    model: YOLO,
    scale_x: float,
    scale_y: float,
    selected_class_names: set[str],
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[Detection]:
    """Convert selected YOLO results to original-frame coordinates."""
    detections = []

    for box in results.boxes:
        cls_name = model.names[int(box.cls)]
        if cls_name not in selected_class_names:
            continue

        confidence = float(box.conf)
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]
        final_x1 = int(x1 * scale_x) + offset_x
        final_y1 = int(y1 * scale_y) + offset_y
        final_x2 = int(x2 * scale_x) + offset_x
        final_y2 = int(y2 * scale_y) + offset_y
        label = f"{cls_name} {confidence:.2f}"
        detections.append((final_x1, final_y1, final_x2, final_y2, label, confidence))

    return detections


def crop_frame_for_exclusion_zone(
    frame: np.ndarray,
    exclusion_zone_points: ZonePoints | None,
    exclusion_zone_enabled: bool,
) -> tuple[np.ndarray, int, int]:
    """Return a cropped analysis frame that excludes the configured polygon whenever possible."""
    points = exclusion_zone_points or []
    if not exclusion_zone_enabled or len(points) < 3:
        return frame, 0, 0

    frame_h, frame_w = frame.shape[:2]
    contour = np.array(points, dtype=np.int32)
    excluded_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(excluded_mask, [contour], 1)
    included_y, included_x = np.where(excluded_mask == 0)
    if included_x.size == 0 or included_y.size == 0:
        return frame, 0, 0

    x1 = int(included_x.min())
    y1 = int(included_y.min())
    x2 = int(included_x.max()) + 1
    y2 = int(included_y.max()) + 1
    analysis_frame = frame[y1:y2, x1:x2].copy()

    excluded_crop = excluded_mask[y1:y2, x1:x2].astype(bool)
    if excluded_crop.any():
        analysis_frame[excluded_crop] = 0

    return analysis_frame, x1, y1


def run_yolo_analysis(
    frame: np.ndarray,
    model: YOLO,
    device: str,
    selected_class_names: set[str],
    exclusion_zone_points: ZonePoints | None = None,
    exclusion_zone_enabled: bool = False,
) -> list[Detection]:
    """Resize the analysis area, run YOLO inference, and return detections for the original frame."""
    analysis_frame, offset_x, offset_y = crop_frame_for_exclusion_zone(frame, exclusion_zone_points, exclusion_zone_enabled)
    inference = run_resized_model_inference(model, analysis_frame, device)
    return extract_yolo_detections(
        inference.results,
        model,
        inference.scale_x,
        inference.scale_y,
        selected_class_names,
        offset_x,
        offset_y,
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
    offset_x: int = 0,
    offset_y: int = 0,
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
                            max(0, min(frame_w - 1, int(x1 + x_start))) + offset_x,
                            max(0, min(frame_h - 1, int(y1 + y_start))) + offset_y,
                            max(0, min(frame_w - 1, int(x2 + x_start))) + offset_x,
                            max(0, min(frame_h - 1, int(y2 + y_start))) + offset_y,
                        ],
                        category_id=cls_id,
                        category_name=cls_name,
                        score=confidence,
                        full_shape=[frame_h + offset_y, frame_w + offset_x],
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
    exclusion_zone_points: ZonePoints | None = None,
    exclusion_zone_enabled: bool = False,
) -> list[Detection]:
    """Run SAHI-style sliced inference with the active YOLO backend, including TensorRT engines."""
    original_h, original_w, _ = frame.shape
    analysis_frame, offset_x, offset_y = crop_frame_for_exclusion_zone(frame, exclusion_zone_points, exclusion_zone_enabled)
    object_predictions = extract_sliced_yolo_predictions(analysis_frame, model, device, selected_class_names, offset_x, offset_y)
    if not object_predictions:
        return []
    if len(object_predictions) == 1:
        return convert_sahi_predictions_to_detections(object_predictions, original_w, original_h)

    nms_postprocess = NMSPostprocess(match_threshold=0.5, match_metric="IOU", class_agnostic=False)
    combined_predictions = nms_postprocess(object_predictions)
    return convert_sahi_predictions_to_detections(combined_predictions, original_w, original_h)


def get_detection_class_name(detection: Detection) -> str:
    """Extract the class name from a WorldCam detection label."""
    return detection[4].rsplit(" ", 1)[0]


def calculate_detection_iou(detection_a: Detection, detection_b: Detection) -> float:
    """Calculate intersection-over-union for two detection boxes."""
    ax1, ay1, ax2, ay2 = detection_a[:4]
    bx1, by1, bx2, by2 = detection_b[:4]

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def calculate_detection_containment(detection_a: Detection, detection_b: Detection) -> float:
    """Return the intersection ratio over the smaller detection area."""
    ax1, ay1, ax2, ay2 = detection_a[:4]
    bx1, by1, bx2, by2 = detection_b[:4]

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    smaller_area = min(area_a, area_b)
    if smaller_area <= 0:
        return 0.0
    return intersection_area / smaller_area


def calculate_detection_center_distance(detection_a: Detection, detection_b: Detection) -> float:
    """Calculate Euclidean distance between two detection centers."""
    ax1, ay1, ax2, ay2 = detection_a[:4]
    bx1, by1, bx2, by2 = detection_b[:4]
    center_a = ((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0)
    center_b = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
    return float(np.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]))


def is_duplicate_detection(candidate: Detection, kept_detection: Detection) -> bool:
    """Return whether two same-class detections likely describe the same object."""
    if get_detection_class_name(candidate) != get_detection_class_name(kept_detection):
        return False

    iou = calculate_detection_iou(candidate, kept_detection)
    if iou >= DETECTION_DEDUP_IOU_THRESHOLD:
        return True

    containment = calculate_detection_containment(candidate, kept_detection)
    if containment >= DETECTION_DEDUP_CONTAINMENT_THRESHOLD:
        return True

    candidate_width = max(1, candidate[2] - candidate[0])
    candidate_height = max(1, candidate[3] - candidate[1])
    kept_width = max(1, kept_detection[2] - kept_detection[0])
    kept_height = max(1, kept_detection[3] - kept_detection[1])
    center_distance_limit = min(candidate_width, candidate_height, kept_width, kept_height) * DETECTION_DEDUP_CENTER_DISTANCE_FACTOR
    return iou >= DETECTION_DEDUP_MIN_IOU_FOR_CENTER_MATCH and calculate_detection_center_distance(candidate, kept_detection) <= center_distance_limit


def deduplicate_detections(detections: list[Detection]) -> list[Detection]:
    """Suppress duplicate detections by class before display and tracking."""
    kept_detections: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item[5], reverse=True):
        if any(is_duplicate_detection(detection, kept_detection) for kept_detection in kept_detections):
            continue
        kept_detections.append(detection)
    return kept_detections


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
