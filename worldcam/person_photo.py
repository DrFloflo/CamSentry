"""Clicked-person cutout helpers for the WorldCam application."""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

from worldcam.detection import Detection
from worldcam.face_scanner import close_face_zoom_window, scan_and_open_face_zoom
from worldcam.image_upscale import upscale_for_display
from worldcam.models import load_segmentation_model
from worldcam.segmentation import SegmentationMask, run_segmentation_analysis

PERSON_PHOTO_WINDOW_NAME = "Photo personne selectionnee"
PERSON_PHOTO_MIN_CONFIDENCE = 0.15
PERSON_PHOTO_BACKGROUND_COLOR = (255, 255, 255)
PERSON_PHOTO_PADDING = 16
PERSON_PHOTO_MIN_WINDOW_WIDTH = 260
PERSON_PHOTO_MIN_WINDOW_HEIGHT = 320


class ClickState(dict):
    """Mutable state shared with the OpenCV mouse callback."""

    click_position: tuple[int, int] | None


def get_detection_class_name(detection: Detection) -> str:
    """Extract the class name from a WorldCam detection label."""
    return detection[4].rsplit(" ", 1)[0]


def get_segmentation_class_name(segmentation: SegmentationMask) -> str:
    """Extract the class name from a WorldCam segmentation label."""
    return segmentation[1].split(" seg ", 1)[0]


def calculate_bbox_iou(bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]) -> float:
    """Calculate intersection-over-union for two bounding boxes."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)
    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def get_mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the bounding box of a binary mask, or None when the mask is empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def find_clicked_person_detection(
    click_x: int,
    click_y: int,
    detections: list[Detection],
    display_threshold: float,
) -> Detection | None:
    """Return the smallest visible person detection containing the click point."""
    clicked_detections = []
    for detection in detections:
        x1, y1, x2, y2, _label, confidence = detection
        if confidence < display_threshold or get_detection_class_name(detection) != "person":
            continue
        if x1 <= click_x <= x2 and y1 <= click_y <= y2:
            clicked_detections.append((max(0, x2 - x1) * max(0, y2 - y1), detection))

    if not clicked_detections:
        return None
    return min(clicked_detections, key=lambda candidate: candidate[0])[1]


def find_matching_person_mask(
    click_x: int,
    click_y: int,
    detection: Detection,
    segmentations: list[SegmentationMask],
    display_threshold: float,
) -> SegmentationMask | None:
    """Find the person segmentation mask that best matches the clicked detection."""
    detection_bbox = detection[:4]
    containing_masks = []
    overlapping_masks = []

    for segmentation in segmentations:
        mask, _label, confidence = segmentation
        if confidence < min(display_threshold, PERSON_PHOTO_MIN_CONFIDENCE) or get_segmentation_class_name(segmentation) != "person":
            continue

        mask_height, mask_width = mask.shape[:2]
        if 0 <= click_x < mask_width and 0 <= click_y < mask_height and mask[click_y, click_x]:
            containing_masks.append((int(mask.sum()), segmentation))
            continue

        mask_bbox = get_mask_bbox(mask)
        if mask_bbox is None:
            continue
        iou = calculate_bbox_iou(detection_bbox, mask_bbox)
        if iou > 0:
            overlapping_masks.append((iou, segmentation))

    if containing_masks:
        return min(containing_masks, key=lambda candidate: candidate[0])[1]
    if overlapping_masks:
        return max(overlapping_masks, key=lambda candidate: candidate[0])[1]
    return None


def build_person_cutout(frame: np.ndarray, detection: Detection, segmentation: SegmentationMask) -> np.ndarray | None:
    """Crop a clicked person and remove the background with the segmentation mask."""
    frame_height, frame_width = frame.shape[:2]
    mask = segmentation[0]
    mask_bbox = get_mask_bbox(mask)
    x1, y1, x2, y2 = detection[:4]
    if mask_bbox is not None:
        x1 = min(x1, mask_bbox[0])
        y1 = min(y1, mask_bbox[1])
        x2 = max(x2, mask_bbox[2])
        y2 = max(y2, mask_bbox[3])

    x1 = max(0, min(frame_width - 1, x1))
    y1 = max(0, min(frame_height - 1, y1))
    x2 = max(x1 + 1, min(frame_width, x2))
    y2 = max(y1 + 1, min(frame_height, y2))

    crop = frame[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]
    if crop.size == 0 or mask_crop.size == 0 or not np.any(mask_crop):
        return None

    cutout = np.full_like(crop, PERSON_PHOTO_BACKGROUND_COLOR)
    cutout[mask_crop] = crop[mask_crop]
    return cv2.copyMakeBorder(
        cutout,
        PERSON_PHOTO_PADDING,
        PERSON_PHOTO_PADDING,
        PERSON_PHOTO_PADDING,
        PERSON_PHOTO_PADDING,
        cv2.BORDER_CONSTANT,
        value=PERSON_PHOTO_BACKGROUND_COLOR,
    )


def open_person_photo_window(photo: np.ndarray) -> None:
    """Open or refresh a dedicated resizable OpenCV window with the selected person photo."""
    photo = upscale_for_display(photo, PERSON_PHOTO_MIN_WINDOW_WIDTH, PERSON_PHOTO_MIN_WINDOW_HEIGHT, max_scale=3.0)
    scanned_photo = scan_and_open_face_zoom(photo)
    photo_height, photo_width = scanned_photo.shape[:2]
    window_width = max(PERSON_PHOTO_MIN_WINDOW_WIDTH, photo_width)
    window_height = max(PERSON_PHOTO_MIN_WINDOW_HEIGHT, photo_height)
    cv2.namedWindow(PERSON_PHOTO_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PERSON_PHOTO_WINDOW_NAME, window_width, window_height)
    cv2.moveWindow(PERSON_PHOTO_WINDOW_NAME, 80, 80)
    cv2.imshow(PERSON_PHOTO_WINDOW_NAME, scanned_photo)
    cv2.waitKey(1)


def crop_detection_with_padding(frame: np.ndarray, detection: Detection, padding_ratio: float = 0.18) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Crop the clicked detection with a small margin and return crop plus source bbox."""
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = detection[:4]
    padding_x = int(max(24, (x2 - x1) * padding_ratio))
    padding_y = int(max(24, (y2 - y1) * padding_ratio))
    crop_x1 = max(0, x1 - padding_x)
    crop_y1 = max(0, y1 - padding_y)
    crop_x2 = min(frame_width, x2 + padding_x)
    crop_y2 = min(frame_height, y2 + padding_y)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None
    return frame[crop_y1:crop_y2, crop_x1:crop_x2].copy(), (crop_x1, crop_y1, crop_x2, crop_y2)


def build_crop_person_photo(crop: np.ndarray, segmentations: list[SegmentationMask]) -> np.ndarray | None:
    """Build a standalone person photo from crop-local segmentation masks."""
    person_segmentations = [
        segmentation
        for segmentation in segmentations
        if get_segmentation_class_name(segmentation) == "person" and segmentation[2] >= PERSON_PHOTO_MIN_CONFIDENCE
    ]
    if not person_segmentations:
        return None

    segmentation = max(person_segmentations, key=lambda candidate: int(candidate[0].sum()))
    return build_person_cutout(crop, (0, 0, crop.shape[1], crop.shape[0], "person", 1.0), segmentation)


def show_clicked_person_photo(
    frame: np.ndarray,
    click_position: tuple[int, int],
    detections: list[Detection],
    segmentations: list[SegmentationMask],
    segmentation_model: YOLO | None,
    device: str,
    display_threshold: float,
) -> tuple[YOLO | None, list[SegmentationMask]]:
    """Open a separate window with the clicked person cut out by segmentation."""
    click_x, click_y = click_position
    detection = find_clicked_person_detection(click_x, click_y, detections, display_threshold)
    if detection is None:
        print("Aucune detection person sous le clic.")
        return segmentation_model, segmentations

    crop_result = crop_detection_with_padding(frame, detection)
    if crop_result is None:
        print("Impossible de recadrer la detection person cliquee.")
        return segmentation_model, segmentations
    person_crop, _source_bbox = crop_result

    if segmentation_model is None:
        try:
            segmentation_model = load_segmentation_model(device)
        except Exception as exc:
            print(f"Impossible de charger la segmentation pour le clic person: {exc}")
            open_person_photo_window(person_crop)
            return segmentation_model, segmentations

    try:
        crop_segmentations = run_segmentation_analysis(person_crop, segmentation_model, device, {"person"})
        photo = build_crop_person_photo(person_crop, crop_segmentations)
    except Exception as exc:
        print(f"Erreur segmentation pendant le clic person: {exc}")
        photo = None

    if photo is None:
        segmentation = find_matching_person_mask(click_x, click_y, detection, segmentations, display_threshold)
        if segmentation is None:
            try:
                segmentations = run_segmentation_analysis(frame, segmentation_model, device, {"person"})
                segmentation = find_matching_person_mask(click_x, click_y, detection, segmentations, display_threshold)
            except Exception as exc:
                print(f"Erreur segmentation frame entiere pendant le clic person: {exc}")

        photo = build_person_cutout(frame, detection, segmentation) if segmentation is not None else None

    if photo is None:
        print("Aucun masque person exploitable; affichage de la photo recadree sans decoupage.")
        photo = cv2.copyMakeBorder(
            person_crop,
            PERSON_PHOTO_PADDING,
            PERSON_PHOTO_PADDING,
            PERSON_PHOTO_PADDING,
            PERSON_PHOTO_PADDING,
            cv2.BORDER_CONSTANT,
            value=PERSON_PHOTO_BACKGROUND_COLOR,
        )

    open_person_photo_window(photo)
    return segmentation_model, segmentations


def handle_main_window_mouse(event: int, x: int, y: int, _flags: int, userdata: ClickState) -> None:
    """Store left-click positions from the main OpenCV window."""
    if event == cv2.EVENT_LBUTTONDOWN:
        userdata["click_position"] = (x, y)


def close_person_photo_window() -> None:
    """Close the segmented person photo and face zoom windows if they exist."""
    close_face_zoom_window()
    try:
        cv2.destroyWindow(PERSON_PHOTO_WINDOW_NAME)
    except cv2.error:
        pass
