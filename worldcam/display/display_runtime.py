"""Display overlays, pacing, and cleanup helpers for WorldCam."""

import subprocess
import time

import cv2
import torch
from ultralytics import YOLO

from worldcam.core.config import (
    COUNTING_ZONE_COLOR,
    COUNTING_ZONE_EDIT_COLOR,
    COUNTING_ZONE_HANDLE_COLOR,
    COUNTING_ZONE_HANDLE_RADIUS,
    EXCLUSION_ZONE_COLOR,
    EXCLUSION_ZONE_EDIT_COLOR,
    EXCLUSION_ZONE_HANDLE_COLOR,
    FRAME_INTERVAL,
)
from worldcam.analysis.detection import Detection, draw_yolo_detections
from worldcam.analysis.pose import Pose, draw_pose_detections
from worldcam.analysis.segmentation import SegmentationMask, draw_segmentation_masks
from worldcam.stream.stream_control import release_stream_resources
from worldcam.analysis.tracking import ObjectTrack, draw_object_tracks, draw_vehicle_counts

ZonePoints = list[tuple[int, int]]

def draw_timestamp(frame: cv2.Mat) -> None:
    """Draw the current timestamp in the top-left corner."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    margin = 10
    text_size, _ = cv2.getTextSize(timestamp, font, font_scale, thickness)
    text_width, text_height = text_size
    x = margin
    y = margin + text_height
    cv2.putText(frame, timestamp, (x, y), font, font_scale, (255, 255, 255), thickness)

def draw_overlay(
    frame,
    detections: list[Detection],
    poses: list[Pose],
    segmentations: list[SegmentationMask],
    display_threshold: float,
    object_tracks: list[ObjectTrack],
    stream_index: int,
    stream_total: int,
    vehicle_counts: dict[str, int],
    tracking_enabled: bool,
    counting_zone_points: ZonePoints | None = None,
    counting_zone_enabled: bool = False,
    counting_zone_edit_enabled: bool = False,
    exclusion_zone_points: ZonePoints | None = None,
    exclusion_zone_display_enabled: bool = False,
    exclusion_zone_processing_enabled: bool = False,
    exclusion_zone_edit_enabled: bool = False,
) -> None:
    """Draw every visual overlay on the current frame."""
    draw_segmentation_masks(frame, segmentations, display_threshold)
    if tracking_enabled:
        draw_object_tracks(frame, object_tracks, display_threshold)
    else:
        draw_yolo_detections(frame, detections, display_threshold)
    draw_pose_detections(frame, poses)
    draw_vehicle_counts(frame, vehicle_counts)
    draw_counting_zone(frame, counting_zone_points, counting_zone_enabled, counting_zone_edit_enabled)
    draw_exclusion_zone(
        frame,
        exclusion_zone_points,
        exclusion_zone_display_enabled,
        exclusion_zone_processing_enabled,
        exclusion_zone_edit_enabled,
    )
    draw_timestamp(frame)


def draw_counting_zone(
    frame,
    counting_zone_points: ZonePoints | None,
    counting_zone_enabled: bool,
    counting_zone_edit_enabled: bool,
) -> None:
    """Draw the counting-zone polygon with a subtle fill and unobtrusive edit handles."""
    points = counting_zone_points or []

    if not (counting_zone_enabled or counting_zone_edit_enabled):
        return

    color = COUNTING_ZONE_EDIT_COLOR if counting_zone_edit_enabled else COUNTING_ZONE_COLOR
    if len(points) >= 2:
        import numpy as np

        contour = np.array(points, dtype=np.int32)
        is_closed = len(points) >= 3
        if is_closed:
            zone_overlay = frame.copy()
            cv2.fillPoly(zone_overlay, [contour], (255, 255, 255))
            cv2.addWeighted(zone_overlay, 0.12, frame, 0.88, 0, frame)
        cv2.polylines(frame, [contour], is_closed, color, 1)

    if not counting_zone_edit_enabled:
        return

    for index, point in enumerate(points, start=1):
        cv2.circle(frame, point, max(3, COUNTING_ZONE_HANDLE_RADIUS // 2), COUNTING_ZONE_HANDLE_COLOR, 1)
        cv2.putText(frame, str(index), (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COUNTING_ZONE_HANDLE_COLOR, 1)


def draw_exclusion_zone(
    frame,
    exclusion_zone_points: ZonePoints | None,
    exclusion_zone_display_enabled: bool,
    exclusion_zone_processing_enabled: bool,
    exclusion_zone_edit_enabled: bool,
) -> None:
    """Draw the exclusion-zone polygon, its active state, and edit handles."""
    points = exclusion_zone_points or []
    if not (exclusion_zone_display_enabled or exclusion_zone_edit_enabled):
        return

    color = EXCLUSION_ZONE_EDIT_COLOR if exclusion_zone_edit_enabled else EXCLUSION_ZONE_COLOR
    if len(points) >= 2:
        import numpy as np

        contour = np.array(points, dtype=np.int32)
        is_closed = len(points) >= 3
        if is_closed:
            zone_overlay = frame.copy()
            fill_color = EXCLUSION_ZONE_COLOR if exclusion_zone_processing_enabled else (128, 128, 128)
            cv2.fillPoly(zone_overlay, [contour], fill_color)
            cv2.addWeighted(zone_overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [contour], is_closed, color, 2)

    if len(points) >= 1:
        label = "EXCLUSION ON" if exclusion_zone_processing_enabled else "EXCLUSION OFF"
        label_origin = (max(0, min(x for x, _y in points)), max(20, min(y for _x, y in points) + 22))
        cv2.putText(frame, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    if not exclusion_zone_edit_enabled:
        return

    for index, point in enumerate(points, start=1):
        cv2.circle(frame, point, max(3, COUNTING_ZONE_HANDLE_RADIUS // 2), EXCLUSION_ZONE_HANDLE_COLOR, 1)
        cv2.putText(frame, str(index), (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, EXCLUSION_ZONE_HANDLE_COLOR, 1)


def throttle_display(next_frame_at: float) -> float:
    """Stabilize display timing so delayed frames are not replayed too quickly."""
    now = time.perf_counter()
    if now < next_frame_at:
        time.sleep(next_frame_at - now)
    elif now - next_frame_at > FRAME_INTERVAL:
        next_frame_at = now
    return next_frame_at + FRAME_INTERVAL


def cleanup_resources(
    cap: cv2.VideoCapture | None,
    ffmpeg_process: subprocess.Popen | None,
    model: YOLO,
    pose_model: YOLO | None,
    segmentation_model: YOLO | None,
    device: str,
) -> None:
    """Release stream, model, and OpenCV resources."""
    release_stream_resources(cap, ffmpeg_process)
    if device == "cuda":
        del model
        if pose_model is not None:
            del pose_model
        if segmentation_model is not None:
            del segmentation_model
        torch.cuda.empty_cache()
    cv2.destroyAllWindows()
