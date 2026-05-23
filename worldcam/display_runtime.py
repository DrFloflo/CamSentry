"""Display overlays, pacing, and cleanup helpers for WorldCam."""

import subprocess
import time

import cv2
import torch
from ultralytics import YOLO

from worldcam.config import FRAME_INTERVAL
from worldcam.detection import Detection, draw_yolo_detections
from worldcam.pose import Pose, draw_pose_detections
from worldcam.segmentation import SegmentationMask, draw_segmentation_masks
from worldcam.stream_control import release_stream_resources
from worldcam.tracking import ObjectTrack, draw_object_tracks, draw_vehicle_counts
from worldcam.ui import draw_fps, draw_stream_counter


def draw_overlay(
    frame,
    fps: float,
    detections: list[Detection],
    poses: list[Pose],
    segmentations: list[SegmentationMask],
    display_threshold: float,
    object_tracks: list[ObjectTrack],
    stream_index: int,
    stream_total: int,
    vehicle_counts: dict[str, int],
) -> None:
    """Draw every visual overlay on the current frame."""
    draw_segmentation_masks(frame, segmentations, display_threshold)
    draw_yolo_detections(frame, detections, display_threshold)
    draw_object_tracks(frame, object_tracks, display_threshold)
    draw_pose_detections(frame, poses)
    draw_fps(frame, fps)
    draw_stream_counter(frame, stream_index, stream_total)
    draw_vehicle_counts(frame, vehicle_counts)


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
