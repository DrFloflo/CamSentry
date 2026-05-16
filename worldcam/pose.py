"""Pose estimation analysis and drawing utilities."""

import cv2
import numpy as np
from ultralytics import YOLO

from worldcam.config import (
    INFERENCE_WIDTH,
    POSE_CONFIDENCE_THRESHOLD,
    POSE_KEYPOINT_COLOR,
    POSE_SKELETON,
    POSE_SKELETON_COLOR,
)

PosePoint = tuple[int, int, float]
Pose = list[PosePoint]


def extract_pose_keypoints(results, scale_x: float, scale_y: float) -> list[Pose]:
    """Convert YOLO pose keypoints to original-frame coordinates."""
    poses = []
    keypoints = getattr(results, "keypoints", None)
    if keypoints is None or keypoints.xy is None:
        return poses

    xy_values = keypoints.xy.cpu().numpy()
    conf_values = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None

    for pose_index, pose_points in enumerate(xy_values):
        pose = []
        for keypoint_index, (x, y) in enumerate(pose_points):
            confidence = 1.0 if conf_values is None else float(conf_values[pose_index][keypoint_index])
            pose.append((int(x * scale_x), int(y * scale_y), confidence))
        poses.append(pose)

    return poses


def run_pose_analysis(frame: np.ndarray, pose_model: YOLO, device: str) -> list[Pose]:
    """Resize the frame, run YOLO pose inference, and return poses for the original frame."""
    frame_h, frame_w, _ = frame.shape
    new_width = min(INFERENCE_WIDTH, frame_w)
    new_height = int(frame_h * (new_width / frame_w))
    resized_frame = cv2.resize(frame, (new_width, new_height))

    results = pose_model(resized_frame, verbose=False, device=device)[0]
    scale_x = frame_w / new_width
    scale_y = frame_h / new_height
    return extract_pose_keypoints(results, scale_x, scale_y)


def draw_pose_detections(frame: np.ndarray, poses: list[Pose]) -> None:
    """Draw pose keypoints and skeletons on the displayed frame."""
    for pose in poses:
        for point_a, point_b in POSE_SKELETON:
            if point_a >= len(pose) or point_b >= len(pose):
                continue

            x1, y1, conf1 = pose[point_a]
            x2, y2, conf2 = pose[point_b]
            if conf1 < POSE_CONFIDENCE_THRESHOLD or conf2 < POSE_CONFIDENCE_THRESHOLD:
                continue

            cv2.line(frame, (x1, y1), (x2, y2), POSE_SKELETON_COLOR, 2)

        for x, y, confidence in pose:
            if confidence >= POSE_CONFIDENCE_THRESHOLD:
                cv2.circle(frame, (x, y), 3, POSE_KEYPOINT_COLOR, -1)
