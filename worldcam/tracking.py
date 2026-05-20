"""Lightweight backend-independent object tracking utilities."""

from dataclasses import dataclass, field
import math

import cv2
import numpy as np

from worldcam.config import (
    PERSON_TRACK_COLOR,
    PERSON_TRACK_DEBUG,
    PERSON_TRACK_MAX_AGE,
    PERSON_TRACK_MAX_DISTANCE,
    PERSON_TRACK_MIN_IOU,
    PERSON_TRACK_TRAIL_LENGTH,
)
from worldcam.detection import Detection, get_detection_color

VEHICLE_COUNT_CLASSES = {"car", "truck"}


@dataclass
class PersonTrack:
    """Persistent state for one tracked object."""

    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    class_name: str
    age: int = 0
    hits: int = 1
    trail: list[tuple[int, int]] = field(default_factory=list)

    @property
    def center(self) -> tuple[int, int]:
        """Return the current center point of the track bbox."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


def get_detection_class_name(detection: Detection) -> str:
    """Extract the class name from a WorldCam detection label."""
    return detection[4].rsplit(" ", 1)[0]


def calculate_iou(bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]) -> float:
    """Calculate intersection-over-union for two bounding boxes."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

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


def calculate_center_distance(bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]) -> float:
    """Calculate Euclidean distance between two bbox centers."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    center_a = ((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0)
    center_b = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
    return math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])


class PersonTracker:
    """Simple IoU/centroid tracker for all selected object detections."""

    def __init__(self) -> None:
        self.tracks: dict[int, PersonTrack] = {}
        self.next_track_id = 1
        self.vehicle_counts = {class_name: 0 for class_name in sorted(VEHICLE_COUNT_CLASSES)}

    def reset(self) -> None:
        """Clear all active tracks and restart ID allocation."""
        if PERSON_TRACK_DEBUG and self.tracks:
            print(f"Tracking objects: reset active_ids={sorted(self.tracks)}")
        self.tracks.clear()
        self.next_track_id = 1

    def update(self, detections: list[Detection]) -> list[PersonTrack]:
        """Update tracks from the latest selected detections and return active tracks."""
        object_boxes = [
            (x1, y1, x2, y2, confidence, get_detection_class_name(detection))
            for detection in detections
            for x1, y1, x2, y2, _label, confidence in [detection]
        ]

        unmatched_track_ids = set(self.tracks)
        matched_detection_indexes: set[int] = set()
        matches: list[tuple[int, int]] = []

        candidates: list[tuple[float, float, int, int]] = []
        for track_id, track in self.tracks.items():
            for detection_index, (x1, y1, x2, y2, _confidence, class_name) in enumerate(object_boxes):
                if track.class_name != class_name:
                    continue
                bbox = (x1, y1, x2, y2)
                iou = calculate_iou(track.bbox, bbox)
                distance = calculate_center_distance(track.bbox, bbox)
                if iou >= PERSON_TRACK_MIN_IOU or distance <= PERSON_TRACK_MAX_DISTANCE:
                    candidates.append((iou, -distance, track_id, detection_index))

        for _iou, _negative_distance, track_id, detection_index in sorted(candidates, reverse=True):
            if track_id not in unmatched_track_ids or detection_index in matched_detection_indexes:
                continue
            unmatched_track_ids.remove(track_id)
            matched_detection_indexes.add(detection_index)
            matches.append((track_id, detection_index))

        for track_id, detection_index in matches:
            x1, y1, x2, y2, confidence, class_name = object_boxes[detection_index]
            track = self.tracks[track_id]
            track.bbox = (x1, y1, x2, y2)
            track.confidence = confidence
            track.class_name = class_name
            track.age = 0
            track.hits += 1
            track.trail.append(track.center)
            if len(track.trail) > PERSON_TRACK_TRAIL_LENGTH:
                track.trail = track.trail[-PERSON_TRACK_TRAIL_LENGTH:]

        for detection_index, (x1, y1, x2, y2, confidence, class_name) in enumerate(object_boxes):
            if detection_index in matched_detection_indexes:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            track = PersonTrack(track_id=track_id, bbox=(x1, y1, x2, y2), confidence=confidence, class_name=class_name)
            track.trail.append(track.center)
            self.tracks[track_id] = track
            if class_name in self.vehicle_counts:
                self.vehicle_counts[class_name] += 1

        removed_track_ids = []
        for track_id in list(unmatched_track_ids):
            track = self.tracks[track_id]
            track.age += 1
            if track.age > PERSON_TRACK_MAX_AGE:
                removed_track_ids.append(track_id)
                del self.tracks[track_id]

        if PERSON_TRACK_DEBUG:
            print(
                "Tracking objects: "
                f"detections={len(object_boxes)}, matched={len(matches)}, "
                f"new={len(object_boxes) - len(matched_detection_indexes)}, lost={len(removed_track_ids)}, "
                f"active_ids={sorted(self.tracks)}"
            )

        return self.active_tracks()

    def active_tracks(self) -> list[PersonTrack]:
        """Return active tracks sorted by stable ID."""
        return [self.tracks[track_id] for track_id in sorted(self.tracks)]


def draw_person_tracks(frame: np.ndarray, tracks: list[PersonTrack], display_threshold: float = 0.5) -> None:
    """Draw tracked object IDs, boxes, and short motion trails."""
    for track in tracks:
        if track.confidence < display_threshold:
            continue

        x1, y1, x2, y2 = track.bbox
        label = f"{track.class_name} #{track.track_id} {track.confidence:.2f}"
        color = get_detection_color(f"{track.class_name} {track.confidence:.2f}")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            frame,
            label,
            (x1, min(y2 + 22, frame.shape[0] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

        for point_index in range(1, len(track.trail)):
            cv2.line(frame, track.trail[point_index - 1], track.trail[point_index], color, 1)


def draw_vehicle_counts(frame: np.ndarray, vehicle_counts: dict[str, int]) -> None:
    """Draw cumulative car and truck counters in the bottom-left corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    margin = 12
    line_height = 26
    labels = [
        f"Cars: {vehicle_counts.get('car', 0)}",
        f"Trucks: {vehicle_counts.get('truck', 0)}",
    ]
    y = max(margin + line_height, frame.shape[0] - margin - line_height * (len(labels) - 1))

    for label in labels:
        cv2.putText(
            frame,
            label,
            (margin, y),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
        )
        y += line_height
