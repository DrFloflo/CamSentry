"""Lightweight backend-independent object tracking utilities."""

from dataclasses import dataclass, field
import math

import cv2
import numpy as np

from worldcam.core.config import (
    COUNTING_ZONE_REAL_LENGTH_METERS,
    COUNTING_ZONE_REAL_WIDTH_METERS,
    FRAME_SKIP,
    PERSON_TRACK_DEBUG,
    PERSON_TRACK_MAX_AGE,
    PERSON_TRACK_MAX_DISTANCE,
    PERSON_TRACK_MIN_IOU,
    PERSON_TRACK_TRAIL_LENGTH,
    SPEED_ESTIMATION_ENABLED,
    TARGET_FPS,
)
from worldcam.analysis.count_persistence import VehicleCountStore
from worldcam.analysis.counting_zone import ZonePoints, point_inside_zone
from worldcam.analysis.detection import Detection, get_detection_color

VEHICLE_SOURCE_CLASSES = {"car", "truck", "bus", "motorcycle"}
VEHICLE_COUNT_KEY = "vehicle"
VEHICLE_COUNT_MIN_HITS = 3
ACTIVE_TRACK_OVERLAP_IOU_THRESHOLD = 0.80


@dataclass
class CountedVehicle:
    """Short-lived memory for a vehicle that was already counted."""

    class_name: str
    bbox: tuple[int, int, int, int]
    track_id: int | None = None
    age: int = 0


def estimate_track_speed_kmh(track: "ObjectTrack", zone_points: ZonePoints) -> float | None:
    """Estimate track speed by projecting zone progress into an approximate real rectangle."""
    if not SPEED_ESTIMATION_ENABLED or len(zone_points) < 4 or len(track.trail) < 2:
        return None

    homography = cv2.getPerspectiveTransform(
        np.array(zone_points[:4], dtype=np.float32),
        np.array(
            [
                (0.0, 0.0),
                (COUNTING_ZONE_REAL_WIDTH_METERS, 0.0),
                (COUNTING_ZONE_REAL_WIDTH_METERS, COUNTING_ZONE_REAL_LENGTH_METERS),
                (0.0, COUNTING_ZONE_REAL_LENGTH_METERS),
            ],
            dtype=np.float32,
        ),
    )
    trail_points = np.array(track.trail, dtype=np.float32).reshape(-1, 1, 2)
    projected_points = cv2.perspectiveTransform(trail_points, homography).reshape(-1, 2)

    distance_meters = 0.0
    for point_index in range(1, len(projected_points)):
        previous_x, previous_y = projected_points[point_index - 1]
        current_x, current_y = projected_points[point_index]
        distance_meters += math.hypot(current_x - previous_x, current_y - previous_y)

    duration_seconds = (len(projected_points) - 1) * FRAME_SKIP / TARGET_FPS
    if duration_seconds <= 0.0 or distance_meters <= 0.0:
        return None
    return distance_meters / duration_seconds * 3.6


@dataclass
class ObjectTrack:
    """Persistent state for one tracked object."""

    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    class_name: str
    age: int = 0
    hits: int = 1
    counted: bool = False
    trail: list[tuple[int, int]] = field(default_factory=list)

    @property
    def center(self) -> tuple[int, int]:
        """Return the current center point of the track bbox."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class VehicleCounter:
    """Cumulative vehicle counter for confirmed car/truck tracks."""

    def __init__(self) -> None:
        self.count_store = VehicleCountStore()
        self.counts = {VEHICLE_COUNT_KEY: self.count_store.total()}
        self.counted_memory: list[CountedVehicle] = []
        self.count_date = self.count_store.current_date

    def reset_memory(self) -> None:
        """Clear compatibility memory without resetting cumulative counts."""
        self.counted_memory.clear()

    def reset_daily_counts_if_needed(self) -> None:
        """Reset vehicle counts when the local calendar day changes, logging the previous total first."""
        current_date = self.count_store.today()
        if current_date == self.count_date:
            return

        previous_total = self.counts.get(VEHICLE_COUNT_KEY, 0)
        print(f"Daily vehicle counter reset: date={self.count_date.isoformat()}, vehicles_counted={previous_total}")
        restored_total = self.count_store.open_day(current_date)
        self.counts[VEHICLE_COUNT_KEY] = restored_total
        self.counted_memory.clear()
        self.count_date = current_date

    def age_counted_memory(self) -> None:
        """Keep the legacy compatibility hook without suppressing future vehicles."""
        self.counted_memory.clear()

    def update_counts(self, tracks: list[ObjectTrack], counting_zone_points: ZonePoints | None = None, counting_zone_enabled: bool = False) -> None:
        """Count each confirmed vehicle track once, optionally only when its center is inside the counting zone."""
        zone_points = counting_zone_points or []
        for track in tracks:
            if track.counted or track.class_name not in VEHICLE_SOURCE_CLASSES or track.hits < VEHICLE_COUNT_MIN_HITS:
                continue
            if counting_zone_enabled and not point_inside_zone(track.center, zone_points):
                continue
            self.counts[VEHICLE_COUNT_KEY] += 1
            speed_kmh = estimate_track_speed_kmh(track, zone_points)
            self.count_store.record_vehicle_event(
                track_id=track.track_id,
                class_name=track.class_name,
                total_after_event=self.counts[VEHICLE_COUNT_KEY],
                speed_kmh=speed_kmh,
            )
            track.counted = True

    def refresh_counted_memory(self, tracks: list[ObjectTrack]) -> None:
        """Keep the legacy compatibility hook without suppressing future vehicles."""
        self.counted_memory.clear()

    def is_recently_counted(self, class_name: str, bbox: tuple[int, int, int, int]) -> bool:
        """Return whether a matching vehicle was already counted recently."""
        return False

    def remember_counted(self, track: ObjectTrack) -> None:
        """Keep the legacy compatibility hook without suppressing future vehicles."""


def get_detection_class_name(detection: Detection) -> str:
    """Extract the class name from a WorldCam detection label."""
    return detection[4].rsplit(" ", 1)[0]


def are_compatible_track_classes(track_class_name: str, detection_class_name: str) -> bool:
    """Return whether a detection can update an existing track despite label jitter."""
    if track_class_name == detection_class_name:
        return True
    return track_class_name in VEHICLE_SOURCE_CLASSES and detection_class_name in VEHICLE_SOURCE_CLASSES


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


def should_replace_overlapping_track(candidate: ObjectTrack, kept_track: ObjectTrack) -> bool:
    """Return whether an overlapping candidate is more reliable than the kept track."""
    if candidate.counted != kept_track.counted:
        return candidate.counted
    if candidate.hits != kept_track.hits:
        return candidate.hits > kept_track.hits
    if candidate.confidence != kept_track.confidence:
        return candidate.confidence > kept_track.confidence
    return candidate.track_id < kept_track.track_id


class ObjectTracker:
    """Simple IoU/centroid tracker for all selected object detections."""

    def __init__(self) -> None:
        self.tracks: dict[int, ObjectTrack] = {}
        self.next_track_id = 1
        self.vehicle_counter = VehicleCounter()

    @property
    def vehicle_counts(self) -> dict[str, int]:
        """Return cumulative vehicle counts used by the display overlay."""
        return self.vehicle_counter.counts

    def reset(self) -> None:
        """Clear all active tracks and restart ID allocation."""
        if PERSON_TRACK_DEBUG and self.tracks:
            print(f"Tracking objects: reset active_ids={sorted(self.tracks)}")
        self.tracks.clear()
        self.next_track_id = 1
        self.vehicle_counter.reset_memory()

    def update(
        self,
        detections: list[Detection],
        counting_zone_points: ZonePoints | None = None,
        counting_zone_enabled: bool = False,
    ) -> list[ObjectTrack]:
        """Update tracks from the latest selected detections and return active tracks."""
        self.vehicle_counter.reset_daily_counts_if_needed()
        self.vehicle_counter.age_counted_memory()
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
                if not are_compatible_track_classes(track.class_name, class_name):
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
            track = ObjectTrack(track_id=track_id, bbox=(x1, y1, x2, y2), confidence=confidence, class_name=class_name)
            track.trail.append(track.center)
            self.tracks[track_id] = track

        removed_track_ids = []
        for track_id in list(unmatched_track_ids):
            track = self.tracks[track_id]
            track.age += 1
            if track.age > PERSON_TRACK_MAX_AGE:
                removed_track_ids.append(track_id)
                del self.tracks[track_id]

        self.suppress_overlapping_active_tracks()
        active_tracks = self.active_tracks()
        self.vehicle_counter.update_counts(active_tracks, counting_zone_points, counting_zone_enabled)
        self.vehicle_counter.refresh_counted_memory(active_tracks)

        if PERSON_TRACK_DEBUG:
            print(
                "Tracking objects: "
                f"detections={len(object_boxes)}, matched={len(matches)}, "
                f"new={len(object_boxes) - len(matched_detection_indexes)}, lost={len(removed_track_ids)}, "
                f"active_ids={sorted(self.tracks)}"
            )

        return active_tracks

    def active_tracks(self) -> list[ObjectTrack]:
        """Return active tracks sorted by stable ID."""
        return [self.tracks[track_id] for track_id in sorted(self.tracks)]

    def suppress_overlapping_active_tracks(self) -> None:
        """Remove duplicate active tracks whose boxes overlap too strongly."""
        kept_tracks: list[ObjectTrack] = []
        removed_track_ids: set[int] = set()

        for track in sorted(self.tracks.values(), key=lambda item: (item.counted, item.hits, item.confidence), reverse=True):
            duplicate_index = next(
                (
                    index
                    for index, kept_track in enumerate(kept_tracks)
                    if are_compatible_track_classes(track.class_name, kept_track.class_name)
                    and calculate_iou(track.bbox, kept_track.bbox) >= ACTIVE_TRACK_OVERLAP_IOU_THRESHOLD
                ),
                None,
            )
            if duplicate_index is None:
                kept_tracks.append(track)
                continue

            kept_track = kept_tracks[duplicate_index]
            if should_replace_overlapping_track(track, kept_track):
                track.counted = track.counted or kept_track.counted
                removed_track_ids.add(kept_track.track_id)
                kept_tracks[duplicate_index] = track
            else:
                kept_track.counted = kept_track.counted or track.counted
                removed_track_ids.add(track.track_id)

        for track_id in removed_track_ids:
            self.tracks.pop(track_id, None)


def draw_object_tracks(frame: np.ndarray, tracks: list[ObjectTrack], display_threshold: float = 0.5) -> None:
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
    """Draw the cumulative vehicle counter in the bottom-left corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    margin = 12
    line_height = 26
    labels = [f"Nb de vehicules aujourd'hui: {vehicle_counts.get(VEHICLE_COUNT_KEY, 0)}"]
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


# Compatibility aliases for external callers using the previous person-centric names.
PersonTrack = ObjectTrack
PersonTracker = ObjectTracker
draw_person_tracks = draw_object_tracks
