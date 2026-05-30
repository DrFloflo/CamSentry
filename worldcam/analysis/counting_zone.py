"""Interactive counting-zone free-point editor for the OpenCV display."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from worldcam.core.config import COUNTING_ZONE_HANDLE_RADIUS, COUNTING_ZONE_POINTS

Point = tuple[int, int]
ZonePoints = list[Point]


@dataclass
class CountingZoneState:
    """Mutable free-point zone state shared by the mouse editor and display overlay."""

    enabled: bool = False
    edit_enabled: bool = False
    points: ZonePoints = field(default_factory=lambda: list(COUNTING_ZONE_POINTS))
    frame_size: tuple[int, int] | None = None


class CountingZoneEditor:
    """Handle mouse-based creation, movement, and editing of free zone points."""

    def __init__(self, state: CountingZoneState | None = None) -> None:
        self.state = state or CountingZoneState()
        self._drag_mode: str | None = None
        self._active_point_index: int | None = None
        self._drag_start: Point | None = None
        self._drag_start_points: ZonePoints | None = None
        self._last_printed_points: tuple[Point, ...] | None = None

    @property
    def points(self) -> ZonePoints:
        """Return a copy of the current bounded zone points."""
        return list(self.state.points)

    def update_frame_size(self, frame) -> None:
        """Track the current frame size and keep every point bounded."""
        height, width = frame.shape[:2]
        self.state.frame_size = (width, height)
        if self.state.points:
            self._set_points(self.state.points, print_change=False)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or hide the counting-zone overlay."""
        self.state.enabled = enabled

    def set_edit_enabled(self, enabled: bool) -> None:
        """Enable or disable mouse editing for the counting-zone points."""
        self.state.edit_enabled = enabled
        if not enabled:
            self._reset_drag()

    def mouse_callback(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        """OpenCV mouse callback for point creation, movement, and deletion."""
        if self.state.frame_size is None or not self.state.edit_enabled:
            return

        point = self._bounded_point(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._start_left_drag(point)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag_mode is not None:
            self._update_drag(point)
        elif event == cv2.EVENT_LBUTTONUP and self._drag_mode is not None:
            self._update_drag(point)
            self._reset_drag()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._delete_point_at(point)

    def _start_left_drag(self, point: Point) -> None:
        point_index = self._hit_test_point(point)
        self._drag_start = point
        self._drag_start_points = self.points

        if point_index is not None:
            self._drag_mode = "point"
            self._active_point_index = point_index
            return

        if len(self.state.points) >= 3 and self._point_inside_polygon(point):
            self._drag_mode = "move"
            self._active_point_index = None
            return

        self.state.points.append(point)
        self._set_points(self.state.points)
        self._drag_mode = "point"
        self._active_point_index = len(self.state.points) - 1
        self._drag_start_points = self.points

    def _update_drag(self, point: Point) -> None:
        if self._drag_mode == "point" and self._active_point_index is not None:
            points = self.points
            if 0 <= self._active_point_index < len(points):
                points[self._active_point_index] = point
                self._set_points(points)
            return

        if self._drag_mode == "move" and self._drag_start is not None and self._drag_start_points is not None:
            dx = point[0] - self._drag_start[0]
            dy = point[1] - self._drag_start[1]
            self._set_points(self._move_points(self._drag_start_points, dx, dy))

    def _delete_point_at(self, point: Point) -> None:
        point_index = self._hit_test_point(point)
        if point_index is None:
            return
        points = self.points
        del points[point_index]
        self._set_points(points)
        self._reset_drag()

    def _set_points(self, points: ZonePoints, print_change: bool = True) -> None:
        bounded_points = [self._bounded_point(x, y) for x, y in points]
        self.state.points = bounded_points
        printable_points = tuple(bounded_points)
        if print_change and printable_points != self._last_printed_points:
            self._last_printed_points = printable_points
            print(f"counting_zone_points={list(printable_points)}")

    def _move_points(self, points: ZonePoints, dx: int, dy: int) -> ZonePoints:
        if not points:
            return []

        min_x = min(x for x, _y in points)
        max_x = max(x for x, _y in points)
        min_y = min(y for _x, y in points)
        max_y = max(y for _x, y in points)
        bounded_dx = min(max(dx, -min_x), self._max_x - max_x)
        bounded_dy = min(max(dy, -min_y), self._max_y - max_y)
        return [(x + bounded_dx, y + bounded_dy) for x, y in points]

    def _hit_test_point(self, point: Point) -> int | None:
        px, py = point
        for index, (x, y) in enumerate(self.state.points):
            if abs(px - x) <= COUNTING_ZONE_HANDLE_RADIUS and abs(py - y) <= COUNTING_ZONE_HANDLE_RADIUS:
                return index
        return None

    def _point_inside_polygon(self, point: Point) -> bool:
        contour = self._opencv_contour()
        if contour is None:
            return False
        return cv2.pointPolygonTest(contour, point, False) >= 0

    def _opencv_contour(self):
        if len(self.state.points) < 3:
            return None
        return np.array(self.state.points, dtype=np.int32)

    def _bounded_point(self, x: int, y: int) -> Point:
        return (min(max(0, int(x)), self._max_x), min(max(0, int(y)), self._max_y))

    def _reset_drag(self) -> None:
        self._drag_mode = None
        self._active_point_index = None
        self._drag_start = None
        self._drag_start_points = None

    @property
    def _max_x(self) -> int:
        return max(0, (self.state.frame_size or (1, 1))[0] - 1)

    @property
    def _max_y(self) -> int:
        return max(0, (self.state.frame_size or (1, 1))[1] - 1)


def point_inside_zone(point: Point, zone_points: ZonePoints) -> bool:
    """Return whether a point is inside a closed counting-zone polygon."""
    if len(zone_points) < 3:
        return False

    contour = np.array(zone_points, dtype=np.int32)
    return cv2.pointPolygonTest(contour, point, False) >= 0
