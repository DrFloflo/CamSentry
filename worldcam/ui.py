"""On-screen display and keyboard interaction helpers."""

import cv2
import numpy as np

from worldcam.menu import (
    MenuChanges,
    MenuSnapshot,
    MenuState,
    close_class_menu_window,
    consume_menu_changes,
    handle_class_menu_key,
    snapshot_menu_state,
    start_class_menu_window,
)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Draw the current FPS in yellow at the top-right corner."""
    label = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    margin = 12
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_width, text_height = text_size
    x = max(frame.shape[1] - text_width - margin, margin)
    y = margin + text_height
    cv2.putText(frame, label, (x, y), font, font_scale, (0, 255, 255), thickness)


def draw_stream_counter(frame: np.ndarray, stream_index: int, stream_total: int) -> None:
    """Draw the active stream number in yellow at the bottom-right corner."""
    label = f"Stream {stream_index + 1}/{stream_total}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    margin = 12
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_width, text_height = text_size
    x = max(frame.shape[1] - text_width - margin, margin)
    y = max(frame.shape[0] - margin, margin + text_height)
    cv2.putText(frame, label, (x, y), font, font_scale, (0, 255, 255), thickness)


__all__ = [
    "MenuChanges",
    "MenuSnapshot",
    "MenuState",
    "close_class_menu_window",
    "consume_menu_changes",
    "draw_fps",
    "draw_stream_counter",
    "handle_class_menu_key",
    "snapshot_menu_state",
    "start_class_menu_window",
]
