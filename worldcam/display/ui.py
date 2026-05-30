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




__all__ = [
    "MenuChanges",
    "MenuSnapshot",
    "MenuState",
    "close_class_menu_window",
    "consume_menu_changes",
    "handle_class_menu_key",
    "snapshot_menu_state",
    "start_class_menu_window",
]
