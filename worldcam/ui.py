"""On-screen display and keyboard interaction helpers."""

from dataclasses import dataclass

import cv2
import numpy as np

from worldcam.config import (
    MENU_BACKGROUND_COLOR,
    MENU_ENABLED_COLOR,
    MENU_PAGE_SIZE,
    MENU_SELECTED_COLOR,
    MENU_TEXT_COLOR,
)


@dataclass
class MenuState:
    """Mutable state for the YOLO class selection menu."""

    is_open: bool = False
    index: int = 0
    scroll: int = 0
    pose_enabled: bool = False


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


def draw_class_menu(
    frame: np.ndarray,
    class_names: list[str],
    selected_class_names: set[str],
    menu_state: MenuState,
) -> None:
    """Draw a small on-screen class selection menu."""
    menu_x = 16
    menu_y = 16
    menu_width = 360
    line_height = 24
    menu_height = 116 + (MENU_PAGE_SIZE * line_height)

    cv2.rectangle(frame, (menu_x, menu_y), (menu_x + menu_width, menu_y + menu_height), MENU_BACKGROUND_COLOR, -1)
    cv2.rectangle(frame, (menu_x, menu_y), (menu_x + menu_width, menu_y + menu_height), MENU_SELECTED_COLOR, 1)
    cv2.putText(frame, "Menu analyse - M fermer", (menu_x + 12, menu_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MENU_SELECTED_COLOR, 2)
    cv2.putText(frame, "Haut/Bas: classes | Espace: cocher | P: pose", (menu_x + 12, menu_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, MENU_TEXT_COLOR, 1)
    pose_label = "[x] Pose" if menu_state.pose_enabled else "[ ] Pose"
    pose_color = MENU_ENABLED_COLOR if menu_state.pose_enabled else MENU_TEXT_COLOR
    cv2.putText(frame, pose_label, (menu_x + 12, menu_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, pose_color, 1)

    visible_classes = class_names[menu_state.scroll:menu_state.scroll + MENU_PAGE_SIZE]
    for visible_index, class_name in enumerate(visible_classes):
        class_index = menu_state.scroll + visible_index
        y = menu_y + 108 + (visible_index * line_height)
        is_current = class_index == menu_state.index
        is_enabled = class_name in selected_class_names
        cursor = ">" if is_current else " "
        checkbox = "[x]" if is_enabled else "[ ]"
        color = MENU_SELECTED_COLOR if is_current else MENU_ENABLED_COLOR if is_enabled else MENU_TEXT_COLOR
        cv2.putText(frame, f"{cursor} {checkbox} {class_name}", (menu_x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def handle_class_menu_key(
    key: int,
    class_names: list[str],
    selected_class_names: set[str],
    menu_state: MenuState,
) -> tuple[bool, bool]:
    """Update menu state and return class-change and pose-toggle flags."""
    if key == ord("m"):
        menu_state.is_open = not menu_state.is_open
        return False, False

    if not menu_state.is_open:
        return False, False

    if key == ord("p"):
        menu_state.pose_enabled = not menu_state.pose_enabled
        return False, True

    if key in (82, ord("z"), ord("w")):
        menu_state.index = max(0, menu_state.index - 1)
        if menu_state.index < menu_state.scroll:
            menu_state.scroll = menu_state.index
    elif key in (84, ord("s")):
        menu_state.index = min(len(class_names) - 1, menu_state.index + 1)
        if menu_state.index >= menu_state.scroll + MENU_PAGE_SIZE:
            menu_state.scroll = menu_state.index - MENU_PAGE_SIZE + 1
    elif key == ord(" "):
        class_name = class_names[menu_state.index]
        if class_name in selected_class_names:
            selected_class_names.remove(class_name)
        else:
            selected_class_names.add(class_name)
        return True, False

    return False, False
