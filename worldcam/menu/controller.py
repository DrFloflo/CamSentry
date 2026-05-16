"""Menu process lifecycle, event consumption, and keyboard handlers."""

import multiprocessing as mp
import queue as queue_module

from worldcam.menu.constants import (
    MENU_EVENT_CLASS,
    MENU_EVENT_CLOSE,
    MENU_EVENT_CLOSED,
    MENU_EVENT_POSE,
    MENU_EVENT_SAHI,
    MENU_EVENT_TRACKING,
    MENU_EVENT_SEGMENTATION,
    MENU_EVENT_THRESHOLD,
)
from worldcam.menu.state import MenuState
from worldcam.menu.window import run_class_menu_process


def start_class_menu_window(class_names: list[str], selected_class_names: set[str], menu_state: MenuState) -> None:
    """Start the independent class menu process if it is not already running."""
    if menu_state.menu_process is not None and menu_state.menu_process.is_alive():
        menu_state.is_open = True
        return

    menu_state.is_open = True
    menu_process = mp.Process(
        target=run_class_menu_process,
        args=(
            class_names,
            set(selected_class_names),
            menu_state.pose_enabled,
            menu_state.sahi_enabled,
            menu_state.tracking_enabled,
            menu_state.segmentation_enabled,
            menu_state.display_threshold,
            menu_state.event_queue,
            menu_state.command_queue,
        ),
        daemon=True,
    )
    menu_state.menu_process = menu_process
    menu_process.start()


def close_class_menu_window(menu_state: MenuState) -> None:
    """Request the class menu process to close, then force it if needed."""
    menu_state.is_open = False
    process = menu_state.menu_process
    if process is None:
        return

    if process.is_alive():
        menu_state.command_queue.put(MENU_EVENT_CLOSE)
        process.join(timeout=1.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
    menu_state.menu_process = None


def consume_menu_changes(menu_state: MenuState, selected_class_names: set[str]) -> tuple[bool, bool, bool, bool, bool, bool]:
    """Apply pending menu-process events and return change flags."""
    try:
        while True:
            event_name, payload = menu_state.event_queue.get_nowait()
            if event_name == MENU_EVENT_CLASS:
                class_name, is_selected = payload
                if is_selected:
                    selected_class_names.add(class_name)
                else:
                    selected_class_names.discard(class_name)
                menu_state.class_selection_changed = True
            elif event_name == MENU_EVENT_POSE:
                menu_state.pose_enabled = bool(payload)
                menu_state.pose_toggled = True
            elif event_name == MENU_EVENT_SAHI:
                menu_state.sahi_enabled = bool(payload)
                menu_state.sahi_toggled = True
            elif event_name == MENU_EVENT_TRACKING:
                menu_state.tracking_enabled = bool(payload)
                menu_state.tracking_toggled = True
            elif event_name == MENU_EVENT_SEGMENTATION:
                menu_state.segmentation_enabled = bool(payload)
                menu_state.segmentation_toggled = True
            elif event_name == MENU_EVENT_THRESHOLD:
                menu_state.display_threshold = float(payload)
                menu_state.threshold_changed = True
            elif event_name == MENU_EVENT_CLOSED:
                menu_state.is_open = False
    except queue_module.Empty:
        pass

    if menu_state.menu_process is not None and not menu_state.menu_process.is_alive():
        menu_state.menu_process.join(timeout=0.1)
        menu_state.menu_process = None
        menu_state.is_open = False

    changes = (
        menu_state.class_selection_changed,
        menu_state.pose_toggled,
        menu_state.sahi_toggled,
        menu_state.tracking_toggled,
        menu_state.segmentation_toggled,
        menu_state.threshold_changed,
    )
    menu_state.class_selection_changed = False
    menu_state.pose_toggled = False
    menu_state.sahi_toggled = False
    menu_state.tracking_toggled = False
    menu_state.segmentation_toggled = False
    menu_state.threshold_changed = False
    return changes


def snapshot_menu_state(menu_state: MenuState, selected_class_names: set[str]) -> tuple[set[str], bool, bool, bool, bool, float]:
    """Return a stable snapshot of selected classes, feature toggles, and display threshold."""
    return (
        set(selected_class_names),
        menu_state.pose_enabled,
        menu_state.sahi_enabled,
        menu_state.tracking_enabled,
        menu_state.segmentation_enabled,
        menu_state.display_threshold,
    )


def handle_class_menu_key(
    key: int,
    class_names: list[str],
    selected_class_names: set[str],
    menu_state: MenuState,
) -> tuple[bool, bool, bool, bool, bool, bool]:
    """Update menu state and return class-change, pose-toggle, SAHI-toggle, tracking-toggle, segmentation-toggle, and threshold-change flags."""
    if key == ord("m"):
        if menu_state.is_open:
            close_class_menu_window(menu_state)
        else:
            start_class_menu_window(class_names, selected_class_names, menu_state)
        return False, False, False, False, False, False

    if not menu_state.is_open:
        return False, False, False, False, False, False

    if key == ord("p"):
        menu_state.pose_enabled = not menu_state.pose_enabled
        return False, True, False, False, False, False

    if key == ord("a"):
        menu_state.sahi_enabled = not menu_state.sahi_enabled
        return False, False, True, False, False, False

    if key == ord("t"):
        menu_state.tracking_enabled = not menu_state.tracking_enabled
        return False, False, False, True, False, False

    if key == ord("g"):
        menu_state.segmentation_enabled = not menu_state.segmentation_enabled
        return False, False, False, False, True, False

    return False, False, False, False, False, False
