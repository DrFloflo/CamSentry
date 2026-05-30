"""Menu process lifecycle, event consumption, and keyboard handlers."""

import multiprocessing as mp
import queue as queue_module

from worldcam.menu.constants import (
    MENU_EVENT_CLASS,
    MENU_EVENT_CLOSE,
    MENU_EVENT_CLOSED,
    MENU_EVENT_COUNTING_ZONE_EDIT,
    MENU_EVENT_COUNTING_ZONE_ENABLED,
    MENU_EVENT_EXCLUSION_ZONE_DISPLAY,
    MENU_EVENT_EXCLUSION_ZONE_EDIT,
    MENU_EVENT_EXCLUSION_ZONE_PROCESSING,
    MENU_EVENT_POSE,
    MENU_EVENT_SAHI,
    MENU_EVENT_TRACKING,
    MENU_EVENT_SEGMENTATION,
    MENU_EVENT_THRESHOLD,
)
from worldcam.menu.state import MenuChanges, MenuSnapshot, MenuState
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
            menu_state.counting_zone_enabled,
            menu_state.counting_zone_edit_enabled,
            menu_state.exclusion_zone_display_enabled,
            menu_state.exclusion_zone_processing_enabled,
            menu_state.exclusion_zone_edit_enabled,
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


def consume_menu_changes(menu_state: MenuState, selected_class_names: set[str]) -> MenuChanges:
    """Apply pending menu-process events and return named change flags."""
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
            elif event_name == MENU_EVENT_COUNTING_ZONE_ENABLED:
                menu_state.counting_zone_enabled = bool(payload)
                menu_state.counting_zone_toggled = True
            elif event_name == MENU_EVENT_COUNTING_ZONE_EDIT:
                menu_state.counting_zone_edit_enabled = bool(payload)
                menu_state.counting_zone_edit_toggled = True
            elif event_name == MENU_EVENT_EXCLUSION_ZONE_DISPLAY:
                menu_state.exclusion_zone_display_enabled = bool(payload)
                menu_state.exclusion_zone_display_toggled = True
            elif event_name == MENU_EVENT_EXCLUSION_ZONE_PROCESSING:
                menu_state.exclusion_zone_processing_enabled = bool(payload)
                menu_state.exclusion_zone_processing_toggled = True
            elif event_name == MENU_EVENT_EXCLUSION_ZONE_EDIT:
                menu_state.exclusion_zone_edit_enabled = bool(payload)
                menu_state.exclusion_zone_edit_toggled = True
            elif event_name == MENU_EVENT_CLOSED:
                menu_state.is_open = False
    except queue_module.Empty:
        pass

    if menu_state.menu_process is not None and not menu_state.menu_process.is_alive():
        menu_state.menu_process.join(timeout=0.1)
        menu_state.menu_process = None
        menu_state.is_open = False

    changes = MenuChanges(
        class_selection_changed=menu_state.class_selection_changed,
        pose_toggled=menu_state.pose_toggled,
        sahi_toggled=menu_state.sahi_toggled,
        tracking_toggled=menu_state.tracking_toggled,
        segmentation_toggled=menu_state.segmentation_toggled,
        threshold_changed=menu_state.threshold_changed,
        counting_zone_toggled=menu_state.counting_zone_toggled,
        counting_zone_edit_toggled=menu_state.counting_zone_edit_toggled,
        exclusion_zone_display_toggled=menu_state.exclusion_zone_display_toggled,
        exclusion_zone_processing_toggled=menu_state.exclusion_zone_processing_toggled,
        exclusion_zone_edit_toggled=menu_state.exclusion_zone_edit_toggled,
    )
    menu_state.class_selection_changed = False
    menu_state.pose_toggled = False
    menu_state.sahi_toggled = False
    menu_state.tracking_toggled = False
    menu_state.segmentation_toggled = False
    menu_state.threshold_changed = False
    menu_state.counting_zone_toggled = False
    menu_state.counting_zone_edit_toggled = False
    menu_state.exclusion_zone_display_toggled = False
    menu_state.exclusion_zone_processing_toggled = False
    menu_state.exclusion_zone_edit_toggled = False
    return changes


def snapshot_menu_state(menu_state: MenuState, selected_class_names: set[str]) -> MenuSnapshot:
    """Return a stable snapshot of selected classes, feature toggles, and display threshold."""
    return MenuSnapshot(
        selected_class_names=set(selected_class_names),
        pose_enabled=menu_state.pose_enabled,
        sahi_enabled=menu_state.sahi_enabled,
        tracking_enabled=menu_state.tracking_enabled,
        segmentation_enabled=menu_state.segmentation_enabled,
        display_threshold=menu_state.display_threshold,
        counting_zone_enabled=menu_state.counting_zone_enabled,
        counting_zone_edit_enabled=menu_state.counting_zone_edit_enabled,
        exclusion_zone_display_enabled=menu_state.exclusion_zone_display_enabled,
        exclusion_zone_processing_enabled=menu_state.exclusion_zone_processing_enabled,
        exclusion_zone_edit_enabled=menu_state.exclusion_zone_edit_enabled,
    )


def handle_class_menu_key(
    key: int,
    class_names: list[str],
    selected_class_names: set[str],
    menu_state: MenuState,
) -> MenuChanges:
    """Update menu state and return named change flags."""
    if key == ord("m"):
        if menu_state.is_open:
            close_class_menu_window(menu_state)
        else:
            start_class_menu_window(class_names, selected_class_names, menu_state)
        return MenuChanges()

    if not menu_state.is_open:
        return MenuChanges()

    if key == ord("p"):
        menu_state.pose_enabled = not menu_state.pose_enabled
        return MenuChanges(pose_toggled=True)

    if key == ord("a"):
        menu_state.sahi_enabled = not menu_state.sahi_enabled
        return MenuChanges(sahi_toggled=True)

    if key == ord("t"):
        menu_state.tracking_enabled = not menu_state.tracking_enabled
        return MenuChanges(tracking_toggled=True)

    if key == ord("g"):
        menu_state.segmentation_enabled = not menu_state.segmentation_enabled
        return MenuChanges(segmentation_toggled=True)

    if key == ord("z"):
        menu_state.counting_zone_enabled = not menu_state.counting_zone_enabled
        return MenuChanges(counting_zone_toggled=True)

    if key == ord("e"):
        menu_state.counting_zone_edit_enabled = not menu_state.counting_zone_edit_enabled
        return MenuChanges(counting_zone_edit_toggled=True)

    if key == ord("x"):
        menu_state.exclusion_zone_display_enabled = not menu_state.exclusion_zone_display_enabled
        return MenuChanges(exclusion_zone_display_toggled=True)

    if key == ord("c"):
        menu_state.exclusion_zone_processing_enabled = not menu_state.exclusion_zone_processing_enabled
        return MenuChanges(exclusion_zone_processing_toggled=True)

    if key == ord("v"):
        menu_state.exclusion_zone_edit_enabled = not menu_state.exclusion_zone_edit_enabled
        return MenuChanges(exclusion_zone_edit_toggled=True)

    return MenuChanges()
