"""Application orchestration for the WorldCam video analysis loop."""

import time

import cv2
from ultralytics import YOLO

from worldcam.analysis_runtime import update_runtime_analysis
from worldcam.config import DEFAULT_CLASS_NAMES, FRAME_SKIP, STREAM_URLS
from worldcam.display_runtime import cleanup_resources, draw_overlay, throttle_display
from worldcam.models import load_yolo_model
from worldcam.runtime import (
    RuntimeState,
    register_frame_received,
    register_stream_read,
    register_stream_read_failure,
    reset_analysis_state,
    reset_stream_statistics,
)
from worldcam.streaming import configure_ffmpeg_http_headers, print_videoio_diagnostics
from worldcam.stream_control import open_stream_resources, reconnect_current_stream, switch_stream
from worldcam.tracking import PersonTracker
from worldcam.ui import MenuState, close_class_menu_window, consume_menu_changes, handle_class_menu_key, snapshot_menu_state

KEY_LEFT_VALUES = {81, 2424832}
KEY_RIGHT_VALUES = {83, 2555904}
MAIN_WINDOW_NAME = "Analyse Image - Earth cam"
STREAM_READ_TIMEOUT = 1.0
MAX_STREAM_READ_FAILURES = 5


def build_class_selection(model: YOLO) -> tuple[list[str], set[str]]:
    """Build the full class list and initial enabled class selection."""
    class_names = [model.names[index] for index in sorted(model.names)]
    selected_class_names = {class_name for class_name in DEFAULT_CLASS_NAMES if class_name in class_names}
    return class_names, selected_class_names


def handle_menu_changes(
    key: int,
    class_names: list[str],
    selected_class_names: set[str],
    menu_state: MenuState,
    runtime: RuntimeState,
    person_tracker: PersonTracker,
) -> None:
    """Apply keyboard and menu-process changes, resetting affected cached analysis state."""
    (
        keyboard_class_changed,
        keyboard_pose_toggled,
        keyboard_sahi_toggled,
        keyboard_tracking_toggled,
        keyboard_segmentation_toggled,
        _keyboard_threshold_changed,
    ) = handle_class_menu_key(
        key,
        class_names,
        selected_class_names,
        menu_state,
    )
    (
        mouse_class_changed,
        mouse_pose_toggled,
        mouse_sahi_toggled,
        mouse_tracking_toggled,
        mouse_segmentation_toggled,
        _mouse_threshold_changed,
    ) = consume_menu_changes(menu_state, selected_class_names)

    class_selection_changed = keyboard_class_changed or mouse_class_changed
    pose_toggled = keyboard_pose_toggled or mouse_pose_toggled
    sahi_toggled = keyboard_sahi_toggled or mouse_sahi_toggled
    tracking_toggled = keyboard_tracking_toggled or mouse_tracking_toggled
    segmentation_toggled = keyboard_segmentation_toggled or mouse_segmentation_toggled

    if class_selection_changed or sahi_toggled or tracking_toggled or segmentation_toggled:
        runtime.latest_detections = []
        runtime.latest_segmentations = []
        runtime.latest_person_tracks = []
        person_tracker.reset()
    if pose_toggled and not menu_state.pose_enabled:
        runtime.latest_poses = []


def main() -> None:
    """Run the WorldCam analysis application."""
    configure_ffmpeg_http_headers()
    if not STREAM_URLS:
        print("Erreur : aucun stream configuré dans STREAM_URLS.")
        return

    stream_total = len(STREAM_URLS)
    stream_index = 0
    print_videoio_diagnostics(STREAM_URLS[stream_index])

    model, device = load_yolo_model()
    pose_model = None
    segmentation_model = None

    try:
        resources = open_stream_resources(STREAM_URLS[stream_index], stream_index, stream_total)
    except RuntimeError as exc:
        print(f"Erreur : {exc}")
        return

    runtime = RuntimeState()
    person_tracker = PersonTracker()
    class_names, selected_class_names = build_class_selection(model)
    menu_state = MenuState()
    cv2.namedWindow(MAIN_WINDOW_NAME)

    try:
        while True:
            read_started_at = time.perf_counter()
            ret, frame = resources.stream_reader.read(timeout=STREAM_READ_TIMEOUT)
            read_duration = time.perf_counter() - read_started_at
            register_stream_read(runtime, read_duration)

            if not ret:
                if not register_stream_read_failure(runtime, MAX_STREAM_READ_FAILURES):
                    continue

                print("Reconnexion automatique du flux...")
                try:
                    resources = reconnect_current_stream(resources, stream_index, stream_total)
                    runtime.stream_read_failures = 0
                    reset_analysis_state(runtime, person_tracker)
                    continue
                except RuntimeError as exc:
                    print(f"Erreur pendant la reconnexion automatique : {exc}")
                    break

            register_frame_received(runtime, read_duration)

            (
                selected_snapshot,
                pose_enabled,
                sahi_enabled,
                tracking_enabled,
                segmentation_enabled,
                display_threshold,
            ) = snapshot_menu_state(menu_state, selected_class_names)
            if runtime.frame_count % FRAME_SKIP == 0:
                pose_model, segmentation_model = update_runtime_analysis(
                    runtime,
                    frame,
                    model,
                    pose_model,
                    segmentation_model,
                    device,
                    selected_snapshot,
                    pose_enabled,
                    segmentation_enabled,
                    sahi_enabled,
                    tracking_enabled,
                    person_tracker,
                )

            draw_overlay(
                frame,
                runtime.current_fps,
                runtime.latest_detections,
                runtime.latest_poses,
                runtime.latest_segmentations,
                display_threshold,
                runtime.latest_person_tracks,
                stream_index,
                stream_total,
                person_tracker.vehicle_counts,
            )
            runtime.next_frame_at = throttle_display(runtime.next_frame_at)
            cv2.imshow(MAIN_WINDOW_NAME, frame)

            key = cv2.waitKeyEx(1)
            if key == ord("q"):
                break
            if key in KEY_LEFT_VALUES or key in KEY_RIGHT_VALUES:
                step = -1 if key in KEY_LEFT_VALUES else 1
                resources, stream_index = switch_stream(resources, stream_index, step, stream_total)
                reset_analysis_state(runtime, person_tracker)
                reset_stream_statistics(runtime)
                continue
            handle_menu_changes(
                key,
                class_names,
                selected_class_names,
                menu_state,
                runtime,
                person_tracker,
            )
    finally:
        close_class_menu_window(menu_state)
        resources.stream_reader.stop()
        cleanup_resources(resources.cap, resources.ffmpeg_process, model, pose_model, segmentation_model, device)
