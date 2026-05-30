"""Application orchestration for the WorldCam video analysis loop."""

import argparse
import time

import cv2
from ultralytics import YOLO

from worldcam.analysis.analysis_runtime import update_runtime_analysis
from worldcam.core.config import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_MODEL_KEY,
    FRAME_SKIP,
    MAX_STREAM_READ_FAILURES,
    SAHI_ENABLED,
    STREAM_READ_TIMEOUT_SECONDS,
    STREAM_URL,
    EXCLUSION_ZONE_POINTS,
)
from worldcam.analysis.counting_zone import CountingZoneEditor, CountingZoneState
from worldcam.display.display_runtime import cleanup_resources, draw_overlay, throttle_display
from worldcam.core.models import load_yolo_model
from worldcam.core.runtime import (
    RuntimeState,
    register_frame_received,
    register_stream_read,
    register_stream_read_failure,
    reset_analysis_state,
)
from worldcam.stream.streaming import configure_ffmpeg_http_headers, print_videoio_diagnostics
from worldcam.stream.stream_control import open_stream_resources_with_retry, reconnect_current_stream
from worldcam.analysis.tracking import ObjectTracker
from worldcam.display.ui import MenuState, close_class_menu_window, consume_menu_changes, handle_class_menu_key, snapshot_menu_state
from worldcam.stream.web_stream import WebStreamServer, start_web_stream_server

MAIN_WINDOW_NAME = "Analyse Image - Earth cam"


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
    object_tracker: ObjectTracker,
) -> None:
    """Apply keyboard and menu-process changes, resetting affected cached analysis state."""
    keyboard_changes = handle_class_menu_key(
        key,
        class_names,
        selected_class_names,
        menu_state,
    )
    menu_changes = consume_menu_changes(menu_state, selected_class_names)
    changes = keyboard_changes.merge(menu_changes)

    if (
        changes.class_selection_changed
        or changes.sahi_toggled
        or changes.tracking_toggled
        or changes.segmentation_toggled
        or changes.exclusion_zone_processing_toggled
    ):
        runtime.latest_detections = []
        runtime.latest_segmentations = []
        runtime.latest_object_tracks = []
        object_tracker.reset()
    if changes.pose_toggled and not menu_state.pose_enabled:
        runtime.latest_poses = []


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse WorldCam command-line options."""
    parser = argparse.ArgumentParser(description="Run the WorldCam analysis application.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_KEY,
        help=(
            "YOLO model key used from models/ without the 'yolo' prefix "
            "(examples: 26l loads yolo26l.*, 26m loads yolo26m.*, with matching -pose/-seg variants)."
        ),
    )
    parser.add_argument(
        "--sahi",
        action="store_true",
        default=SAHI_ENABLED,
        help="Enable SAHI sliced inference at startup.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable OpenCV windows and publish the annotated stream over HTTP.",
    )
    parser.add_argument(
        "--stream-host",
        default="0.0.0.0",
        help="Host/IP address used by the headless HTTP stream server.",
    )
    parser.add_argument(
        "--stream-port",
        type=int,
        default=8080,
        help="TCP port used by the headless HTTP stream server.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the WorldCam analysis application."""
    args = parse_args(argv)
    configure_ffmpeg_http_headers()
    stream_total = 1
    stream_index = 0
    print_videoio_diagnostics(STREAM_URL)

    model_key = args.model.strip().removeprefix("yolo")
    model, device = load_yolo_model(model_key)
    pose_model = None
    segmentation_model = None

    resources = open_stream_resources_with_retry(STREAM_URL, stream_index, stream_total)

    runtime = RuntimeState()
    object_tracker = ObjectTracker()
    class_names, selected_class_names = build_class_selection(model)
    menu_state = MenuState(sahi_enabled=args.sahi)
    counting_zone_editor = CountingZoneEditor()
    exclusion_zone_editor = CountingZoneEditor(CountingZoneState(points=list(EXCLUSION_ZONE_POINTS), print_name="exclusion_zone_points"))
    web_stream_server: WebStreamServer | None = None

    if args.headless:
        web_stream_server = start_web_stream_server(args.stream_host, args.stream_port)
    else:
        cv2.namedWindow(MAIN_WINDOW_NAME)
        cv2.setMouseCallback(MAIN_WINDOW_NAME, lambda event, x, y, flags, param: (
            exclusion_zone_editor.mouse_callback(event, x, y, flags, param)
            if exclusion_zone_editor.state.edit_enabled
            else counting_zone_editor.mouse_callback(event, x, y, flags, param)
        ))

    try:
        while True:
            read_started_at = time.perf_counter()
            ret, frame = resources.stream_reader.read(timeout=STREAM_READ_TIMEOUT_SECONDS)
            read_duration = time.perf_counter() - read_started_at
            reader_stats = resources.stream_reader.stats()
            register_stream_read(runtime, read_duration)

            if not ret:
                if not register_stream_read_failure(runtime, MAX_STREAM_READ_FAILURES):
                    continue

                print("Reconnexion en cours...")
                resources = reconnect_current_stream(resources, stream_index, stream_total)
                runtime.stream_read_failures = 0
                reset_analysis_state(runtime, object_tracker)
                continue

            if reader_stats.stale:
                if not register_stream_read_failure(runtime, MAX_STREAM_READ_FAILURES):
                    continue

                print("Flux obsolète: reconnexion en cours...")
                resources = reconnect_current_stream(resources, stream_index, stream_total)
                runtime.stream_read_failures = 0
                reset_analysis_state(runtime, object_tracker)
                continue

            register_frame_received(runtime, read_duration, reader_stats.latest_frame_age_seconds)

            counting_zone_editor.update_frame_size(frame)
            exclusion_zone_editor.update_frame_size(frame)
            menu_snapshot = snapshot_menu_state(menu_state, selected_class_names)
            counting_zone_editor.set_enabled(menu_snapshot.counting_zone_enabled)
            counting_zone_editor.set_edit_enabled(menu_snapshot.counting_zone_edit_enabled and not menu_snapshot.exclusion_zone_edit_enabled)
            exclusion_zone_editor.set_enabled(menu_snapshot.exclusion_zone_display_enabled)
            exclusion_zone_editor.set_edit_enabled(menu_snapshot.exclusion_zone_edit_enabled)
            if runtime.frame_count % FRAME_SKIP == 0:
                pose_model, segmentation_model = update_runtime_analysis(
                    runtime,
                    frame,
                    model,
                    pose_model,
                    segmentation_model,
                    device,
                    menu_snapshot.selected_class_names,
                    menu_snapshot.pose_enabled,
                    menu_snapshot.segmentation_enabled,
                    menu_snapshot.sahi_enabled,
                    menu_snapshot.tracking_enabled,
                    object_tracker,
                    counting_zone_editor.points,
                    menu_snapshot.counting_zone_enabled,
                    model_key,
                    exclusion_zone_editor.points,
                    menu_snapshot.exclusion_zone_processing_enabled and not args.headless,
                )

            draw_overlay(
                frame,
                runtime.latest_detections,
                runtime.latest_poses,
                runtime.latest_segmentations,
                menu_snapshot.display_threshold,
                runtime.latest_object_tracks,
                stream_index,
                stream_total,
                object_tracker.vehicle_counts,
                menu_snapshot.tracking_enabled,
                counting_zone_editor.points,
                menu_snapshot.counting_zone_enabled,
                menu_snapshot.counting_zone_edit_enabled and not menu_snapshot.exclusion_zone_edit_enabled,
                exclusion_zone_editor.points,
                menu_snapshot.exclusion_zone_display_enabled and not args.headless,
                menu_snapshot.exclusion_zone_processing_enabled and not args.headless,
                menu_snapshot.exclusion_zone_edit_enabled and not args.headless,
            )
            if web_stream_server is not None:
                web_stream_server.update_frame(frame)

            runtime.next_frame_at = throttle_display(runtime.next_frame_at)
            key = -1
            if not args.headless:
                cv2.imshow(MAIN_WINDOW_NAME, frame)
                key = cv2.waitKeyEx(1)
                if key == ord("q"):
                    break
            handle_menu_changes(
                key,
                class_names,
                selected_class_names,
                menu_state,
                runtime,
                object_tracker,
            )
    except KeyboardInterrupt:
        print("Arrêt demandé par l'utilisateur.")
    finally:
        if web_stream_server is not None:
            web_stream_server.stop()
        close_class_menu_window(menu_state)
        resources.stream_reader.stop()
        cleanup_resources(resources.cap, resources.ffmpeg_process, model, pose_model, segmentation_model, device)
