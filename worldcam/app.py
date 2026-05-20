"""Application orchestration for the WorldCam video analysis loop."""

import subprocess
import time

import cv2
import torch
from ultralytics import YOLO

from worldcam.config import (
    DEFAULT_CLASS_NAMES,
    FRAME_INTERVAL,
    FRAME_SKIP,
    READ_WARN_SECONDS,
    STATS_LOG_SECONDS,
    STREAM_URLS,
    TARGET_FPS,
)
from worldcam.detection import Detection, draw_yolo_detections, run_sahi_analysis, run_yolo_analysis
from worldcam.models import load_pose_model, load_segmentation_model, load_yolo_model
from worldcam.pose import Pose, draw_pose_detections, run_pose_analysis
from worldcam.person_photo import close_person_photo_window, handle_main_window_mouse, show_clicked_person_photo
from worldcam.segmentation import SegmentationMask, draw_segmentation_masks, run_segmentation_analysis
from worldcam.streaming import (
    BufferedStreamReader,
    configure_ffmpeg_http_headers,
    open_with_opencv,
    print_videoio_diagnostics,
    start_ffmpeg_pipe,
)
from worldcam.tracking import PersonTrack, PersonTracker, draw_person_tracks, draw_vehicle_counts
from worldcam.ui import MenuState, close_class_menu_window, consume_menu_changes, draw_fps, draw_stream_counter, handle_class_menu_key, snapshot_menu_state

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


def open_stream(url: str, stream_index: int, stream_total: int) -> tuple[cv2.VideoCapture | None, subprocess.Popen | None]:
    """Open the stream through OpenCV, falling back to an external ffmpeg pipe."""
    print(f"Ouverture du stream {stream_index + 1}/{stream_total}...")
    cap = open_with_opencv(url)
    ffmpeg_process = None

    if cap is None:
        ffmpeg_process = start_ffmpeg_pipe(url)
        print(f"Connexion réussie via ffmpeg.exe. Lecture stabilisée à {TARGET_FPS} FPS. Appuyez sur 'q' pour quitter.")
    else:
        print("Analyse en cours... Appuyez sur 'q' pour quitter.")

    return cap, ffmpeg_process


def release_stream_resources(
    cap: cv2.VideoCapture | None,
    ffmpeg_process: subprocess.Popen | None,
) -> None:
    """Release only the active stream resources."""
    if cap is not None:
        cap.release()
    if ffmpeg_process is not None:
        ffmpeg_process.terminate()
        ffmpeg_process.wait(timeout=5)


def start_buffered_stream_reader(
    cap: cv2.VideoCapture | None,
    ffmpeg_process: subprocess.Popen | None,
) -> BufferedStreamReader:
    """Start a latest-frame RAM buffer for the active stream backend."""
    stream_reader = BufferedStreamReader(cap, ffmpeg_process)
    stream_reader.start()
    return stream_reader


def run_model_analysis(
    frame,
    model: YOLO,
    pose_model: YOLO | None,
    segmentation_model: YOLO | None,
    device: str,
    selected_class_names: set[str],
    latest_detections: list[Detection],
    latest_poses: list[Pose],
    latest_segmentations: list[SegmentationMask],
    pose_enabled: bool,
    segmentation_enabled: bool,
    sahi_enabled: bool = False,
) -> tuple[list[Detection], list[Pose], list[SegmentationMask], YOLO | None, YOLO | None]:
    """Run object and optional pose analysis while preserving previous results on errors."""
    try:
        if sahi_enabled:
            latest_detections = run_sahi_analysis(frame, model, device, selected_class_names)
        else:
            latest_detections = run_yolo_analysis(frame, model, device, selected_class_names)
    except Exception as exc:
        print(f"Erreur pendant l'analyse YOLO26L: {exc}")

    if segmentation_enabled:
        if segmentation_model is None:
            try:
                segmentation_model = load_segmentation_model(device)
            except Exception as exc:
                print(f"Erreur pendant le chargement du modèle segmentation YOLO: {exc}")
                latest_segmentations = []
        if segmentation_model is not None:
            try:
                latest_segmentations = run_segmentation_analysis(frame, segmentation_model, device, selected_class_names)
            except Exception as exc:
                print(f"Erreur pendant l'analyse de segmentation YOLO: {exc}")
    else:
        latest_segmentations = []

    if not pose_enabled:
        return latest_detections, [], latest_segmentations, pose_model, segmentation_model

    if pose_model is None:
        try:
            pose_model = load_pose_model(device)
        except Exception as exc:
            print(f"Erreur pendant le chargement du modèle pose YOLO: {exc}")
            return latest_detections, latest_poses, latest_segmentations, pose_model, segmentation_model

    try:
        latest_poses = run_pose_analysis(frame, pose_model, device)
    except Exception as exc:
        print(f"Erreur pendant l'analyse de pose YOLO: {exc}")

    return latest_detections, latest_poses, latest_segmentations, pose_model, segmentation_model


def draw_overlay(
    frame,
    fps: float,
    detections: list[Detection],
    poses: list[Pose],
    segmentations: list[SegmentationMask],
    display_threshold: float,
    person_tracks: list[PersonTrack],
    stream_index: int,
    stream_total: int,
    vehicle_counts: dict[str, int],
) -> None:
    """Draw every visual overlay on the current frame."""
    draw_segmentation_masks(frame, segmentations, display_threshold)
    draw_yolo_detections(frame, detections, display_threshold)
    draw_person_tracks(frame, person_tracks, display_threshold)
    draw_pose_detections(frame, poses)
    draw_fps(frame, fps)
    draw_stream_counter(frame, stream_index, stream_total)
    draw_vehicle_counts(frame, vehicle_counts)


def throttle_display(next_frame_at: float) -> float:
    """Stabilize display timing so delayed frames are not replayed too quickly."""
    now = time.perf_counter()
    if now < next_frame_at:
        time.sleep(next_frame_at - now)
    elif now - next_frame_at > FRAME_INTERVAL:
        next_frame_at = now
    return next_frame_at + FRAME_INTERVAL


def cleanup_resources(
    cap: cv2.VideoCapture | None,
    ffmpeg_process: subprocess.Popen | None,
    model: YOLO,
    pose_model: YOLO | None,
    segmentation_model: YOLO | None,
    device: str,
) -> None:
    """Release stream, model, and OpenCV resources."""
    release_stream_resources(cap, ffmpeg_process)
    if device == "cuda":
        del model
        if pose_model is not None:
            del pose_model
        if segmentation_model is not None:
            del segmentation_model
        torch.cuda.empty_cache()
    cv2.destroyAllWindows()


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
        cap, ffmpeg_process = open_stream(STREAM_URLS[stream_index], stream_index, stream_total)
    except RuntimeError as exc:
        print(f"Erreur : {exc}")
        return
    stream_reader = start_buffered_stream_reader(cap, ffmpeg_process)

    frame_count = 0
    slow_reads = 0
    stream_read_failures = 0
    started_at = time.perf_counter()
    last_stats_at = started_at
    next_frame_at = time.perf_counter()
    last_frame_at = time.perf_counter()
    current_fps = 0.0

    latest_detections: list[Detection] = []
    latest_poses: list[Pose] = []
    latest_segmentations: list[SegmentationMask] = []
    latest_person_tracks: list[PersonTrack] = []
    person_tracker = PersonTracker()
    class_names, selected_class_names = build_class_selection(model)
    menu_state = MenuState()
    click_state: dict[str, tuple[int, int] | None] = {"click_position": None}
    cv2.namedWindow(MAIN_WINDOW_NAME)
    cv2.setMouseCallback(MAIN_WINDOW_NAME, handle_main_window_mouse, click_state)

    try:
        while True:
            read_started_at = time.perf_counter()
            ret, frame = stream_reader.read(timeout=STREAM_READ_TIMEOUT)
            read_duration = time.perf_counter() - read_started_at

            if read_duration > READ_WARN_SECONDS:
                slow_reads += 1
                print(f"Lecture lente: {read_duration:.3f}s pour récupérer une frame.")

            if not ret:
                stream_read_failures += 1
                print(f"Lecture flux indisponible ({stream_read_failures}/{MAX_STREAM_READ_FAILURES}).")
                if stream_read_failures < MAX_STREAM_READ_FAILURES:
                    continue

                print("Reconnexion automatique du flux...")
                stream_reader.request_stop()
                release_stream_resources(cap, ffmpeg_process)
                try:
                    cap, ffmpeg_process = open_stream(STREAM_URLS[stream_index], stream_index, stream_total)
                    stream_reader = start_buffered_stream_reader(cap, ffmpeg_process)
                    print_videoio_diagnostics(STREAM_URLS[stream_index])
                    stream_read_failures = 0
                    latest_detections = []
                    latest_poses = []
                    latest_segmentations = []
                    latest_person_tracks = []
                    click_state["click_position"] = None
                    close_person_photo_window()
                    person_tracker.reset()
                    continue
                except RuntimeError as exc:
                    print(f"Erreur pendant la reconnexion automatique : {exc}")
                    break

            stream_read_failures = 0

            frame_count += 1
            now = time.perf_counter()
            frame_delta = now - last_frame_at
            if frame_delta > 0:
                current_fps = 1.0 / frame_delta
            last_frame_at = now

            if now - last_stats_at >= STATS_LOG_SECONDS:
                elapsed = now - started_at
                average_fps = frame_count / elapsed if elapsed > 0 else 0.0
                print(
                    f"Stats flux: frames={frame_count}, fps_moyen={average_fps:.1f}, "
                    f"lectures_lentes={slow_reads}, derniere_lecture={read_duration:.3f}s"
                )
                last_stats_at = now

            selected_snapshot, pose_enabled, sahi_enabled, tracking_enabled, segmentation_enabled, display_threshold = snapshot_menu_state(menu_state, selected_class_names)
            if frame_count % FRAME_SKIP == 0:
                latest_detections, latest_poses, latest_segmentations, pose_model, segmentation_model = run_model_analysis(
                    frame,
                    model,
                    pose_model,
                    segmentation_model,
                    device,
                    selected_snapshot,
                    latest_detections,
                    latest_poses,
                    latest_segmentations,
                    pose_enabled,
                    segmentation_enabled,
                    sahi_enabled,
                )
                if tracking_enabled:
                    latest_person_tracks = person_tracker.update(latest_detections)
                else:
                    latest_person_tracks = []
                    person_tracker.reset()

            click_position = click_state["click_position"]
            if click_position is not None:
                click_state["click_position"] = None
                segmentation_model, latest_segmentations = show_clicked_person_photo(
                    frame,
                    click_position,
                    latest_detections,
                    latest_segmentations,
                    segmentation_model,
                    device,
                    display_threshold,
                )

            draw_overlay(
                frame,
                current_fps,
                latest_detections,
                latest_poses,
                latest_segmentations,
                display_threshold,
                latest_person_tracks,
                stream_index,
                stream_total,
                person_tracker.vehicle_counts,
            )
            next_frame_at = throttle_display(next_frame_at)
            cv2.imshow(MAIN_WINDOW_NAME, frame)

            key = cv2.waitKeyEx(1)
            if key == ord("q"):
                break
            if key in KEY_LEFT_VALUES or key in KEY_RIGHT_VALUES:
                step = -1 if key in KEY_LEFT_VALUES else 1
                next_stream_index = (stream_index + step) % stream_total
                stream_reader.request_stop()
                release_stream_resources(cap, ffmpeg_process)
                try:
                    cap, ffmpeg_process = open_stream(STREAM_URLS[next_stream_index], next_stream_index, stream_total)
                    stream_reader = start_buffered_stream_reader(cap, ffmpeg_process)
                    stream_index = next_stream_index
                    print_videoio_diagnostics(STREAM_URLS[stream_index])
                except RuntimeError as exc:
                    print(f"Erreur pendant le changement de stream : {exc}")
                    cap, ffmpeg_process = open_stream(STREAM_URLS[stream_index], stream_index, stream_total)
                    stream_reader = start_buffered_stream_reader(cap, ffmpeg_process)
                latest_detections = []
                latest_poses = []
                latest_segmentations = []
                latest_person_tracks = []
                click_state["click_position"] = None
                close_person_photo_window()
                person_tracker.reset()
                frame_count = 0
                slow_reads = 0
                stream_read_failures = 0
                started_at = time.perf_counter()
                last_stats_at = started_at
                next_frame_at = started_at
                last_frame_at = started_at
                current_fps = 0.0
                continue
            keyboard_class_changed, keyboard_pose_toggled, keyboard_sahi_toggled, keyboard_tracking_toggled, keyboard_segmentation_toggled, keyboard_threshold_changed = handle_class_menu_key(
                key,
                class_names,
                selected_class_names,
                menu_state,
            )
            mouse_class_changed, mouse_pose_toggled, mouse_sahi_toggled, mouse_tracking_toggled, mouse_segmentation_toggled, mouse_threshold_changed = consume_menu_changes(menu_state, selected_class_names)
            class_selection_changed = keyboard_class_changed or mouse_class_changed
            pose_toggled = keyboard_pose_toggled or mouse_pose_toggled
            sahi_toggled = keyboard_sahi_toggled or mouse_sahi_toggled
            tracking_toggled = keyboard_tracking_toggled or mouse_tracking_toggled
            segmentation_toggled = keyboard_segmentation_toggled or mouse_segmentation_toggled
            threshold_changed = keyboard_threshold_changed or mouse_threshold_changed
            if class_selection_changed or sahi_toggled or tracking_toggled or segmentation_toggled:
                latest_detections = []
                latest_segmentations = []
                latest_person_tracks = []
                click_state["click_position"] = None
                close_person_photo_window()
                person_tracker.reset()
            if threshold_changed:
                display_threshold = menu_state.display_threshold
            if pose_toggled and not menu_state.pose_enabled:
                latest_poses = []
    finally:
        close_class_menu_window(menu_state)
        stream_reader.stop()
        cleanup_resources(cap, ffmpeg_process, model, pose_model, segmentation_model, device)
