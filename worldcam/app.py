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
    STREAM_URL,
    TARGET_FPS,
)
from worldcam.detection import Detection, draw_yolo_detections, run_yolo_analysis
from worldcam.models import load_pose_model, load_yolo_model
from worldcam.pose import Pose, draw_pose_detections, run_pose_analysis
from worldcam.streaming import (
    configure_ffmpeg_http_headers,
    open_with_opencv,
    print_videoio_diagnostics,
    read_ffmpeg_frame,
    start_ffmpeg_pipe,
)
from worldcam.ui import MenuState, draw_class_menu, draw_fps, handle_class_menu_key


def build_class_selection(model: YOLO) -> tuple[list[str], set[str]]:
    """Build the full class list and initial enabled class selection."""
    class_names = [model.names[index] for index in sorted(model.names)]
    selected_class_names = {class_name for class_name in DEFAULT_CLASS_NAMES if class_name in class_names}
    return class_names, selected_class_names


def open_stream(url: str) -> tuple[cv2.VideoCapture | None, subprocess.Popen | None]:
    """Open the stream through OpenCV, falling back to an external ffmpeg pipe."""
    cap = open_with_opencv(url)
    ffmpeg_process = None

    if cap is None:
        ffmpeg_process = start_ffmpeg_pipe(url)
        print(f"Connexion réussie via ffmpeg.exe. Lecture stabilisée à {TARGET_FPS} FPS. Appuyez sur 'q' pour quitter.")
    else:
        print("Analyse en cours... Appuyez sur 'q' pour quitter.")

    return cap, ffmpeg_process


def read_stream_frame(
    cap: cv2.VideoCapture | None,
    ffmpeg_process: subprocess.Popen | None,
) -> tuple[bool, object]:
    """Read one frame from the active stream backend."""
    if cap is not None:
        return cap.read()

    if ffmpeg_process is None:
        return False, None

    return read_ffmpeg_frame(ffmpeg_process)


def run_model_analysis(
    frame,
    model: YOLO,
    pose_model: YOLO,
    device: str,
    selected_class_names: set[str],
    latest_detections: list[Detection],
    latest_poses: list[Pose],
) -> tuple[list[Detection], list[Pose]]:
    """Run object and pose analysis while preserving previous results on errors."""
    try:
        latest_detections = run_yolo_analysis(frame, model, device, selected_class_names)
    except Exception as exc:
        print(f"Erreur pendant l'analyse YOLO26L: {exc}")

    try:
        latest_poses = run_pose_analysis(frame, pose_model, device)
    except Exception as exc:
        print(f"Erreur pendant l'analyse de pose YOLO: {exc}")

    return latest_detections, latest_poses


def draw_overlay(
    frame,
    fps: float,
    detections: list[Detection],
    poses: list[Pose],
    class_names: list[str],
    selected_class_names: set[str],
    menu_state: MenuState,
) -> None:
    """Draw every visual overlay on the current frame."""
    draw_yolo_detections(frame, detections)
    draw_pose_detections(frame, poses)
    draw_fps(frame, fps)
    if menu_state.is_open:
        draw_class_menu(frame, class_names, selected_class_names, menu_state)


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
    pose_model: YOLO,
    device: str,
) -> None:
    """Release stream, model, and OpenCV resources."""
    if cap is not None:
        cap.release()
    if ffmpeg_process is not None:
        ffmpeg_process.terminate()
        ffmpeg_process.wait(timeout=5)
    if device == "cuda":
        del model
        del pose_model
        torch.cuda.empty_cache()
    cv2.destroyAllWindows()


def main() -> None:
    """Run the WorldCam analysis application."""
    configure_ffmpeg_http_headers()
    print_videoio_diagnostics(STREAM_URL)

    model, device = load_yolo_model()
    pose_model = load_pose_model(device)

    try:
        cap, ffmpeg_process = open_stream(STREAM_URL)
    except RuntimeError as exc:
        print(f"Erreur : {exc}")
        return

    frame_count = 0
    slow_reads = 0
    started_at = time.perf_counter()
    last_stats_at = started_at
    next_frame_at = time.perf_counter()
    last_frame_at = time.perf_counter()
    current_fps = 0.0

    latest_detections: list[Detection] = []
    latest_poses: list[Pose] = []
    class_names, selected_class_names = build_class_selection(model)
    menu_state = MenuState()

    try:
        while True:
            read_started_at = time.perf_counter()
            ret, frame = read_stream_frame(cap, ffmpeg_process)
            read_duration = time.perf_counter() - read_started_at

            if read_duration > READ_WARN_SECONDS:
                slow_reads += 1
                print(f"Lecture lente: {read_duration:.3f}s pour récupérer une frame.")

            if not ret:
                print("Erreur : Impossible de recevoir la frame.")
                break

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

            if frame_count % FRAME_SKIP == 0:
                latest_detections, latest_poses = run_model_analysis(
                    frame,
                    model,
                    pose_model,
                    device,
                    selected_class_names,
                    latest_detections,
                    latest_poses,
                )

            draw_overlay(
                frame,
                current_fps,
                latest_detections,
                latest_poses,
                class_names,
                selected_class_names,
                menu_state,
            )

            next_frame_at = throttle_display(next_frame_at)
            cv2.imshow("Analyse Image - Dublin Cam", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if handle_class_menu_key(key, class_names, selected_class_names, menu_state):
                latest_detections = []
    finally:
        cleanup_resources(cap, ffmpeg_process, model, pose_model, device)
