import os
import shutil
import subprocess
import time
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def configure_ffmpeg_http_headers() -> None:
    """Make OpenCV/FFmpeg send the same basic headers as a browser."""
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    extra_headers = "\r\n".join(
        [
            "Origin: https://www.earthcam.com",
            "Accept: */*",
            "",
        ]
    )
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"user_agent;{user_agent}"
        "|referer;https://www.earthcam.com/world/ireland/dublin/"
        f"|headers;{extra_headers}"
    )


def print_videoio_diagnostics(url: str) -> None:
    """Print concise diagnostics before OpenCV tries to open the stream."""
    parsed_url = urlparse(url)
    has_ffmpeg = "FFMPEG:                      YES" in cv2.getBuildInformation()

    print("--- Diagnostics OpenCV VideoCapture ---")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Stream host: {parsed_url.netloc}")
    print(f"FFmpeg backend available: {has_ffmpeg}")
    print("HTTP headers for EarthCam: enabled via OPENCV_FFMPEG_CAPTURE_OPTIONS")
    print("---------------------------------------")


# ATTENTION : Remplacez cette URL par celle contenant le token le plus récent
stream_url = "https://videos-3.earthcam.com/fecnetwork/24322.flv/playlist.m3u8?t=qAP3aum0UbcBtTuO%2Fx%2F7Lz9UytxcCWnrPDJyjgaIxep8QE4xtRu4RMqXEWHwdbnk&td=202605160341"

OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
TARGET_FPS = 25
FRAME_INTERVAL = 1.0 / TARGET_FPS
FRAME_SIZE = OUTPUT_WIDTH * OUTPUT_HEIGHT * 3
READ_WARN_SECONDS = 0.25
STATS_LOG_SECONDS = 5.0

MODEL_PT = "yolo26l.pt"
MODEL_ENGINE = "yolo26l.engine"
DEFAULT_CLASS_NAMES = {"person", "cat"}
INFERENCE_WIDTH = 640
FRAME_SKIP = 4
DETECTION_COLOR = (0, 255, 0)
MENU_BACKGROUND_COLOR = (35, 35, 35)
MENU_SELECTED_COLOR = (0, 255, 255)
MENU_TEXT_COLOR = (255, 255, 255)
MENU_ENABLED_COLOR = (0, 220, 0)
MENU_PAGE_SIZE = 12

FFMPEG_HEADERS = "\r\n".join(
    [
        "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Referer: https://www.earthcam.com/world/ireland/dublin/",
        "Origin: https://www.earthcam.com",
        "Accept: */*",
        "",
    ]
)


def load_yolo_model() -> tuple[YOLO, str]:
    """Load YOLO26L, preferring a TensorRT engine when available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Périphérique YOLO utilisé: {device}")

    try:
        print(f"Chargement du modèle TensorRT: {MODEL_ENGINE}")
        model = YOLO(MODEL_ENGINE)
    except Exception as exc:
        print(f"TensorRT indisponible ({exc}); fallback vers le modèle PyTorch: {MODEL_PT}")
        model = YOLO(MODEL_PT)
        if device == "cuda":
            model.half()

    return model, device


def extract_yolo_detections(
    results,
    model: YOLO,
    scale_x: float,
    scale_y: float,
    selected_class_names: set[str],
) -> list[tuple[int, int, int, int, str]]:
    """Convert selected YOLO results to original-frame coordinates."""
    detections = []

    for box in results.boxes:
        cls_name = model.names[int(box.cls)]
        if cls_name not in selected_class_names:
            continue

        confidence = float(box.conf)
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]
        final_x1 = int(x1 * scale_x)
        final_y1 = int(y1 * scale_y)
        final_x2 = int(x2 * scale_x)
        final_y2 = int(y2 * scale_y)
        label = f"{cls_name} {confidence:.2f}"
        detections.append((final_x1, final_y1, final_x2, final_y2, label))

    return detections


def draw_yolo_detections(frame: np.ndarray, detections: list[tuple[int, int, int, int, str]]) -> None:
    """Draw the latest YOLO detections on the displayed frame."""
    for final_x1, final_y1, final_x2, final_y2, label in detections:
        cv2.rectangle(frame, (final_x1, final_y1), (final_x2, final_y2), DETECTION_COLOR, 2)
        cv2.putText(
            frame,
            label,
            (final_x1, max(final_y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            DETECTION_COLOR,
            2,
        )


def run_yolo_analysis(
    frame: np.ndarray,
    model: YOLO,
    device: str,
    selected_class_names: set[str],
) -> list[tuple[int, int, int, int, str]]:
    """Resize the frame, run YOLO26L inference, and return detections for the original frame."""
    frame_h, frame_w, _ = frame.shape
    new_width = min(INFERENCE_WIDTH, frame_w)
    new_height = int(frame_h * (new_width / frame_w))
    resized_frame = cv2.resize(frame, (new_width, new_height))

    results = model(resized_frame, verbose=False, device=device)[0]
    scale_x = frame_w / new_width
    scale_y = frame_h / new_height
    return extract_yolo_detections(results, model, scale_x, scale_y, selected_class_names)


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
    menu_index: int,
    menu_scroll: int,
) -> None:
    """Draw a small on-screen class selection menu."""
    menu_x = 16
    menu_y = 16
    menu_width = 360
    line_height = 24
    menu_height = 92 + (MENU_PAGE_SIZE * line_height)

    cv2.rectangle(frame, (menu_x, menu_y), (menu_x + menu_width, menu_y + menu_height), MENU_BACKGROUND_COLOR, -1)
    cv2.rectangle(frame, (menu_x, menu_y), (menu_x + menu_width, menu_y + menu_height), MENU_SELECTED_COLOR, 1)
    cv2.putText(frame, "Classes YOLO - M fermer", (menu_x + 12, menu_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MENU_SELECTED_COLOR, 2)
    cv2.putText(frame, "Haut/Bas: naviguer | Espace: cocher", (menu_x + 12, menu_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, MENU_TEXT_COLOR, 1)

    visible_classes = class_names[menu_scroll:menu_scroll + MENU_PAGE_SIZE]
    for visible_index, class_name in enumerate(visible_classes):
        class_index = menu_scroll + visible_index
        y = menu_y + 84 + (visible_index * line_height)
        is_current = class_index == menu_index
        is_enabled = class_name in selected_class_names
        cursor = ">" if is_current else " "
        checkbox = "[x]" if is_enabled else "[ ]"
        color = MENU_SELECTED_COLOR if is_current else MENU_ENABLED_COLOR if is_enabled else MENU_TEXT_COLOR
        cv2.putText(frame, f"{cursor} {checkbox} {class_name}", (menu_x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def open_with_opencv(url: str) -> cv2.VideoCapture | None:
    """Try OpenCV first; return None if OpenCV's bundled FFmpeg refuses the stream."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        print("Connexion réussie via OpenCV/FFmpeg.")
        return cap

    cap.release()
    print("OpenCV/FFmpeg n'ouvre pas ce flux avec headers; fallback vers ffmpeg.exe.")
    return None


def start_ffmpeg_pipe(url: str) -> subprocess.Popen:
    """Start external ffmpeg and pipe decoded BGR frames to Python/OpenCV."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg.exe est requis pour ce fallback, mais il est introuvable dans le PATH.")

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-re",
        "-headers",
        FFMPEG_HEADERS,
        "-i",
        url,
        "-an",
        "-vf",
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},fps={TARGET_FPS}",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def read_ffmpeg_frame(process: subprocess.Popen) -> tuple[bool, np.ndarray | None]:
    """Read exactly one decoded frame from the external ffmpeg pipe."""
    if process.stdout is None:
        return False, None

    raw_frame = process.stdout.read(FRAME_SIZE)
    if len(raw_frame) != FRAME_SIZE:
        return False, None

    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3)).copy()
    return True, frame


def main() -> None:
    configure_ffmpeg_http_headers()
    print_videoio_diagnostics(stream_url)

    model, device = load_yolo_model()
    cap = open_with_opencv(stream_url)
    ffmpeg_process = None

    if cap is None:
        try:
            ffmpeg_process = start_ffmpeg_pipe(stream_url)
            print(f"Connexion réussie via ffmpeg.exe. Lecture stabilisée à {TARGET_FPS} FPS. Appuyez sur 'q' pour quitter.")
        except RuntimeError as exc:
            print(f"Erreur : {exc}")
            return
    else:
        print("Analyse en cours... Appuyez sur 'q' pour quitter.")

    frame_count = 0
    slow_reads = 0
    started_at = time.perf_counter()
    last_stats_at = started_at

    next_frame_at = time.perf_counter()
    last_frame_at = time.perf_counter()
    current_fps = 0.0
    latest_detections: list[tuple[int, int, int, int, str]] = []
    class_names = [model.names[index] for index in sorted(model.names)]
    selected_class_names = {class_name for class_name in DEFAULT_CLASS_NAMES if class_name in class_names}
    menu_open = False
    menu_index = 0
    menu_scroll = 0

    while True:
        read_started_at = time.perf_counter()
        if cap is not None:
            ret, frame = cap.read()
        else:
            ret, frame = read_ffmpeg_frame(ffmpeg_process)
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
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            print(
                f"Stats flux: frames={frame_count}, fps_moyen={fps:.1f}, "
                f"lectures_lentes={slow_reads}, derniere_lecture={read_duration:.3f}s"
            )
            last_stats_at = now

        if frame_count % FRAME_SKIP == 0:
            try:
                latest_detections = run_yolo_analysis(frame, model, device, selected_class_names)
            except Exception as exc:
                print(f"Erreur pendant l'analyse YOLO26L: {exc}")

        draw_yolo_detections(frame, latest_detections)
        draw_fps(frame, current_fps)
        if menu_open:
            draw_class_menu(frame, class_names, selected_class_names, menu_index, menu_scroll)

        # Stabilisation de l'affichage: ne pas rattraper les frames en accéléré
        now = time.perf_counter()
        if now < next_frame_at:
            time.sleep(next_frame_at - now)
        elif now - next_frame_at > FRAME_INTERVAL:
            next_frame_at = now
        next_frame_at += FRAME_INTERVAL

        # Affichage couleur du résultat
        cv2.imshow('Analyse Image - Dublin Cam', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('m'):
            menu_open = not menu_open
        elif menu_open and key in (82, ord('z'), ord('w')):
            menu_index = max(0, menu_index - 1)
            if menu_index < menu_scroll:
                menu_scroll = menu_index
        elif menu_open and key in (84, ord('s')):
            menu_index = min(len(class_names) - 1, menu_index + 1)
            if menu_index >= menu_scroll + MENU_PAGE_SIZE:
                menu_scroll = menu_index - MENU_PAGE_SIZE + 1
        elif menu_open and key == ord(' '):
            class_name = class_names[menu_index]
            if class_name in selected_class_names:
                selected_class_names.remove(class_name)
            else:
                selected_class_names.add(class_name)
            latest_detections = []

    if cap is not None:
        cap.release()
    if ffmpeg_process is not None:
        ffmpeg_process.terminate()
        ffmpeg_process.wait(timeout=5)
    if device == "cuda":
        del model
        torch.cuda.empty_cache()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()