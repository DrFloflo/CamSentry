import os
import shutil
import subprocess
import time
from urllib.parse import urlparse

import cv2
import numpy as np


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

    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3))
    return True, frame


def main() -> None:
    configure_ffmpeg_http_headers()
    print_videoio_diagnostics(stream_url)

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
        if now - last_stats_at >= STATS_LOG_SECONDS:
            elapsed = now - started_at
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            print(
                f"Stats flux: frames={frame_count}, fps_moyen={fps:.1f}, "
                f"lectures_lentes={slow_reads}, derniere_lecture={read_duration:.3f}s"
            )
            last_stats_at = now

        # ---------------------------------------------------------
        # ZONE DE VOTRE ANALYSE D'IMAGE
        # Vous pouvez injecter ici votre modèle YOLO, détection de mouvement, etc.
        # ---------------------------------------------------------

        # Stabilisation de l'affichage: ne pas rattraper les frames en accéléré
        now = time.perf_counter()
        if now < next_frame_at:
            time.sleep(next_frame_at - now)
        elif now - next_frame_at > FRAME_INTERVAL:
            next_frame_at = now
        next_frame_at += FRAME_INTERVAL

        # Affichage couleur du résultat
        cv2.imshow('Analyse Image - Dublin Cam', frame)

        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if cap is not None:
        cap.release()
    if ffmpeg_process is not None:
        ffmpeg_process.terminate()
        ffmpeg_process.wait(timeout=5)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()