"""Video stream setup and frame reading utilities."""

import os
import shutil
import subprocess
from urllib.parse import urlparse

import cv2
import numpy as np

from worldcam.config import (
    FFMPEG_HEADERS,
    FRAME_SIZE,
    ORIGIN,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    REFERER,
    TARGET_FPS,
    USER_AGENT,
)


def configure_ffmpeg_http_headers() -> None:
    """Make OpenCV/FFmpeg send the same basic headers as a browser."""
    extra_headers = "\r\n".join(
        [
            f"Origin: {ORIGIN}",
            "Accept: */*",
            "",
        ]
    )
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"user_agent;{USER_AGENT}"
        f"|referer;{REFERER}"
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
