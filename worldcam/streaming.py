"""Video stream setup and frame reading utilities."""

import os
import shutil
import subprocess
import threading
import time
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


def read_stream_frame(
    cap: cv2.VideoCapture | None,
    ffmpeg_process: subprocess.Popen | None,
) -> tuple[bool, np.ndarray | None]:
    """Read one frame from the active stream backend."""
    if cap is not None:
        return cap.read()

    if ffmpeg_process is None:
        return False, None

    return read_ffmpeg_frame(ffmpeg_process)


class BufferedStreamReader:
    """Read stream frames in a background thread and expose only the latest frame."""

    def __init__(self, cap: cv2.VideoCapture | None, ffmpeg_process: subprocess.Popen | None) -> None:
        self.cap = cap
        self.ffmpeg_process = ffmpeg_process
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.latest_frame: np.ndarray | None = None
        self.latest_frame_id = 0
        self.last_returned_frame_id = 0
        self.frames_read = 0
        self.frames_replaced = 0
        self.consecutive_failures = 0
        self.last_frame_at = 0.0

    def start(self) -> None:
        """Start the background stream reader."""
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._read_loop, name="WorldCamStreamReader", daemon=True)
        self.thread.start()

    def request_stop(self) -> None:
        """Request the background reader to stop without waiting for blocked I/O."""
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()

    def stop(self) -> None:
        """Stop the background stream reader and wake any waiting consumer."""
        self.request_stop()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def _read_loop(self) -> None:
        """Continuously read frames and replace the cached frame with the newest one."""
        while not self.stop_event.is_set():
            ret, frame = read_stream_frame(self.cap, self.ffmpeg_process)
            if not ret or frame is None:
                self.consecutive_failures += 1
                time.sleep(0.05)
                continue

            now = time.perf_counter()
            with self.condition:
                if self.latest_frame_id > self.last_returned_frame_id:
                    self.frames_replaced += 1
                self.latest_frame = frame
                self.latest_frame_id += 1
                self.frames_read += 1
                self.consecutive_failures = 0
                self.last_frame_at = now
                self.condition.notify_all()

    def read(self, timeout: float = 1.0) -> tuple[bool, np.ndarray | None]:
        """Return the next newest frame, waiting briefly for fresh data."""
        deadline = time.perf_counter() + timeout
        with self.condition:
            while self.latest_frame_id == self.last_returned_frame_id and not self.stop_event.is_set():
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return False, None
                self.condition.wait(timeout=remaining)

            if self.latest_frame is None or self.stop_event.is_set():
                return False, None

            self.last_returned_frame_id = self.latest_frame_id
            return True, self.latest_frame
