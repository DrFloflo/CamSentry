"""Stream resource orchestration helpers for WorldCam."""

from dataclasses import dataclass
import subprocess

import cv2

from worldcam.config import STREAM_URLS, TARGET_FPS
from worldcam.streaming import BufferedStreamReader, open_with_opencv, print_videoio_diagnostics, start_ffmpeg_pipe


@dataclass
class StreamResources:
    """Active stream handles used by the main loop."""

    cap: cv2.VideoCapture | None
    ffmpeg_process: subprocess.Popen | None
    stream_reader: BufferedStreamReader


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


def open_stream_resources(url: str, stream_index: int, stream_total: int) -> StreamResources:
    """Open stream handles and start their buffered reader."""
    cap, ffmpeg_process = open_stream(url, stream_index, stream_total)
    stream_reader = start_buffered_stream_reader(cap, ffmpeg_process)
    return StreamResources(cap=cap, ffmpeg_process=ffmpeg_process, stream_reader=stream_reader)


def close_active_stream(resources: StreamResources) -> None:
    """Request the active reader to stop and release its stream handles."""
    resources.stream_reader.request_stop()
    release_stream_resources(resources.cap, resources.ffmpeg_process)


def reconnect_current_stream(resources: StreamResources, stream_index: int, stream_total: int) -> StreamResources:
    """Reconnect the currently selected stream and return fresh resources."""
    close_active_stream(resources)
    resources = open_stream_resources(STREAM_URLS[stream_index], stream_index, stream_total)
    print_videoio_diagnostics(STREAM_URLS[stream_index])
    return resources


def switch_stream(resources: StreamResources, stream_index: int, step: int, stream_total: int) -> tuple[StreamResources, int]:
    """Move left/right between streams, falling back to the current stream on errors."""
    next_stream_index = (stream_index + step) % stream_total
    close_active_stream(resources)
    try:
        resources = open_stream_resources(STREAM_URLS[next_stream_index], next_stream_index, stream_total)
        stream_index = next_stream_index
        print_videoio_diagnostics(STREAM_URLS[stream_index])
    except RuntimeError as exc:
        print(f"Erreur pendant le changement de stream : {exc}")
        resources = open_stream_resources(STREAM_URLS[stream_index], stream_index, stream_total)
    return resources, stream_index
