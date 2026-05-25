"""Stream resource orchestration helpers for WorldCam."""

from dataclasses import dataclass
import subprocess
import time

import cv2

from worldcam.core.config import STREAM_URLS, TARGET_FPS
from worldcam.stream.streaming import BufferedStreamReader, open_with_opencv, print_videoio_diagnostics, start_ffmpeg_pipe


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
    """Release only the active stream resources without letting stuck FFmpeg crash the app."""
    if cap is not None:
        try:
            cap.release()
        except cv2.error as exc:
            print(f"Avertissement: impossible de libérer OpenCV VideoCapture proprement: {exc}")

    if ffmpeg_process is None:
        return

    for pipe in (ffmpeg_process.stdout, ffmpeg_process.stderr):
        if pipe is not None:
            try:
                pipe.close()
            except OSError:
                pass

    if ffmpeg_process.poll() is not None:
        return

    try:
        ffmpeg_process.terminate()
        ffmpeg_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("Avertissement: FFmpeg ne répond pas à terminate(); arrêt forcé...")
        ffmpeg_process.kill()
        try:
            ffmpeg_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Avertissement: FFmpeg ne s'est pas arrêté après kill(); poursuite sans crash.")
    except OSError as exc:
        print(f"Avertissement: erreur pendant l'arrêt de FFmpeg: {exc}")


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
    resources.stream_reader.stop()


def reconnect_retry_delay(attempt: int) -> int:
    """Return the retry delay for a 5x1s, 5x5s, 5x10s, then 60s reconnect schedule."""
    if attempt <= 5:
        return 1
    if attempt <= 10:
        return 5
    if attempt <= 15:
        return 10
    return 60


def open_stream_resources_with_retry(url: str, stream_index: int, stream_total: int) -> StreamResources:
    """Open stream resources forever using the progressive reconnect retry schedule."""
    attempt = 1
    while True:
        try:
            return open_stream_resources(url, stream_index, stream_total)
        except RuntimeError as exc:
            delay = reconnect_retry_delay(attempt)
            print(f"Reconnexion en cours... tentative {attempt} pour le stream {stream_index + 1}/{stream_total}.")
            print(f"Connexion impossible: {exc}")
            print(f"Nouvel essai dans {delay}s...")
            time.sleep(delay)
            attempt += 1


def reconnect_current_stream(resources: StreamResources, stream_index: int, stream_total: int) -> StreamResources:
    """Reconnect the current stream forever using a progressive retry schedule."""
    close_active_stream(resources)
    url = STREAM_URLS[stream_index]
    attempt = 1

    while True:
        print(f"Reconnexion en cours... tentative {attempt} pour le stream {stream_index + 1}/{stream_total}.")
        try:
            resources = open_stream_resources(url, stream_index, stream_total)
            print_videoio_diagnostics(url)
            print("Reconnexion réussie.")
            return resources
        except RuntimeError as exc:
            delay = reconnect_retry_delay(attempt)
            print(f"Reconnexion impossible: {exc}")
            print(f"Nouvel essai dans {delay}s...")
            time.sleep(delay)
            attempt += 1


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
