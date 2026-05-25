"""FFmpeg pipe helpers for WorldCam stream analysis tests."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np

from worldcam.core.config import (
    FFMPEG_ANALYZEDURATION_US,
    FFMPEG_HEADERS,
    FFMPEG_HWACCEL,
    FFMPEG_INPUT_REALTIME,
    FFMPEG_MAX_DELAY_US,
    FFMPEG_PROBESIZE,
    FFMPEG_RECONNECT_DELAY_MAX_SECONDS,
    FFMPEG_THREAD_QUEUE_SIZE,
)


def start_analysis_ffmpeg_pipe(url: str, width: int, height: int, fps: int) -> subprocess.Popen:
    """Start ffmpeg with an explicit output profile for measurable raw frames."""

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg.exe is required but was not found in PATH.")

    video_filter = f"scale={width}:{height}:flags=bicubic"
    if fps > 0:
        video_filter = f"{video_filter},fps={fps}"

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-fflags",
        "nobuffer+discardcorrupt",
        "-flags",
        "low_delay",
        "-max_delay",
        str(FFMPEG_MAX_DELAY_US),
        "-probesize",
        str(FFMPEG_PROBESIZE),
        "-analyzeduration",
        str(FFMPEG_ANALYZEDURATION_US),
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        str(FFMPEG_RECONNECT_DELAY_MAX_SECONDS),
        "-thread_queue_size",
        str(FFMPEG_THREAD_QUEUE_SIZE),
    ]
    if FFMPEG_HWACCEL:
        command.extend(["-hwaccel", FFMPEG_HWACCEL])
    if FFMPEG_INPUT_REALTIME:
        command.append("-re")
    command.extend(
        [
            "-headers",
            FFMPEG_HEADERS,
            "-i",
            url,
            "-an",
            "-vf",
            video_filter,
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def read_profile_frame(process: subprocess.Popen, width: int, height: int) -> tuple[bool, np.ndarray | None]:
    """Read exactly one BGR frame for the selected output profile."""

    if process.stdout is None:
        return False, None
    frame_size = width * height * 3
    raw_frame = process.stdout.read(frame_size)
    if len(raw_frame) != frame_size:
        return False, None
    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3)).copy()
    return True, frame


def stop_ffmpeg_process(process: subprocess.Popen) -> None:
    """Stop an external FFmpeg process without leaving a decoder running."""

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
