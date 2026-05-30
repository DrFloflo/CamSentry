"""Metric collection for WorldCam stream analysis tests."""

from __future__ import annotations

import math
import statistics
import time

import cv2
import numpy as np

from worldcam.core.config import FFMPEG_REPEAT_DELTA_THRESHOLD
from worldcam.tests.stream_analysis.stream_analysis_ffmpeg import read_profile_frame, start_analysis_ffmpeg_pipe, stop_ffmpeg_process
from worldcam.tests.stream_analysis.stream_analysis_types import StreamAnalysis


def percentile(values: list[float], percentile_value: float) -> float | None:
    """Return a percentile from a non-empty numeric list."""

    if not values:
        return None
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * percentile_value
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    lower_weight = upper_index - rank
    upper_weight = rank - lower_index
    return sorted_values[lower_index] * lower_weight + sorted_values[upper_index] * upper_weight


def measure_frame_quality(frame: np.ndarray) -> tuple[float, float, float]:
    """Return brightness, contrast, and sharpness proxy for a decoded frame."""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, contrast, sharpness


def calculate_stability_score(
    frames_read: int,
    read_failures: int,
    expected_frames: float,
    jitter_ms: float | None,
    frozen_frame_ratio: float | None,
) -> float:
    """Calculate a 0-100 stability score from continuity, timing, and freeze indicators."""

    total_attempts = max(1, frames_read + read_failures)
    success_ratio = frames_read / total_attempts
    expected_ratio = min(1.0, frames_read / max(1.0, expected_frames))
    jitter_penalty = min(0.35, (jitter_ms or 0.0) / 200.0)
    freeze_penalty = min(0.35, frozen_frame_ratio or 0.0)
    score = 100.0 * ((0.55 * success_ratio) + (0.45 * expected_ratio))
    score *= 1.0 - jitter_penalty
    score *= 1.0 - freeze_penalty
    return round(max(0.0, min(100.0, score)), 1)


def failed_stream_analysis(stream_index: int, url: str, profile_name: str, exc: RuntimeError) -> StreamAnalysis:
    """Build a failed stream-analysis result when ffmpeg cannot start."""

    return StreamAnalysis(
        stream_index=stream_index,
        url=url,
        profile=profile_name,
        opened=False,
        sampled_seconds=0.0,
        frames_read=0,
        read_failures=1,
        width=None,
        height=None,
        capture_fps_hint=None,
        decoded_fps=0.0,
        fps_vs_target_ratio=None,
        average_read_latency_ms=None,
        p95_read_latency_ms=None,
        average_frame_interval_ms=None,
        frame_interval_jitter_ms=None,
        stability_score=0.0,
        estimated_decoded_mbps=0.0,
        average_brightness=None,
        average_contrast=None,
        average_sharpness=None,
        frozen_frame_ratio=None,
        notes=[f"Could not start ffmpeg.exe fallback: {exc}"],
    )


def collect_frames(
    process,
    width: int,
    height: int,
    duration_seconds: float,
    max_frames: int,
    progress_label: str | None = None,
    progress_interval: float = 2.0,
) -> dict[str, object]:
    """Read decoded frames for one profile and collect raw measurements."""

    read_latencies: list[float] = []
    frame_times: list[float] = []
    brightness_values: list[float] = []
    contrast_values: list[float] = []
    sharpness_values: list[float] = []
    read_failures = 0
    frames_read = 0
    frozen_transitions = 0
    frame_deltas: list[float] = []
    repeated_delta_threshold = min(FFMPEG_REPEAT_DELTA_THRESHOLD, 0.25)
    previous_small_frame: np.ndarray | None = None
    started_at = time.perf_counter()
    deadline = started_at + duration_seconds
    next_progress_at = started_at + max(0.5, progress_interval)

    while time.perf_counter() < deadline:
        if max_frames > 0 and frames_read >= max_frames:
            break

        read_started_at = time.perf_counter()
        ret, frame = read_profile_frame(process, width, height)
        read_finished_at = time.perf_counter()
        read_latencies.append(read_finished_at - read_started_at)

        if not ret or frame is None:
            read_failures += 1
            time.sleep(0.05)
            continue

        frames_read += 1
        frame_times.append(read_finished_at)
        brightness, contrast, sharpness = measure_frame_quality(frame)
        brightness_values.append(brightness)
        contrast_values.append(contrast)
        sharpness_values.append(sharpness)

        small_frame = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        if previous_small_frame is not None:
            mean_delta = float(np.mean(cv2.absdiff(previous_small_frame, small_frame)))
            frame_deltas.append(mean_delta)
            if mean_delta < repeated_delta_threshold:
                frozen_transitions += 1
        previous_small_frame = small_frame

        now = time.perf_counter()
        if progress_label is not None and now >= next_progress_at:
            elapsed = max(0.001, now - started_at)
            current_fps = frames_read / elapsed
            remaining = max(0.0, deadline - now)
            print(
                f"  ... {progress_label}: {elapsed:.1f}/{duration_seconds:.1f}s, "
                f"frames={frames_read}, fps={current_fps:.2f}, failures={read_failures}, remaining={remaining:.1f}s",
                flush=True,
            )
            next_progress_at = now + max(0.5, progress_interval)

    return {
        "sampled_seconds": max(0.0, time.perf_counter() - started_at),
        "read_latencies": read_latencies,
        "frame_times": frame_times,
        "brightness_values": brightness_values,
        "contrast_values": contrast_values,
        "sharpness_values": sharpness_values,
        "read_failures": read_failures,
        "frames_read": frames_read,
        "frozen_transitions": frozen_transitions,
        "frame_deltas": frame_deltas,
    }


def build_analysis_result(
    stream_index: int,
    url: str,
    profile_name: str,
    width: int,
    height: int,
    target_fps: int,
    measurements: dict[str, object],
) -> StreamAnalysis:
    """Convert collected measurements into a reportable analysis result."""

    sampled_seconds = float(measurements["sampled_seconds"])
    frames_read = int(measurements["frames_read"])
    read_failures = int(measurements["read_failures"])
    read_latencies = measurements["read_latencies"]
    frame_times = measurements["frame_times"]
    brightness_values = measurements["brightness_values"]
    contrast_values = measurements["contrast_values"]
    sharpness_values = measurements["sharpness_values"]
    frozen_transitions = int(measurements["frozen_transitions"])
    frame_deltas = measurements["frame_deltas"]

    decoded_fps = frames_read / sampled_seconds if sampled_seconds > 0 else 0.0
    fps_vs_target_ratio = decoded_fps / target_fps if target_fps > 0 else None
    frame_intervals = [later - earlier for earlier, later in zip(frame_times, frame_times[1:])]
    average_interval = statistics.fmean(frame_intervals) if frame_intervals else None
    jitter = statistics.pstdev(frame_intervals) if len(frame_intervals) >= 2 else None
    frozen_frame_ratio = frozen_transitions / max(1, frames_read - 1) if frames_read >= 2 else None
    low_motion_sample = bool(frame_deltas and statistics.median(frame_deltas) < FFMPEG_REPEAT_DELTA_THRESHOLD)
    score_frozen_frame_ratio = 0.0 if low_motion_sample else frozen_frame_ratio
    bytes_per_second = (frames_read * width * height * 3) / sampled_seconds if sampled_seconds > 0 else 0.0
    expected_frames = sampled_seconds * target_fps if target_fps > 0 else frames_read
    notes = ["Analyzed via external ffmpeg.exe rawvideo fallback, without trying OpenCV VideoCapture."]

    if frames_read == 0:
        notes.append("No decodable frames were read during the sample window.")
    if fps_vs_target_ratio is not None and fps_vs_target_ratio < 0.75:
        notes.append("Decoded FPS is below 75% of the tested profile FPS.")
    if frozen_frame_ratio is not None and frozen_frame_ratio > 0.05:
        notes.append("Near-identical repeated frames detected.")
    if low_motion_sample:
        notes.append("Very low scene motion during the sample; repeated-frame ratio is reported but not used as a stability penalty.")
    if jitter is not None and jitter > 0.1:
        notes.append("Frame interval jitter is high.")

    return StreamAnalysis(
        stream_index=stream_index,
        url=url,
        profile=profile_name,
        opened=True,
        sampled_seconds=round(sampled_seconds, 3),
        frames_read=frames_read,
        read_failures=read_failures,
        width=width,
        height=height,
        capture_fps_hint=round(float(target_fps), 3) if target_fps > 0 else None,
        decoded_fps=round(decoded_fps, 3),
        fps_vs_target_ratio=round(fps_vs_target_ratio, 3) if fps_vs_target_ratio is not None else None,
        average_read_latency_ms=round(statistics.fmean(read_latencies) * 1000.0, 3) if read_latencies else None,
        p95_read_latency_ms=round(percentile(read_latencies, 0.95) * 1000.0, 3) if read_latencies else None,
        average_frame_interval_ms=round(average_interval * 1000.0, 3) if average_interval is not None else None,
        frame_interval_jitter_ms=round(jitter * 1000.0, 3) if jitter is not None else None,
        stability_score=calculate_stability_score(frames_read, read_failures, expected_frames, (jitter or 0.0) * 1000.0, score_frozen_frame_ratio),
        estimated_decoded_mbps=round((bytes_per_second * 8.0) / 1_000_000.0, 3),
        average_brightness=round(statistics.fmean(brightness_values), 3) if brightness_values else None,
        average_contrast=round(statistics.fmean(contrast_values), 3) if contrast_values else None,
        average_sharpness=round(statistics.fmean(sharpness_values), 3) if sharpness_values else None,
        frozen_frame_ratio=round(frozen_frame_ratio, 4) if frozen_frame_ratio is not None else None,
        notes=notes,
    )


def analyze_stream(
    stream_index: int,
    url: str,
    profile_name: str,
    width: int,
    height: int,
    target_fps: int,
    duration_seconds: float,
    max_frames: int,
    verbose: bool = True,
    progress_interval: float = 2.0,
) -> StreamAnalysis:
    """Sample one configured stream through an explicit external FFmpeg decode profile."""

    progress_label = f"stream {stream_index} / profile {profile_name}"
    if verbose:
        print(f"Starting {progress_label} for {duration_seconds:.1f}s...", flush=True)

    try:
        process = start_analysis_ffmpeg_pipe(url, width, height, target_fps)
    except RuntimeError as exc:
        return failed_stream_analysis(stream_index, url, profile_name, exc)

    try:
        measurements = collect_frames(
            process,
            width,
            height,
            duration_seconds,
            max_frames,
            progress_label=progress_label if verbose else None,
            progress_interval=progress_interval,
        )
    finally:
        stop_ffmpeg_process(process)

    result = build_analysis_result(stream_index, url, profile_name, width, height, target_fps, measurements)
    if verbose:
        print(
            f"Finished {progress_label}: frames={result.frames_read}, fps={result.decoded_fps}, "
            f"stability={result.stability_score}/100, sharpness={result.average_sharpness}",
            flush=True,
        )
    return result
