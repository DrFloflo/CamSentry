"""Metric collection for WorldCam exclusion-zone CPU/RAM benchmarks."""

from __future__ import annotations

import ctypes
import gc
import math
import os
import statistics
import time
from dataclasses import asdict
from typing import Callable

import numpy as np
from ultralytics import YOLO

from worldcam.analysis.counting_zone import ZonePoints
from worldcam.analysis.detection import run_sahi_analysis, run_yolo_analysis
from worldcam.core.config import DEFAULT_CLASS_NAMES, EXCLUSION_ZONE_POINTS
from worldcam.tests.exclusion_benchmark.exclusion_benchmark_types import BenchmarkResult, BenchmarkScenario
from worldcam.tests.stream_analysis.stream_analysis_ffmpeg import read_profile_frame, start_analysis_ffmpeg_pipe, stop_ffmpeg_process


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


def current_rss_bytes() -> int | None:
    """Return current process resident memory using psutil when available, then stdlib fallbacks."""

    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass

    if os.name == "nt":
        try:
            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(ProcessMemoryCounters)
            process_handle = ctypes.windll.kernel32.GetCurrentProcess()
            success = ctypes.windll.psapi.GetProcessMemoryInfo(process_handle, ctypes.byref(counters), counters.cb)
            if success:
                return int(counters.WorkingSetSize)
        except Exception:
            return None
        return None

    try:
        import resource

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss if sys_platform_is_macos() else rss * 1024
    except Exception:
        return None


def sys_platform_is_macos() -> bool:
    """Return whether the current platform reports ru_maxrss in bytes."""

    return os.sys.platform == "darwin"


def bytes_to_mb(value: int | None) -> float | None:
    """Convert bytes to MiB rounded for reports."""

    if value is None:
        return None
    return round(value / (1024.0 * 1024.0), 3)


def build_scenarios(include_sahi: bool = True) -> list[BenchmarkScenario]:
    """Build the four requested baseline/exclusion and YOLO/SAHI scenarios."""

    scenarios = [
        BenchmarkScenario("yolo_without_exclusion", sahi_enabled=False, exclusion_enabled=False),
        BenchmarkScenario("yolo_with_exclusion", sahi_enabled=False, exclusion_enabled=True),
    ]
    if include_sahi:
        scenarios.extend(
            [
                BenchmarkScenario("sahi_without_exclusion", sahi_enabled=True, exclusion_enabled=False),
                BenchmarkScenario("sahi_with_exclusion", sahi_enabled=True, exclusion_enabled=True),
            ]
        )
    return scenarios


def build_class_selection(model: YOLO) -> set[str]:
    """Return configured class names that exist in the active model."""

    available_class_names = {model.names[index] for index in sorted(model.names)}
    return {class_name for class_name in DEFAULT_CLASS_NAMES if class_name in available_class_names}


def run_detection_for_scenario(
    frame: np.ndarray,
    model: YOLO,
    device: str,
    selected_class_names: set[str],
    scenario: BenchmarkScenario,
    exclusion_zone_points: ZonePoints,
):
    """Run YOLO or SAHI for one scenario and return detections."""

    if scenario.sahi_enabled:
        return run_sahi_analysis(
            frame,
            model,
            device,
            selected_class_names,
            exclusion_zone_points,
            scenario.exclusion_enabled,
        )
    return run_yolo_analysis(
        frame,
        model,
        device,
        selected_class_names,
        exclusion_zone_points,
        scenario.exclusion_enabled,
    )


def warm_up_scenario(
    frames: list[np.ndarray],
    model: YOLO,
    device: str,
    selected_class_names: set[str],
    scenario: BenchmarkScenario,
    exclusion_zone_points: ZonePoints,
) -> None:
    """Run a few frames outside the measured window to initialize kernels/caches."""

    for frame in frames:
        run_detection_for_scenario(frame, model, device, selected_class_names, scenario, exclusion_zone_points)


def collect_benchmark_frames(
    url: str,
    width: int,
    height: int,
    target_fps: int,
    duration_seconds: float,
    max_frames: int,
    warmup_frames: int,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Decode frames once so every scenario processes the same image sequence."""

    process = start_analysis_ffmpeg_pipe(url, width, height, target_fps)
    frames: list[np.ndarray] = []
    warmup: list[np.ndarray] = []
    started_at = time.perf_counter()
    deadline = started_at + duration_seconds
    total_limit = max_frames if max_frames > 0 else None

    try:
        while time.perf_counter() < deadline:
            if total_limit is not None and len(frames) >= total_limit:
                break
            ret, frame = read_profile_frame(process, width, height)
            if not ret or frame is None:
                time.sleep(0.02)
                continue
            if len(warmup) < warmup_frames:
                warmup.append(frame)
            else:
                frames.append(frame)
            if progress is not None and len(frames) > 0 and len(frames) % 25 == 0:
                progress(f"decoded measured frames={len(frames)}")
    finally:
        stop_ffmpeg_process(process)

    return warmup, frames


def benchmark_scenario(
    scenario: BenchmarkScenario,
    frames: list[np.ndarray],
    warmup_frames: list[np.ndarray],
    model: YOLO,
    device: str,
    selected_class_names: set[str],
    exclusion_zone_points: ZonePoints,
    stream_index: int,
    profile_name: str,
    width: int,
    height: int,
    verbose: bool = True,
) -> BenchmarkResult:
    """Measure CPU time, wall time, RSS, and detection throughput for one scenario."""

    notes: list[str] = []
    if not frames:
        notes.append("No frames available for this scenario.")
        return BenchmarkResult(
            scenario=scenario.name,
            sahi_enabled=scenario.sahi_enabled,
            exclusion_enabled=scenario.exclusion_enabled,
            stream_index=stream_index,
            profile=profile_name,
            width=width,
            height=height,
            frames_processed=0,
            sampled_seconds=0.0,
            frames_per_second=0.0,
            average_frame_ms=None,
            p95_frame_ms=None,
            process_cpu_percent=0.0,
            process_cpu_percent_per_core=0.0,
            ram_start_mb=None,
            ram_peak_mb=None,
            ram_delta_mb=None,
            detections_total=0,
            detections_per_frame=0.0,
            exclusion_zone_points=list(exclusion_zone_points),
            notes=notes,
        )

    if verbose:
        print(f"Starting scenario {scenario.name}: frames={len(frames)}", flush=True)

    warm_up_scenario(warmup_frames, model, device, selected_class_names, scenario, exclusion_zone_points)
    gc.collect()

    ram_start = current_rss_bytes()
    ram_peak = ram_start or 0
    frame_durations: list[float] = []
    detections_total = 0
    wall_started_at = time.perf_counter()
    cpu_started_at = time.process_time()

    for frame in frames:
        frame_started_at = time.perf_counter()
        detections = run_detection_for_scenario(frame, model, device, selected_class_names, scenario, exclusion_zone_points)
        frame_finished_at = time.perf_counter()
        frame_durations.append(frame_finished_at - frame_started_at)
        detections_total += len(detections)
        ram_current = current_rss_bytes()
        if ram_current is not None:
            ram_peak = max(ram_peak, ram_current)

    cpu_seconds = time.process_time() - cpu_started_at
    sampled_seconds = time.perf_counter() - wall_started_at
    cpu_count = max(1, os.cpu_count() or 1)
    cpu_percent = (cpu_seconds / sampled_seconds) * 100.0 if sampled_seconds > 0 else 0.0
    cpu_percent_per_core = cpu_percent / cpu_count
    frames_processed = len(frames)
    frames_per_second = frames_processed / sampled_seconds if sampled_seconds > 0 else 0.0

    if ram_start is None:
        notes.append("RSS memory unavailable; install psutil for the most portable measurement.")
        ram_peak_mb = None
        ram_delta_mb = None
    else:
        ram_peak_mb = bytes_to_mb(ram_peak)
        ram_delta_mb = bytes_to_mb(max(0, ram_peak - ram_start))

    result = BenchmarkResult(
        scenario=scenario.name,
        sahi_enabled=scenario.sahi_enabled,
        exclusion_enabled=scenario.exclusion_enabled,
        stream_index=stream_index,
        profile=profile_name,
        width=width,
        height=height,
        frames_processed=frames_processed,
        sampled_seconds=round(sampled_seconds, 3),
        frames_per_second=round(frames_per_second, 3),
        average_frame_ms=round(statistics.fmean(frame_durations) * 1000.0, 3) if frame_durations else None,
        p95_frame_ms=round(percentile(frame_durations, 0.95) * 1000.0, 3) if frame_durations else None,
        process_cpu_percent=round(cpu_percent, 3),
        process_cpu_percent_per_core=round(cpu_percent_per_core, 3),
        ram_start_mb=bytes_to_mb(ram_start),
        ram_peak_mb=ram_peak_mb,
        ram_delta_mb=ram_delta_mb,
        detections_total=detections_total,
        detections_per_frame=round(detections_total / max(1, frames_processed), 3),
        exclusion_zone_points=list(exclusion_zone_points),
        notes=notes,
    )
    if verbose:
        print(
            f"Finished {scenario.name}: fps={result.frames_per_second}, "
            f"cpu={result.process_cpu_percent:.1f}%, ram_peak={result.ram_peak_mb} MiB",
            flush=True,
        )
    return result


def compare_results(results: list[BenchmarkResult]) -> dict[str, object]:
    """Build compact deltas between exclusion and non-exclusion results for YOLO and SAHI."""

    by_name = {result.scenario: result for result in results}
    comparisons: dict[str, object] = {}
    for prefix in ("yolo", "sahi"):
        without_result = by_name.get(f"{prefix}_without_exclusion")
        with_result = by_name.get(f"{prefix}_with_exclusion")
        if without_result is None or with_result is None:
            continue
        cpu_delta = with_result.process_cpu_percent - without_result.process_cpu_percent
        ram_delta = None
        if with_result.ram_peak_mb is not None and without_result.ram_peak_mb is not None:
            ram_delta = round(with_result.ram_peak_mb - without_result.ram_peak_mb, 3)
        frame_ms_delta = None
        if with_result.average_frame_ms is not None and without_result.average_frame_ms is not None:
            frame_ms_delta = round(with_result.average_frame_ms - without_result.average_frame_ms, 3)
        comparisons[prefix] = {
            "cpu_percent_delta": round(cpu_delta, 3),
            "cpu_percent_reduction": round((-cpu_delta / without_result.process_cpu_percent) * 100.0, 3) if without_result.process_cpu_percent else None,
            "ram_peak_mb_delta": ram_delta,
            "average_frame_ms_delta": frame_ms_delta,
            "average_frame_ms_reduction": round((-(frame_ms_delta or 0.0) / without_result.average_frame_ms) * 100.0, 3)
            if without_result.average_frame_ms
            else None,
        }
    return comparisons


def results_to_payload(results: list[BenchmarkResult]) -> dict[str, object]:
    """Convert benchmark results and comparisons to JSON-serializable data."""

    return {
        "results": [asdict(result) for result in results],
        "comparisons": compare_results(results),
    }


DEFAULT_EXCLUSION_ZONE_POINTS: ZonePoints = list(EXCLUSION_ZONE_POINTS)
