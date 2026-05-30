"""Shared data structures for exclusion-zone CPU/RAM benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

from worldcam.analysis.counting_zone import ZonePoints


@dataclass(frozen=True)
class BenchmarkScenario:
    """One processing variant measured by the benchmark."""

    name: str
    sahi_enabled: bool
    exclusion_enabled: bool


@dataclass(frozen=True)
class BenchmarkResult:
    """Measured CPU/RAM and throughput for one benchmark scenario."""

    scenario: str
    sahi_enabled: bool
    exclusion_enabled: bool
    stream_index: int
    profile: str
    width: int
    height: int
    frames_processed: int
    sampled_seconds: float
    frames_per_second: float
    average_frame_ms: float | None
    p95_frame_ms: float | None
    process_cpu_percent: float
    process_cpu_percent_per_core: float
    ram_start_mb: float | None
    ram_peak_mb: float | None
    ram_delta_mb: float | None
    detections_total: int
    detections_per_frame: float
    exclusion_zone_points: ZonePoints
    notes: list[str]
