"""Shared data structures for WorldCam stream analysis tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HlsVariant:
    """One variant advertised by an HLS master playlist."""

    uri: str | None
    width: int | None
    height: int | None
    fps: float | None
    bandwidth_kbps: float | None
    codecs: str | None


@dataclass(frozen=True)
class HlsProbe:
    """Maximum capabilities advertised by the HLS manifest."""

    available: bool
    is_master_playlist: bool
    variant_count: int
    max_width: int | None
    max_height: int | None
    max_fps: float | None
    max_bandwidth_kbps: float | None
    variants: list[HlsVariant]
    notes: list[str]


@dataclass(frozen=True)
class SourceProbe:
    """Metadata reported by the HLS manifest and ffprobe before decoding the stream."""

    available: bool
    codec: str | None
    width: int | None
    height: int | None
    fps: float | None
    video_bitrate_kbps: float | None
    format_bitrate_kbps: float | None
    hls: HlsProbe
    raw: dict[str, Any] | None
    notes: list[str]


@dataclass(frozen=True)
class StreamAnalysis:
    """Measured health metrics for one video stream decode profile."""

    stream_index: int
    url: str
    profile: str
    opened: bool
    sampled_seconds: float
    frames_read: int
    read_failures: int
    width: int | None
    height: int | None
    capture_fps_hint: float | None
    decoded_fps: float
    fps_vs_target_ratio: float | None
    average_read_latency_ms: float | None
    p95_read_latency_ms: float | None
    average_frame_interval_ms: float | None
    frame_interval_jitter_ms: float | None
    stability_score: float
    estimated_decoded_mbps: float
    average_brightness: float | None
    average_contrast: float | None
    average_sharpness: float | None
    frozen_frame_ratio: float | None
    notes: list[str]
