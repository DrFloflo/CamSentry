"""CLI parsing helpers for WorldCam stream analysis tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from worldcam.core.config import OUTPUT_HEIGHT, OUTPUT_WIDTH, STREAM_URLS, TARGET_FPS


def parse_args() -> argparse.Namespace:
    """Parse CLI options for stream analysis."""

    parser = argparse.ArgumentParser(description="Analyze streams configured in worldcam.core.config.STREAM_URLS.")
    parser.add_argument("--duration", type=float, default=15.0, help="Seconds to sample each decode profile.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional max frames per profile; 0 means no cap.")
    parser.add_argument("--stream-index", type=int, default=-1, help="Analyze only this zero-based stream index; -1 means all.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional path where the JSON report is written.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logs while profiles are being sampled.")
    parser.add_argument("--progress-interval", type=float, default=2.0, help="Seconds between progress logs during sampling.")
    parser.add_argument(
        "--profiles",
        default=f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}@15,{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}@{TARGET_FPS},{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}@native",
        help="Comma-separated decode profiles as WIDTHxHEIGHT@FPS or WIDTHxHEIGHT@native.",
    )
    return parser.parse_args()


def parse_profiles(profile_text: str) -> list[tuple[str, int, int, int]]:
    """Parse WIDTHxHEIGHT@FPS decode profile definitions."""

    profiles: list[tuple[str, int, int, int]] = []
    for raw_profile in profile_text.split(","):
        profile = raw_profile.strip().lower()
        if not profile:
            continue
        try:
            size_text, fps_text = profile.split("@", 1)
            width_text, height_text = size_text.split("x", 1)
            width = int(width_text)
            height = int(height_text)
            fps = 0 if fps_text == "native" else int(fps_text)
        except ValueError as exc:
            raise ValueError(f"Invalid profile '{raw_profile}'. Expected WIDTHxHEIGHT@FPS or WIDTHxHEIGHT@native.") from exc
        if width <= 0 or height <= 0 or fps < 0:
            raise ValueError(f"Invalid profile '{raw_profile}'. Width and height must be positive; FPS must be positive or native.")
        profile_fps_text = "native" if fps == 0 else str(fps)
        profiles.append((f"{width}x{height}@{profile_fps_text}", width, height, fps))
    if not profiles:
        raise ValueError("At least one decode profile is required.")
    return profiles


def select_streams(stream_index: int) -> list[tuple[int, str]]:
    """Return selected configured stream indexes and URLs."""

    indexed_streams = list(enumerate(STREAM_URLS))
    if stream_index < 0:
        return indexed_streams
    if stream_index >= len(indexed_streams):
        raise ValueError(f"stream-index {stream_index} is outside configured range 0..{len(indexed_streams) - 1}")
    return [indexed_streams[stream_index]]
