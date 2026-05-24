"""Console and JSON reporting helpers for WorldCam stream analysis tests."""

from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any

from worldcam.config import OUTPUT_HEIGHT, OUTPUT_WIDTH, TARGET_FPS
from worldcam.tests.stream_analysis_types import SourceProbe, StreamAnalysis


def format_source_probe(source_probe: SourceProbe) -> list[str]:
    """Create human-readable source metadata lines."""

    lines = ["  Source metadata:"]
    if source_probe.available:
        lines.append(
            "    "
            f"codec={source_probe.codec}, resolution={source_probe.width}x{source_probe.height}, "
            f"fps={source_probe.fps}, video_bitrate_kbps={source_probe.video_bitrate_kbps}, "
            f"format_bitrate_kbps={source_probe.format_bitrate_kbps}"
        )
    else:
        lines.append("    unavailable")
    for note in source_probe.notes:
        lines.append(f"    - {note}")
    return lines


def format_profile_result(result: StreamAnalysis) -> list[str]:
    """Create human-readable lines for one decoded profile result."""

    lines = [
        f"  Profile {result.profile}: {'opened' if result.opened else 'not opened'}",
        f"    Sample: {result.sampled_seconds}s, frames={result.frames_read}, failures={result.read_failures}",
        f"    Output resolution: {result.width}x{result.height}, requested_fps={result.capture_fps_hint}",
        (
            "    FPS: "
            f"decoded={result.decoded_fps}, target_ratio={result.fps_vs_target_ratio}, "
            f"avg_interval_ms={result.average_frame_interval_ms}, jitter_ms={result.frame_interval_jitter_ms}"
        ),
        f"    Latency: avg_read_ms={result.average_read_latency_ms}, p95_read_ms={result.p95_read_latency_ms}",
        (
            "    Quality proxies: "
            f"brightness={result.average_brightness}, contrast={result.average_contrast}, "
            f"sharpness={result.average_sharpness}, frozen_ratio={result.frozen_frame_ratio}"
        ),
        f"    Decoded raw throughput: {result.estimated_decoded_mbps} Mbps",
        f"    Stability score: {result.stability_score}/100",
    ]
    if result.notes:
        lines.append("    Notes:")
        lines.extend(f"      - {note}" for note in result.notes)
    return lines


def format_report(results_by_stream: dict[int, tuple[str, SourceProbe, list[StreamAnalysis]]]) -> str:
    """Create a readable text report for console output."""

    lines = ["WorldCam stream analysis", "========================", ""]
    for stream_index, (url, source_probe, results) in results_by_stream.items():
        lines.append(f"Stream {stream_index}")
        lines.append(f"  URL: {url}")
        lines.extend(format_source_probe(source_probe))
        for result in results:
            lines.extend(format_profile_result(result))
        lines.append("")
    return "\n".join(lines)


def build_json_payload(
    profiles: list[tuple[str, int, int, int]],
    results_by_stream: dict[int, tuple[str, SourceProbe, list[StreamAnalysis]]],
) -> dict[str, Any]:
    """Build the JSON-serializable stream analysis report."""

    return {
        "configured_output": {"width": OUTPUT_WIDTH, "height": OUTPUT_HEIGHT, "target_fps": TARGET_FPS},
        "tested_profiles": [profile_name for profile_name, _width, _height, _fps in profiles],
        "generated_at_unix": time.time(),
        "streams": [
            {
                "stream_index": stream_index,
                "url": url,
                "source_probe": asdict(source_probe),
                "profile_results": [asdict(result) for result in results],
            }
            for stream_index, (url, source_probe, results) in results_by_stream.items()
        ],
    }
