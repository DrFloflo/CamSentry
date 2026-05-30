"""Benchmark CPU/RAM impact of WorldCam exclusion-zone processing.

Examples:

    python -m worldcam.tests.exclusion_benchmark.benchmark_exclusion --duration 20
    python -m worldcam.tests.exclusion_benchmark.benchmark_exclusion --duration 20 --no-sahi
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldcam.core.config import DEFAULT_MODEL_KEY, OUTPUT_HEIGHT, OUTPUT_WIDTH, STREAM_URLS, TARGET_FPS
from worldcam.core.models import load_yolo_model
from worldcam.tests.exclusion_benchmark.exclusion_benchmark_metrics import (
    DEFAULT_EXCLUSION_ZONE_POINTS,
    benchmark_scenario,
    build_class_selection,
    build_scenarios,
    collect_benchmark_frames,
    results_to_payload,
)
from worldcam.tests.stream_analysis.stream_analysis_cli import parse_profiles, select_streams


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the exclusion-zone benchmark."""

    parser = argparse.ArgumentParser(description="Measure CPU/RAM difference with and without WorldCam exclusion-zone processing.")
    parser.add_argument("--duration", type=float, default=15.0, help="Seconds of stream frames to collect once and replay through every scenario.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional measured frame cap; 0 means duration-limited.")
    parser.add_argument("--warmup-frames", type=int, default=3, help="Frames processed before measurements for every scenario.")
    parser.add_argument("--stream-index", type=int, default=0, help="Zero-based stream index from worldcam.core.config.STREAM_URLS.")
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY, help="YOLO model key used from models/ without the 'yolo' prefix.")
    parser.add_argument(
        "--profile",
        default=f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}@{TARGET_FPS}",
        help="Decode profile as WIDTHxHEIGHT@FPS or WIDTHxHEIGHT@native. Only the first parsed profile is used.",
    )
    parser.add_argument("--json-output", type=Path, default=None, help="Optional path where the JSON benchmark report is written.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logs.")
    parser.add_argument("--no-sahi", action="store_true", help="Skip SAHI scenarios and benchmark YOLO-only variants.")
    return parser.parse_args()


def format_result_table(payload: dict[str, object]) -> str:
    """Return a compact human-readable benchmark report."""

    lines = ["\nExclusion-zone CPU/RAM benchmark", "=" * 38]
    results = payload["results"]
    for result in results:
        lines.append(
            f"{result['scenario']}: frames={result['frames_processed']}, "
            f"fps={result['frames_per_second']}, avg_ms={result['average_frame_ms']}, "
            f"p95_ms={result['p95_frame_ms']}, cpu={result['process_cpu_percent']}%, "
            f"cpu/core={result['process_cpu_percent_per_core']}%, "
            f"ram_start={result['ram_start_mb']} MiB, ram_peak={result['ram_peak_mb']} MiB, "
            f"ram_delta={result['ram_delta_mb']} MiB, detections/frame={result['detections_per_frame']}"
        )
        for note in result["notes"]:
            lines.append(f"  note: {note}")

    comparisons = payload["comparisons"]
    if comparisons:
        lines.extend(["", "Comparisons", "-----------"])
        for name, comparison in comparisons.items():
            lines.append(
                f"{name}: cpu_delta={comparison['cpu_percent_delta']}%, "
                f"cpu_reduction={comparison['cpu_percent_reduction']}%, "
                f"ram_peak_delta={comparison['ram_peak_mb_delta']} MiB, "
                f"avg_ms_delta={comparison['average_frame_ms_delta']}, "
                f"avg_ms_reduction={comparison['average_frame_ms_reduction']}%"
            )
    return "\n".join(lines)


def main() -> int:
    """Collect frames once and run all requested benchmark scenarios."""

    args = parse_args()
    selected_streams = select_streams(args.stream_index)
    if not selected_streams:
        raise ValueError("No stream selected.")
    stream_index, url = selected_streams[0]
    profile_name, width, height, target_fps = parse_profiles(args.profile)[0]

    verbose = not args.quiet
    if verbose:
        print(f"Selected stream {stream_index}/{len(STREAM_URLS) - 1}: {url}", flush=True)
        print(f"Collecting profile {profile_name} for {args.duration:.1f}s...", flush=True)

    warmup_frames, frames = collect_benchmark_frames(
        url=url,
        width=width,
        height=height,
        target_fps=target_fps,
        duration_seconds=args.duration,
        max_frames=args.max_frames,
        warmup_frames=max(0, args.warmup_frames),
        progress=(lambda message: print(f"  ... {message}", flush=True)) if verbose else None,
    )
    if verbose:
        print(f"Collected warmup_frames={len(warmup_frames)}, measured_frames={len(frames)}", flush=True)

    model_key = args.model.strip().removeprefix("yolo")
    model, device = load_yolo_model(model_key)
    selected_class_names = build_class_selection(model)
    scenarios = build_scenarios(include_sahi=not args.no_sahi)

    results = [
        benchmark_scenario(
            scenario=scenario,
            frames=frames,
            warmup_frames=warmup_frames,
            model=model,
            device=device,
            selected_class_names=selected_class_names,
            exclusion_zone_points=DEFAULT_EXCLUSION_ZONE_POINTS,
            stream_index=stream_index,
            profile_name=profile_name,
            width=width,
            height=height,
            verbose=verbose,
        )
        for scenario in scenarios
    ]
    payload = results_to_payload(results)
    print(format_result_table(payload))

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON benchmark written to {args.json_output}")

    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
