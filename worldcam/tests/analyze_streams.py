"""Analyze video stream health for URLs declared in worldcam.config.

Runnable without pytest:

    python -m worldcam.tests.analyze_streams --duration 20
"""

from __future__ import annotations

import json

from worldcam.tests.stream_analysis_cli import parse_args, parse_profiles, select_streams
from worldcam.tests.stream_analysis_metrics import analyze_stream
from worldcam.tests.stream_analysis_probe import probe_source
from worldcam.tests.stream_analysis_report import build_json_payload, format_report
from worldcam.tests.stream_analysis_types import SourceProbe, StreamAnalysis


def analyze_selected_streams(
    selected_streams: list[tuple[int, str]],
    profiles: list[tuple[str, int, int, int]],
    duration_seconds: float,
    max_frames: int,
    verbose: bool = True,
    progress_interval: float = 2.0,
) -> dict[int, tuple[str, SourceProbe, list[StreamAnalysis]]]:
    """Probe each selected stream and run every requested decode profile."""

    results_by_stream: dict[int, tuple[str, SourceProbe, list[StreamAnalysis]]] = {}
    total_profiles = len(selected_streams) * len(profiles)
    completed_profiles = 0
    for stream_position, (stream_index, url) in enumerate(selected_streams, start=1):
        if verbose:
            print(f"\nStream {stream_index} ({stream_position}/{len(selected_streams)})", flush=True)
            print("Probing source metadata with ffprobe...", flush=True)
        source_probe = probe_source(url)
        if verbose:
            if source_probe.hls.available and source_probe.hls.is_master_playlist:
                print(
                    f"HLS max: resolution={source_probe.hls.max_width}x{source_probe.hls.max_height}, "
                    f"fps={source_probe.hls.max_fps}, bitrate_kbps={source_probe.hls.max_bandwidth_kbps}, "
                    f"variants={source_probe.hls.variant_count}",
                    flush=True,
                )
            if source_probe.available:
                print(
                    f"ffprobe selected: codec={source_probe.codec}, resolution={source_probe.width}x{source_probe.height}, "
                    f"fps={source_probe.fps}, bitrate_kbps={source_probe.video_bitrate_kbps or source_probe.format_bitrate_kbps}",
                    flush=True,
                )
            else:
                print(f"Source metadata unavailable: {'; '.join(source_probe.notes)}", flush=True)

        stream_results = []
        for profile_name, width, height, fps in profiles:
            completed_profiles += 1
            if verbose:
                print(f"\nProfile {completed_profiles}/{total_profiles}: {profile_name}", flush=True)
            stream_results.append(
                analyze_stream(
                    stream_index=stream_index,
                    url=url,
                    profile_name=profile_name,
                    width=width,
                    height=height,
                    target_fps=fps,
                    duration_seconds=duration_seconds,
                    max_frames=max_frames,
                    verbose=verbose,
                    progress_interval=progress_interval,
                )
            )
        results_by_stream[stream_index] = (url, source_probe, stream_results)
    return results_by_stream


def main() -> int:
    """Run stream analysis and optionally write a JSON report."""

    args = parse_args()
    profiles = parse_profiles(args.profiles)
    selected_streams = select_streams(args.stream_index)
    results_by_stream = analyze_selected_streams(
        selected_streams,
        profiles,
        args.duration,
        args.max_frames,
        verbose=not args.quiet,
        progress_interval=args.progress_interval,
    )

    print(format_report(results_by_stream))

    if args.json_output is not None:
        payload = build_json_payload(profiles, results_by_stream)
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON report written to {args.json_output}")

    all_results = [result for _url, _source_probe, results in results_by_stream.values() for result in results]
    return 0 if all(result.opened and result.frames_read > 0 for result in all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
