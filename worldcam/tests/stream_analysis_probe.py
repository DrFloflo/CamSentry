"""Source probing helpers for WorldCam stream analysis tests."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from worldcam.config import FFMPEG_HEADERS
from worldcam.tests.stream_analysis_types import SourceProbe


def parse_rate(rate: str | None) -> float | None:
    """Parse an ffprobe frame-rate fraction such as 30000/1001."""

    if not rate or rate == "0/0":
        return None
    if "/" not in rate:
        try:
            value = float(rate)
            return value if value > 0 else None
        except ValueError:
            return None
    numerator_text, denominator_text = rate.split("/", 1)
    try:
        numerator = float(numerator_text)
        denominator = float(denominator_text)
    except ValueError:
        return None
    if denominator == 0:
        return None
    value = numerator / denominator
    return value if value > 0 else None


def parse_optional_int(value: Any) -> int | None:
    """Parse an optional integer-like ffprobe value."""

    if value in (None, "N/A", ""):
        return None
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def unavailable_probe(note: str) -> SourceProbe:
    """Build an unavailable source-probe result with one diagnostic note."""

    return SourceProbe(
        available=False,
        codec=None,
        width=None,
        height=None,
        fps=None,
        video_bitrate_kbps=None,
        format_bitrate_kbps=None,
        raw=None,
        notes=[note],
    )


def run_ffprobe(url: str) -> dict[str, Any] | SourceProbe:
    """Run ffprobe and return parsed JSON or a failure probe."""

    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        return unavailable_probe("ffprobe.exe not found in PATH; source metadata cannot be inspected.")

    command = [
        ffprobe_path,
        "-v",
        "error",
        "-headers",
        FFMPEG_HEADERS,
        "-show_entries",
        "stream=index,codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,bit_rate:format=bit_rate",
        "-of",
        "json",
        url,
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    except subprocess.TimeoutExpired:
        return unavailable_probe("ffprobe timed out while inspecting the HLS source.")

    if completed.returncode != 0:
        return unavailable_probe(completed.stderr.strip() or "ffprobe failed without stderr output.")

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return unavailable_probe(f"ffprobe returned invalid JSON: {exc}")


def probe_source(url: str) -> SourceProbe:
    """Use ffprobe to read source resolution, FPS, codec and bitrate when available."""

    raw_or_failure = run_ffprobe(url)
    if isinstance(raw_or_failure, SourceProbe):
        return raw_or_failure

    raw = raw_or_failure
    format_bitrate = parse_optional_int(raw.get("format", {}).get("bit_rate"))
    video_streams = [stream for stream in raw.get("streams", []) if stream.get("codec_type") == "video"]
    if not video_streams:
        return SourceProbe(
            available=False,
            codec=None,
            width=None,
            height=None,
            fps=None,
            video_bitrate_kbps=None,
            format_bitrate_kbps=round(format_bitrate / 1000.0, 1) if format_bitrate is not None else None,
            raw=raw,
            notes=["ffprobe did not report a video stream."],
        )

    video = video_streams[0]
    video_bitrate = parse_optional_int(video.get("bit_rate"))
    fps = parse_rate(video.get("avg_frame_rate")) or parse_rate(video.get("r_frame_rate"))
    return SourceProbe(
        available=True,
        codec=video.get("codec_name"),
        width=parse_optional_int(video.get("width")),
        height=parse_optional_int(video.get("height")),
        fps=round(fps, 3) if fps is not None else None,
        video_bitrate_kbps=round(video_bitrate / 1000.0, 1) if video_bitrate is not None else None,
        format_bitrate_kbps=round(format_bitrate / 1000.0, 1) if format_bitrate is not None else None,
        raw=raw,
        notes=[] if video_bitrate is not None or format_bitrate is not None else ["ffprobe did not expose compressed bitrate."],
    )
