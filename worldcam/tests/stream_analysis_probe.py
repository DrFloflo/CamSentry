"""Source probing helpers for WorldCam stream analysis tests."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Any
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from worldcam.core.config import FFMPEG_HEADERS, REFERER, USER_AGENT
from worldcam.tests.stream_analysis_types import HlsProbe, HlsVariant, SourceProbe


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


def unavailable_hls_probe(note: str) -> HlsProbe:
    """Build an unavailable HLS probe with one diagnostic note."""

    return HlsProbe(
        available=False,
        is_master_playlist=False,
        variant_count=0,
        max_width=None,
        max_height=None,
        max_fps=None,
        max_bandwidth_kbps=None,
        variants=[],
        notes=[note],
    )


def parse_hls_attributes(attribute_text: str) -> dict[str, str]:
    """Parse an HLS attribute list from an EXT-X-STREAM-INF line."""

    attributes: dict[str, str] = {}
    for match in re.finditer(r'([A-Z0-9-]+)=((?:"[^"]*")|[^,]*)', attribute_text):
        key = match.group(1)
        value = match.group(2).strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        attributes[key] = value
    return attributes


def parse_resolution(value: str | None) -> tuple[int | None, int | None]:
    """Parse an HLS RESOLUTION value."""

    if not value or "x" not in value:
        return None, None
    width_text, height_text = value.lower().split("x", 1)
    try:
        width = int(width_text)
        height = int(height_text)
    except ValueError:
        return None, None
    return (width if width > 0 else None, height if height > 0 else None)


def fetch_hls_manifest(url: str) -> str:
    """Fetch an HLS manifest using EarthCam-compatible HTTP headers."""

    request = Request(url, headers={"User-Agent": USER_AGENT, "Referer": REFERER, "Accept": "*/*"})
    with urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8", errors="replace")


def probe_hls_manifest(url: str) -> HlsProbe:
    """Read the HLS manifest and extract advertised variants and maxima."""

    try:
        manifest = fetch_hls_manifest(url)
    except Exception as exc:
        return unavailable_hls_probe(f"HLS manifest fetch failed: {type(exc).__name__}: {exc}")

    lines = [line.strip() for line in manifest.splitlines() if line.strip()]
    variants: list[HlsVariant] = []
    for index, line in enumerate(lines):
        if not line.startswith("#EXT-X-STREAM-INF:"):
            continue
        attributes = parse_hls_attributes(line.split(":", 1)[1])
        width, height = parse_resolution(attributes.get("RESOLUTION"))
        fps = parse_rate(attributes.get("FRAME-RATE"))
        bandwidth = parse_optional_int(attributes.get("BANDWIDTH"))
        uri = None
        if index + 1 < len(lines) and not lines[index + 1].startswith("#"):
            uri = urljoin(url, lines[index + 1])
        variants.append(
            HlsVariant(
                uri=uri,
                width=width,
                height=height,
                fps=round(fps, 3) if fps is not None else None,
                bandwidth_kbps=round(bandwidth / 1000.0, 1) if bandwidth is not None else None,
                codecs=attributes.get("CODECS"),
            )
        )

    if not variants:
        return HlsProbe(
            available=True,
            is_master_playlist=False,
            variant_count=0,
            max_width=None,
            max_height=None,
            max_fps=None,
            max_bandwidth_kbps=None,
            variants=[],
            notes=["Manifest is probably a media playlist, not a master playlist; no HLS variants were advertised."],
        )

    best_resolution = max(
        variants,
        key=lambda variant: ((variant.width or 0) * (variant.height or 0), variant.bandwidth_kbps or 0.0),
    )
    max_fps = max((variant.fps for variant in variants if variant.fps is not None), default=None)
    max_bandwidth = max((variant.bandwidth_kbps for variant in variants if variant.bandwidth_kbps is not None), default=None)
    return HlsProbe(
        available=True,
        is_master_playlist=True,
        variant_count=len(variants),
        max_width=best_resolution.width,
        max_height=best_resolution.height,
        max_fps=max_fps,
        max_bandwidth_kbps=max_bandwidth,
        variants=variants,
        notes=[],
    )


def unavailable_probe(note: str, hls_probe: HlsProbe | None = None) -> SourceProbe:
    """Build an unavailable source-probe result with one diagnostic note."""

    return SourceProbe(
        available=False,
        codec=None,
        width=None,
        height=None,
        fps=None,
        video_bitrate_kbps=None,
        format_bitrate_kbps=None,
        hls=hls_probe or unavailable_hls_probe("HLS manifest was not probed."),
        raw=None,
        notes=[note],
    )


def run_ffprobe(url: str, hls_probe: HlsProbe) -> dict[str, Any] | SourceProbe:
    """Run ffprobe and return parsed JSON or a failure probe."""

    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        return unavailable_probe("ffprobe.exe not found in PATH; source metadata cannot be inspected.", hls_probe)

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
        return unavailable_probe("ffprobe timed out while inspecting the HLS source.", hls_probe)

    if completed.returncode != 0:
        return unavailable_probe(completed.stderr.strip() or "ffprobe failed without stderr output.", hls_probe)

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return unavailable_probe(f"ffprobe returned invalid JSON: {exc}", hls_probe)


def probe_source(url: str) -> SourceProbe:
    """Use the HLS manifest and ffprobe to read source capabilities and active stream metadata."""

    hls_probe = probe_hls_manifest(url)
    raw_or_failure = run_ffprobe(url, hls_probe)
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
            hls=hls_probe,
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
        hls=hls_probe,
        raw=raw,
        notes=[] if video_bitrate is not None or format_bitrate is not None else ["ffprobe did not expose compressed bitrate."],
    )
