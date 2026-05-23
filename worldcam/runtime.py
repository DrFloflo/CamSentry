"""Runtime state and stream statistics helpers for WorldCam."""

from dataclasses import dataclass, field
import time

from worldcam.config import READ_WARN_SECONDS, STATS_LOG_SECONDS
from worldcam.detection import Detection
from worldcam.pose import Pose
from worldcam.segmentation import SegmentationMask
from worldcam.tracking import ObjectTrack, ObjectTracker


@dataclass
class RuntimeState:
    """Mutable state for frame analysis results and stream statistics."""

    latest_detections: list[Detection] = field(default_factory=list)
    latest_poses: list[Pose] = field(default_factory=list)
    latest_segmentations: list[SegmentationMask] = field(default_factory=list)
    latest_object_tracks: list[ObjectTrack] = field(default_factory=list)
    frame_count: int = 0
    slow_reads: int = 0
    stream_read_failures: int = 0
    started_at: float = field(default_factory=time.perf_counter)
    last_stats_at: float = field(default_factory=time.perf_counter)
    next_frame_at: float = field(default_factory=time.perf_counter)
    last_frame_at: float = field(default_factory=time.perf_counter)
    current_fps: float = 0.0


def reset_analysis_state(runtime: RuntimeState, object_tracker: ObjectTracker) -> None:
    """Clear cached analysis outputs and reset tracking state."""
    runtime.latest_detections = []
    runtime.latest_poses = []
    runtime.latest_segmentations = []
    runtime.latest_object_tracks = []
    object_tracker.reset()


def reset_stream_statistics(runtime: RuntimeState) -> None:
    """Reset frame counters and timing statistics for a fresh stream."""
    now = time.perf_counter()
    runtime.frame_count = 0
    runtime.slow_reads = 0
    runtime.stream_read_failures = 0
    runtime.started_at = now
    runtime.last_stats_at = now
    runtime.next_frame_at = now
    runtime.last_frame_at = now
    runtime.current_fps = 0.0


def register_stream_read(runtime: RuntimeState, read_duration: float) -> None:
    """Update slow-read diagnostics for one stream read attempt."""
    if read_duration > READ_WARN_SECONDS:
        runtime.slow_reads += 1
        print(f"Lecture lente: {read_duration:.3f}s pour récupérer une frame.")


def register_stream_read_failure(runtime: RuntimeState, max_failures: int) -> bool:
    """Record a failed read and return whether reconnection should be attempted."""
    runtime.stream_read_failures += 1
    print(f"Lecture flux indisponible ({runtime.stream_read_failures}/{max_failures}).")
    return runtime.stream_read_failures >= max_failures


def register_frame_received(runtime: RuntimeState, read_duration: float) -> None:
    """Update frame counters, FPS, and periodic stream statistics."""
    runtime.stream_read_failures = 0
    runtime.frame_count += 1
    now = time.perf_counter()
    frame_delta = now - runtime.last_frame_at
    if frame_delta > 0:
        runtime.current_fps = 1.0 / frame_delta
    runtime.last_frame_at = now

    if now - runtime.last_stats_at >= STATS_LOG_SECONDS:
        elapsed = now - runtime.started_at
        average_fps = runtime.frame_count / elapsed if elapsed > 0 else 0.0
        print(
            f"Stats flux: frames={runtime.frame_count}, fps_moyen={average_fps:.1f}, "
            f"lectures_lentes={runtime.slow_reads}, derniere_lecture={read_duration:.3f}s"
        )
        runtime.last_stats_at = now
