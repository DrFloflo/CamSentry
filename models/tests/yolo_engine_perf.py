"""
Benchmark YOLO TensorRT engine performance on the local machine.

Metrics reported:
- frames per second (FPS)
- average wall-clock time per frame
- average Ultralytics inference time per frame
- RAM usage
- CPU usage
- GPU utilization when available through nvidia-smi
- VRAM usage when available through nvidia-smi or torch.cuda

Examples from the project root:
    python models/tests/yolo_engine_perf.py --engine models/yolo26m.engine
    python models/tests/yolo_engine_perf.py --engine models/yolo26m.engine --source 0
    python models/tests/yolo_engine_perf.py --engine models/yolo26m.engine --source video.mp4 --frames 500
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

try:
    from worldcam.config import (
        FFMPEG_ANALYZEDURATION_US,
        FFMPEG_HEADERS,
        FFMPEG_HWACCEL,
        FFMPEG_INPUT_REALTIME,
        FFMPEG_MAX_DELAY_US,
        FFMPEG_PROBESIZE,
        FFMPEG_RECONNECT_DELAY_MAX_SECONDS,
        FFMPEG_THREAD_QUEUE_SIZE,
        FRAME_SIZE as WORLDCAM_FRAME_SIZE,
        ORIGIN,
        OUTPUT_HEIGHT as WORLDCAM_OUTPUT_HEIGHT,
        OUTPUT_WIDTH as WORLDCAM_OUTPUT_WIDTH,
        REFERER,
        USER_AGENT,
    )
except ImportError:  # pragma: no cover - keep this benchmark usable outside the app package
    FFMPEG_ANALYZEDURATION_US = 1_000_000
    FFMPEG_HEADERS = ""
    FFMPEG_HWACCEL = ""
    FFMPEG_INPUT_REALTIME = True
    FFMPEG_MAX_DELAY_US = 500_000
    FFMPEG_PROBESIZE = "512k"
    FFMPEG_RECONNECT_DELAY_MAX_SECONDS = 2
    FFMPEG_THREAD_QUEUE_SIZE = 512
    WORLDCAM_OUTPUT_WIDTH = 1280
    WORLDCAM_OUTPUT_HEIGHT = 720
    WORLDCAM_FRAME_SIZE = WORLDCAM_OUTPUT_WIDTH * WORLDCAM_OUTPUT_HEIGHT * 3
    ORIGIN = "https://www.earthcam.com"
    REFERER = "https://www.earthcam.com/world/ireland/dublin/"
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


@dataclass
class ResourceSample:
    timestamp: float
    cpu_percent: float | None = None
    ram_used_mb: float | None = None
    ram_percent: float | None = None
    gpu_util_percent: float | None = None
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None
    torch_cuda_allocated_mb: float | None = None
    torch_cuda_reserved_mb: float | None = None


@dataclass
class BenchmarkStats:
    engine: str
    source: str
    frames: int
    warmup: int
    image_size: int
    device: str
    confidence: float
    total_wall_time_s: float
    fps: float
    avg_wall_time_ms: float
    avg_preprocess_ms: float | None
    avg_inference_ms: float | None
    avg_postprocess_ms: float | None
    avg_model_total_ms: float | None
    resources: dict[str, float | None] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a YOLO TensorRT .engine file with FPS and system resource metrics."
    )
    parser.add_argument(
        "--engine",
        default="models/yolo26m.engine",
        help="Path to the YOLO TensorRT engine file.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional source: camera index, video path, image path. If omitted, random frames are used.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Number of measured frames after warmup.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Number of warmup frames excluded from measurements.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size passed to Ultralytics.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device passed to Ultralytics, for example 0, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold passed to Ultralytics.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.25,
        help="Resource sampling interval in seconds.",
    )
    parser.add_argument(
        "--ffmpeg-width",
        type=int,
        default=WORLDCAM_OUTPUT_WIDTH,
        help="Width of frames decoded by the external FFmpeg fallback.",
    )
    parser.add_argument(
        "--ffmpeg-height",
        type=int,
        default=WORLDCAM_OUTPUT_HEIGHT,
        help="Height of frames decoded by the external FFmpeg fallback.",
    )
    parser.add_argument(
        "--ffmpeg-fps",
        type=int,
        default=0,
        help="Optional FPS limit for the external FFmpeg fallback. 0 keeps the input rate.",
    )
    parser.add_argument(
        "--ffmpeg-input-format",
        default=None,
        help="Optional FFmpeg input format, for example dshow on Windows or v4l2 on Linux.",
    )
    parser.add_argument(
        "--force-ffmpeg",
        action="store_true",
        help="Skip OpenCV VideoCapture and read the source directly through external ffmpeg.exe.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV path where raw resource samples are written.",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Optional JSON path where the final benchmark summary is written.",
    )
    return parser.parse_args()


def source_label(source: str | None) -> str:
    if source is None:
        return "random"
    return str(source)


def configure_opencv_ffmpeg_headers() -> None:
    """Configure OpenCV's FFmpeg backend with the same browser-like headers as the app."""
    extra_headers = "\r\n".join([f"Origin: {ORIGIN}", "Accept: */*", ""])
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"user_agent;{USER_AGENT}"
        f"|referer;{REFERER}"
        f"|headers;{extra_headers}"
    )


def is_numeric_camera_source(source: str | None) -> bool:
    return source is not None and source.isdigit()


def open_capture(source: str | None, force_ffmpeg: bool) -> cv2.VideoCapture | None:
    if source is None or force_ffmpeg:
        return None
    capture_source: int | str
    capture_source = int(source) if is_numeric_camera_source(source) else source
    backend = cv2.CAP_FFMPEG if isinstance(capture_source, str) else cv2.CAP_ANY
    capture = cv2.VideoCapture(capture_source, backend)
    if capture.isOpened():
        print("Connexion réussie via OpenCV/FFmpeg.")
        return capture

    capture.release()
    if isinstance(capture_source, int):
        raise RuntimeError(
            f"Camera index {capture_source} is not available through OpenCV. "
            "Numeric camera indexes are not passed to FFmpeg as filenames. "
            "Use an existing camera index, omit --source to benchmark random frames, "
            "or use a real video/stream URL for FFmpeg fallback."
        )

    print("OpenCV/FFmpeg n'ouvre pas cette source; fallback vers ffmpeg.exe.")
    return None


def build_ffmpeg_command(args: argparse.Namespace) -> list[str]:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg.exe est requis pour ce fallback, mais il est introuvable dans le PATH.")
    if args.source is None:
        raise RuntimeError("A --source value is required when using the FFmpeg fallback.")

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-fflags",
        "nobuffer+discardcorrupt",
        "-flags",
        "low_delay",
        "-max_delay",
        str(FFMPEG_MAX_DELAY_US),
        "-probesize",
        str(FFMPEG_PROBESIZE),
        "-analyzeduration",
        str(FFMPEG_ANALYZEDURATION_US),
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        str(FFMPEG_RECONNECT_DELAY_MAX_SECONDS),
        "-thread_queue_size",
        str(FFMPEG_THREAD_QUEUE_SIZE),
    ]
    if FFMPEG_HWACCEL:
        command.extend(["-hwaccel", FFMPEG_HWACCEL])
    if FFMPEG_INPUT_REALTIME:
        command.append("-re")
    if args.ffmpeg_input_format:
        command.extend(["-f", args.ffmpeg_input_format])
    if FFMPEG_HEADERS:
        command.extend(["-headers", FFMPEG_HEADERS])

    command.extend(["-i", args.source, "-an"])
    filters = [f"scale={args.ffmpeg_width}:{args.ffmpeg_height}:flags=bicubic"]
    if args.ffmpeg_fps > 0:
        filters.append(f"fps={args.ffmpeg_fps}")
    command.extend(["-vf", ",".join(filters), "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"])
    return command


def start_ffmpeg_pipe(args: argparse.Namespace) -> subprocess.Popen:
    command = build_ffmpeg_command(args)
    print("Connexion via ffmpeg.exe fallback.")
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def read_ffmpeg_frame(
    process: subprocess.Popen,
    width: int,
    height: int,
) -> tuple[bool, np.ndarray | None]:
    if process.stdout is None:
        return False, None

    frame_size = width * height * 3
    raw_frame = process.stdout.read(frame_size)
    if len(raw_frame) != frame_size:
        return False, None

    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3)).copy()
    return True, frame


def make_frame_generator(args: argparse.Namespace) -> Iterable[np.ndarray]:
    configure_opencv_ffmpeg_headers()
    capture = open_capture(args.source, args.force_ffmpeg)
    ffmpeg_process: subprocess.Popen | None = None
    random_frame = np.random.randint(
        0,
        256,
        size=(args.imgsz, args.imgsz, 3),
        dtype=np.uint8,
    )

    if args.source is not None and capture is None and not is_numeric_camera_source(args.source):
        ffmpeg_process = start_ffmpeg_pipe(args)

    try:
        while True:
            if capture is None and ffmpeg_process is None:
                yield random_frame.copy()
                continue

            if capture is not None:
                ok, frame = capture.read()
                if ok:
                    yield frame
                    continue

                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("OpenCV source ended and could not be restarted.")
                yield frame
                continue

            if ffmpeg_process is not None:
                ok, frame = read_ffmpeg_frame(ffmpeg_process, args.ffmpeg_width, args.ffmpeg_height)
                if ok and frame is not None:
                    yield frame
                    continue
                stderr = ""
                if ffmpeg_process.stderr is not None:
                    stderr = ffmpeg_process.stderr.read().decode(errors="replace").strip()
                raise RuntimeError(f"FFmpeg source ended or failed. {stderr}")
    finally:
        if capture is not None:
            capture.release()
        if ffmpeg_process is not None:
            ffmpeg_process.terminate()
            try:
                ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ffmpeg_process.kill()
                ffmpeg_process.wait(timeout=5)


def query_nvidia_smi() -> tuple[float | None, float | None, float | None]:
    if shutil.which("nvidia-smi") is None:
        return None, None, None

    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=2).strip()
    except (subprocess.SubprocessError, OSError):
        return None, None, None

    if not output:
        return None, None, None

    first_gpu = output.splitlines()[0]
    parts = [part.strip() for part in first_gpu.split(",")]
    if len(parts) != 3:
        return None, None, None

    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None, None, None


def query_torch_cuda_memory() -> tuple[float | None, float | None]:
    if torch is None or not torch.cuda.is_available():
        return None, None
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
    reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
    return allocated, reserved


def collect_resource_sample() -> ResourceSample:
    sample = ResourceSample(timestamp=time.perf_counter())

    if psutil is not None:
        sample.cpu_percent = psutil.cpu_percent(interval=None)
        virtual_memory = psutil.virtual_memory()
        sample.ram_used_mb = virtual_memory.used / (1024 * 1024)
        sample.ram_percent = virtual_memory.percent

    gpu_util, vram_used, vram_total = query_nvidia_smi()
    sample.gpu_util_percent = gpu_util
    sample.vram_used_mb = vram_used
    sample.vram_total_mb = vram_total

    cuda_allocated, cuda_reserved = query_torch_cuda_memory()
    sample.torch_cuda_allocated_mb = cuda_allocated
    sample.torch_cuda_reserved_mb = cuda_reserved

    return sample


class ResourceMonitor:
    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.samples: list[ResourceSample] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "ResourceMonitor":
        if psutil is not None:
            psutil.cpu_percent(interval=None)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop_event.set()
        self._thread.join(timeout=max(self.interval_s * 2, 1.0))
        self.samples.append(collect_resource_sample())

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.samples.append(collect_resource_sample())
            self._stop_event.wait(self.interval_s)


def average(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return mean(valid) if valid else None


def maximum(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return max(valid) if valid else None


def summarize_resources(samples: list[ResourceSample]) -> dict[str, float | None]:
    return {
        "avg_cpu_percent": average([sample.cpu_percent for sample in samples]),
        "max_cpu_percent": maximum([sample.cpu_percent for sample in samples]),
        "avg_ram_used_mb": average([sample.ram_used_mb for sample in samples]),
        "max_ram_used_mb": maximum([sample.ram_used_mb for sample in samples]),
        "avg_ram_percent": average([sample.ram_percent for sample in samples]),
        "max_ram_percent": maximum([sample.ram_percent for sample in samples]),
        "avg_gpu_util_percent": average([sample.gpu_util_percent for sample in samples]),
        "max_gpu_util_percent": maximum([sample.gpu_util_percent for sample in samples]),
        "avg_vram_used_mb": average([sample.vram_used_mb for sample in samples]),
        "max_vram_used_mb": maximum([sample.vram_used_mb for sample in samples]),
        "vram_total_mb": maximum([sample.vram_total_mb for sample in samples]),
        "avg_torch_cuda_allocated_mb": average([sample.torch_cuda_allocated_mb for sample in samples]),
        "max_torch_cuda_allocated_mb": maximum([sample.torch_cuda_allocated_mb for sample in samples]),
        "avg_torch_cuda_reserved_mb": average([sample.torch_cuda_reserved_mb for sample in samples]),
        "max_torch_cuda_reserved_mb": maximum([sample.torch_cuda_reserved_mb for sample in samples]),
    }


def write_samples_csv(path: str, samples: list[ResourceSample]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(ResourceSample.__annotations__.keys()))
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.__dict__)


def write_summary_json(path: str, stats: BenchmarkStats) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(stats.__dict__, json_file, indent=2)


def format_value(value: float | None, suffix: str = "", precision: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}{suffix}"


def run_benchmark(args: argparse.Namespace) -> BenchmarkStats:
    engine_path = Path(args.engine)
    if not engine_path.exists():
        raise FileNotFoundError(f"Engine file not found: {engine_path}")

    model = YOLO(str(engine_path), task="detect")
    frames = make_frame_generator(args)

    preprocess_times_ms: list[float] = []
    inference_times_ms: list[float] = []
    postprocess_times_ms: list[float] = []
    model_total_times_ms: list[float] = []
    wall_times_ms: list[float] = []

    total_iterations = args.warmup + args.frames
    print(f"Engine: {engine_path}")
    print(f"Source: {source_label(args.source)}")
    print(f"Warmup frames: {args.warmup}")
    print(f"Measured frames: {args.frames}")
    print("Starting benchmark...")

    with ResourceMonitor(args.sample_interval) as monitor:
        for index in range(total_iterations):
            frame = next(frames)
            start = time.perf_counter()
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                device=args.device,
                conf=args.conf,
                verbose=False,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            if index < args.warmup:
                continue

            wall_times_ms.append(elapsed_ms)
            if results and getattr(results[0], "speed", None):
                speed = results[0].speed
                preprocess = speed.get("preprocess")
                inference = speed.get("inference")
                postprocess = speed.get("postprocess")
                preprocess_times_ms.append(preprocess)
                inference_times_ms.append(inference)
                postprocess_times_ms.append(postprocess)
                valid_parts = [part for part in (preprocess, inference, postprocess) if part is not None]
                if valid_parts:
                    model_total_times_ms.append(sum(valid_parts))

    total_wall_time_s = sum(wall_times_ms) / 1000
    fps = args.frames / total_wall_time_s if total_wall_time_s > 0 else 0.0
    resources = summarize_resources(monitor.samples)

    stats = BenchmarkStats(
        engine=str(engine_path),
        source=source_label(args.source),
        frames=args.frames,
        warmup=args.warmup,
        image_size=args.imgsz,
        device=args.device,
        confidence=args.conf,
        total_wall_time_s=total_wall_time_s,
        fps=fps,
        avg_wall_time_ms=mean(wall_times_ms),
        avg_preprocess_ms=average(preprocess_times_ms),
        avg_inference_ms=average(inference_times_ms),
        avg_postprocess_ms=average(postprocess_times_ms),
        avg_model_total_ms=average(model_total_times_ms),
        resources=resources,
    )

    if args.csv:
        write_samples_csv(args.csv, monitor.samples)
    if args.json:
        write_summary_json(args.json, stats)

    return stats


def print_summary(stats: BenchmarkStats) -> None:
    print("\n=== YOLO TensorRT Engine Benchmark ===")
    print(f"Engine: {stats.engine}")
    print(f"Source: {stats.source}")
    print(f"Frames measured: {stats.frames}")
    print(f"FPS: {format_value(stats.fps, precision=2)}")
    print(f"Average wall time/frame: {format_value(stats.avg_wall_time_ms, ' ms')}")
    print(f"Average preprocessing/frame: {format_value(stats.avg_preprocess_ms, ' ms')}")
    print(f"Average inference/frame: {format_value(stats.avg_inference_ms, ' ms')}")
    print(f"Average postprocessing/frame: {format_value(stats.avg_postprocess_ms, ' ms')}")
    print(f"Average model total/frame: {format_value(stats.avg_model_total_ms, ' ms')}")

    resources = stats.resources
    print("\n=== Resources ===")
    print(f"CPU avg/max: {format_value(resources['avg_cpu_percent'], '%')} / {format_value(resources['max_cpu_percent'], '%')}")
    print(f"RAM used avg/max: {format_value(resources['avg_ram_used_mb'], ' MB')} / {format_value(resources['max_ram_used_mb'], ' MB')}")
    print(f"RAM percent avg/max: {format_value(resources['avg_ram_percent'], '%')} / {format_value(resources['max_ram_percent'], '%')}")
    print(f"GPU util avg/max: {format_value(resources['avg_gpu_util_percent'], '%')} / {format_value(resources['max_gpu_util_percent'], '%')}")
    print(f"VRAM used avg/max: {format_value(resources['avg_vram_used_mb'], ' MB')} / {format_value(resources['max_vram_used_mb'], ' MB')}")
    print(f"VRAM total: {format_value(resources['vram_total_mb'], ' MB')}")
    print(f"Torch CUDA allocated avg/max: {format_value(resources['avg_torch_cuda_allocated_mb'], ' MB')} / {format_value(resources['max_torch_cuda_allocated_mb'], ' MB')}")
    print(f"Torch CUDA reserved avg/max: {format_value(resources['avg_torch_cuda_reserved_mb'], ' MB')} / {format_value(resources['max_torch_cuda_reserved_mb'], ' MB')}")

    if psutil is None:
        print("\nNote: install psutil to enable CPU and RAM metrics: python -m pip install psutil")
    if shutil.which("nvidia-smi") is None:
        print("Note: nvidia-smi was not found; GPU utilization and total VRAM may be unavailable on this machine.")


def main() -> None:
    args = parse_args()
    stats = run_benchmark(args)
    print_summary(stats)


if __name__ == "__main__":
    main()
