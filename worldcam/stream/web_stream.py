"""HTTP MJPEG streaming helpers for WorldCam headless mode."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import threading
import time

import cv2

from worldcam.analysis.count_persistence import VehicleCountStore


BOUNDARY = "frame"
BOUNDARY_BYTES = BOUNDARY.encode("ascii")
WEB_STREAM_WIDTH = 960
WEB_STREAM_HEIGHT = 540
WEB_STREAM_JPEG_QUALITY = 80
WEB_STREAM_MAX_FPS = 15
WEB_STREAM_MIN_INTERVAL = 1.0 / WEB_STREAM_MAX_FPS
WEB_STREAM_JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), WEB_STREAM_JPEG_QUALITY]
WEB_STREAM_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Connection": "keep-alive",
    "Expires": "0",
    "Pragma": "no-cache",
    "X-Accel-Buffering": "no",
}


class FrameBuffer:
    """Thread-safe latest JPEG frame buffer shared by HTTP streaming clients."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest_frame = None
        self._latest_raw_frame_id = 0
        self._latest_jpeg: bytes | None = None
        self._frame_id = 0
        self._stopped = False
        self._encoder_thread = threading.Thread(target=self._encode_loop, name="WorldCamJpegEncoder", daemon=True)
        self._encoder_thread.start()

    def update(self, frame) -> None:
        """Store the newest annotated frame reference without blocking on JPEG encoding."""
        with self._condition:
            self._latest_frame = frame
            self._latest_raw_frame_id += 1
            self._condition.notify_all()

    def _encode_loop(self) -> None:
        """Encode the latest available frame at the web streaming cadence."""
        next_encode_at = time.perf_counter()
        last_encoded_raw_frame_id = 0

        while True:
            with self._condition:
                while (
                    not self._stopped
                    and (self._latest_frame is None or self._latest_raw_frame_id == last_encoded_raw_frame_id)
                ):
                    self._condition.wait(timeout=0.25)
                if self._stopped:
                    return

            remaining = next_encode_at - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

            with self._condition:
                if self._stopped:
                    return
                if self._latest_frame is None:
                    continue
                frame = self._latest_frame.copy()
                raw_frame_id = self._latest_raw_frame_id

            resized_frame = cv2.resize(frame, (WEB_STREAM_WIDTH, WEB_STREAM_HEIGHT), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", resized_frame, WEB_STREAM_JPEG_PARAMS)
            next_encode_at = max(next_encode_at + WEB_STREAM_MIN_INTERVAL, time.perf_counter())
            if not ok:
                continue

            with self._condition:
                if self._stopped:
                    return
                self._latest_jpeg = encoded.tobytes()
                self._frame_id += 1
                last_encoded_raw_frame_id = raw_frame_id
                self._condition.notify_all()

    def wait_for_frame(self, last_frame_id: int, timeout: float = 2.0):
        """Return the newest encoded JPEG frame and its id, waiting until it changes or times out."""
        deadline = time.perf_counter() + timeout
        with self._condition:
            while self._frame_id == last_frame_id:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            if self._latest_jpeg is None:
                return None, last_frame_id
            return self._latest_jpeg, self._frame_id

    def stop(self) -> None:
        """Stop the background JPEG encoder thread."""
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._encoder_thread.join(timeout=2.0)


def create_app(frame_buffer: FrameBuffer):
    """Create the FastAPI app used by the headless MJPEG stream."""
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - dependency is runtime-only for headless mode
        raise RuntimeError("fastapi is required for WorldCam headless streaming") from exc

    app = FastAPI(title="WorldCam Headless Stream")
    count_store = VehicleCountStore()

    def frame_generator():
        last_frame_id = 0
        while True:
            jpeg_frame, last_frame_id = frame_buffer.wait_for_frame(last_frame_id)
            if jpeg_frame is None:
                continue
            yield (
                b"--" + BOUNDARY_BYTES + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg_frame)).encode("ascii") + b"\r\n\r\n" + jpeg_frame + b"\r\n"
            )

    @app.get("/")
    def index():
        return HTMLResponse(
            """
            <!doctype html>
            <html>
              <head>
                <meta charset="utf-8">
                <title>WorldCam Headless Stream</title>
                <style>
                  body { margin: 0; background: #111; color: #eee; font-family: sans-serif; }
                  header { padding: 0.75rem 1rem; background: #222; }
                  img { display: block; width: 100vw; height: calc(100vh - 3rem); object-fit: contain; }
                </style>
              </head>
              <body>
                <header>WorldCam Headless Stream - 960x540 @ 15 FPS</header>
                <img src="/video_feed" alt="WorldCam stream">
              </body>
            </html>
            """
        )

    @app.get("/json_data")
    def json_data(day: date | None = Query(default=None, alias="date")):
        try:
            snapshot = count_store.read_day_snapshot(day)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return JSONResponse(content=snapshot, headers={"Cache-Control": "no-store"})

    @app.get("/video_feed")
    def video_feed():
        return StreamingResponse(
            frame_generator(),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers=WEB_STREAM_HEADERS,
        )

    return app


@dataclass
class WebStreamServer:
    """Running uvicorn server handle for the headless stream."""

    frame_buffer: FrameBuffer
    server: object
    thread: threading.Thread
    host: str
    port: int

    def update_frame(self, frame) -> None:
        """Publish a new annotated frame to connected HTTP clients."""
        self.frame_buffer.update(frame)

    def stop(self) -> None:
        """Request the HTTP server and JPEG encoder to stop, then wait briefly for their threads."""
        self.frame_buffer.stop()
        setattr(self.server, "should_exit", True)
        self.thread.join(timeout=2.0)


def start_web_stream_server(host: str = "0.0.0.0", port: int = 8080) -> WebStreamServer:
    """Start a background HTTP MJPEG server for annotated frames."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency is runtime-only for headless mode
        raise RuntimeError("uvicorn is required for WorldCam headless streaming") from exc

    frame_buffer = FrameBuffer()
    app = create_app(frame_buffer)
    config = uvicorn.Config(app, host=host, port=port, log_level="info", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="WorldCamWebStream", daemon=True)
    thread.start()
    print(f"Flux headless disponible sur http://{host}:{port}/ ({WEB_STREAM_WIDTH}x{WEB_STREAM_HEIGHT} @ {WEB_STREAM_MAX_FPS} FPS)")
    return WebStreamServer(frame_buffer=frame_buffer, server=server, thread=thread, host=host, port=port)
