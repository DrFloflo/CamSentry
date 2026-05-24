"""HTTP MJPEG streaming helpers for WorldCam headless mode."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import cv2


BOUNDARY = "frame"


class FrameBuffer:
    """Thread-safe latest-frame buffer encoded by HTTP streaming clients."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest_frame = None
        self._frame_id = 0

    def update(self, frame) -> None:
        """Store a copy of the newest annotated frame and wake streaming clients."""
        with self._condition:
            self._latest_frame = frame.copy()
            self._frame_id += 1
            self._condition.notify_all()

    def wait_for_frame(self, last_frame_id: int, timeout: float = 2.0):
        """Return the newest frame and its id, waiting until it changes or times out."""
        deadline = time.perf_counter() + timeout
        with self._condition:
            while self._frame_id == last_frame_id:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            if self._latest_frame is None:
                return None, last_frame_id
            return self._latest_frame.copy(), self._frame_id


def create_app(frame_buffer: FrameBuffer):
    """Create the FastAPI app used by the headless MJPEG stream."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - dependency is runtime-only for headless mode
        raise RuntimeError("fastapi is required for WorldCam headless streaming") from exc

    app = FastAPI(title="WorldCam Headless Stream")

    def frame_generator():
        last_frame_id = 0
        while True:
            frame, last_frame_id = frame_buffer.wait_for_frame(last_frame_id)
            if frame is None:
                continue
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (
                b"--" + BOUNDARY.encode("ascii") + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
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
                <header>WorldCam Headless Stream</header>
                <img src="/video_feed" alt="WorldCam stream">
              </body>
            </html>
            """
        )

    @app.get("/video_feed")
    def video_feed():
        return StreamingResponse(
            frame_generator(),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
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
        """Request the HTTP server to stop and wait briefly for its thread."""
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
    print(f"Flux headless disponible sur http://{host}:{port}/")
    return WebStreamServer(frame_buffer=frame_buffer, server=server, thread=thread, host=host, port=port)
