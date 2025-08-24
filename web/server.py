from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import threading
import time
from core.logger import logger

app = FastAPI()

# Shared frame variable
latest_frame = None
frame_lock = threading.Lock()
frame_condition = threading.Condition(frame_lock)

def update_frame(frame):
    """Update the latest frame for streaming."""
    global latest_frame
    with frame_lock:
        latest_frame = frame.copy()
        frame_condition.notify_all() # Notify waiting generator
    logger.debug("Frame updated for web server.")

def frame_generator():
    """Generator that yields JPEG frames."""
    global latest_frame
    while True:
        with frame_lock:
            # Wait until a new frame is available
            frame_condition.wait()
            if latest_frame is None:
                continue

            ret, buffer = cv2.imencode('.jpg', latest_frame)
            if not ret:
                logger.warning("Could not encode frame to JPEG.")
                continue
            frame_data = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
        )

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
def index():
    return HTMLResponse(content="""
    <html>
    <head>
        <title>Camera Stream</title>
    </head>
    <body>
        <h1>Live Camera Stream</h1>
        <img src='/video_feed' style='max-width: 100%;'/>
    </body>
    </html>
    """)