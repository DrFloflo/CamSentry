import cv2
import time
import requests
import base64
from ultralytics import YOLO
import torch
import threading
import signal

from core.config import settings
from core.logger import logger
from web import server as web_server
import uvicorn

# --- CONFIG ---
COOLDOWN = 2  # seconds
CLASS_NAMES = ["person", "cat"]
TARGET_FPS = 4

# --- Init YOLO ---
# For Jetson Nano, using a smaller model and a TensorRT engine is crucial for performance.
# 1. Export your model to TensorRT: yolo export model=yolov8n.pt format=engine device=0
# 2. This will create a 'yolov8n.engine' file.
MODEL_PT = "yolo11n.pt"  # A smaller model is better for Jetson
MODEL_ENGINE = "yolo11n.engine"

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

try:
    logger.info(f"Attempting to load TensorRT model: {MODEL_ENGINE}")
    model = YOLO(MODEL_ENGINE)
except Exception as e:
    logger.warning(f"Could not load TensorRT model ({e}), falling back to PyTorch model: {MODEL_PT}")
    model = YOLO(MODEL_PT)
    if device == "cuda":
        model.half() # FP16 is a good optimization for PyTorch models on GPU

# --- Graceful Shutdown ---
stop_event = threading.Event()

def signal_handler(signum, frame):
    logger.info("Shutdown signal received. Stopping threads...")
    stop_event.set()

def process_camera_stream(channel: int):
    """
    Processes a single camera stream.
    Captures video, performs object detection, and sends webhooks.
    """
    # --- GStreamer Pipeline for Hardware-Accelerated Video Decoding ---
    # Using a GStreamer pipeline with nvv4l2decoder leverages the Jetson's hardware decoder,
    # significantly reducing CPU load compared to the default OpenCV backend.
    rtsp_url = settings.RTSP_URL_BASE + f"&channel={channel}&stream=0.sdp"
    if settings.PLATFORM == "windows":
         cap = cv2.VideoCapture(rtsp_url)
    else:
        gstreamer_pipeline = (
            f"rtspsrc location={rtsp_url} latency=0 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw, format=(string)BGRx ! "
            "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1"
        )
        cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        logger.error(f"[Channel {channel}] Impossible d'ouvrir le flux RTSP avec GStreamer : {rtsp_url}")
        return

    logger.info(f"[Channel {channel}] Stream opened successfully.")
    last_sent = 0
    window_name = f"stream_channel_{channel}"
    frame_count = 0
    try:
        while not stop_event.is_set():
            loop_start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"[Channel {channel}] No frame received, retrying...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(rtsp_url)
                continue

            frame_count += 1
            
            # --- Prepare Frame for Processing ---
            processing_frame = frame
            roi_offset = (0, 0)
            roi = settings.CAMERA_ROIS.get(channel)
            if roi:
                x, y, w, h = roi
                processing_frame = frame[y:y+h, x:x+w]
                roi_offset = (x, y)

            # --- Visualization: Draw ROI on original frame ---
            if settings.ENVIRONMENT == "development" and roi:
                x, y, w, h = roi
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            # --- Process every Nth frame ---
            if frame_count % settings.FRAME_SKIP == 0:
                # Resize
                proc_h, proc_w, _ = processing_frame.shape
                new_width = settings.INFERENCE_WIDTH
                new_height = int(proc_h * (new_width / proc_w))
                resized_frame = cv2.resize(processing_frame, (new_width, new_height))

                # Détection
                results = model(resized_frame, verbose=False, device=device)[0]

                # --- Draw detections on original frame for display ---
                if settings.ENVIRONMENT == "development":
                    scale_x = proc_w / new_width
                    scale_y = proc_h / new_height
                    for box in results.boxes:
                        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                        # Scale coords back to processing_frame size
                        orig_x1 = int(x1 * scale_x)
                        orig_y1 = int(y1 * scale_y)
                        orig_x2 = int(x2 * scale_x)
                        orig_y2 = int(y2 * scale_y)
                        # Offset coords back to original frame size
                        final_x1 = orig_x1 + roi_offset[0]
                        final_y1 = orig_y1 + roi_offset[1]
                        final_x2 = orig_x2 + roi_offset[0]
                        final_y2 = orig_y2 + roi_offset[1]
                        
                        cls_name = model.names[int(box.cls)]
                        confidence = float(box.conf)
                        label = f"{cls_name} {confidence:.2f}"
                        
                        cv2.rectangle(frame, (final_x1, final_y1), (final_x2, final_y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (final_x1, final_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                # --- Webhook Logic ---
                for box in results.boxes:
                    cls_name = model.names[int(box.cls)]
                    if cls_name in CLASS_NAMES:
                        logger.info(f"[Channel {channel}] Detected: {cls_name} with confidence {float(box.conf):.2f}")
                        now = time.time()
                        if now - last_sent >= COOLDOWN:
                            last_sent = now
                            # Encode image for webhook (using original frame with drawings)
                            _, buffer = cv2.imencode('.jpg', frame)
                            b64_img = base64.b64encode(buffer).decode('utf-8')
                            try:
                                payload = {"image": b64_img, "channel": channel}
                                response = requests.post(settings.WEBHOOK_URL, json=payload, timeout=10)
                                response.raise_for_status()
                                logger.info(f"[Channel {channel}] Webhook sent")
                            except requests.exceptions.RequestException as e:
                                logger.error(f"[Channel {channel}] Webhook error: {e}")
                            break # Send webhook only once per detection cycle


            # --- FPS Limiter ---
            elapsed_time = time.time() - loop_start_time
            sleep_time = (1 / TARGET_FPS) - elapsed_time

            # Update web server frame
            web_server.update_frame(frame)

            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        logger.info(f"[Channel {channel}] Cleaning up resources...")
        cap.release()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start FastAPI server in a separate thread
    def run_web():
        uvicorn.run(web_server.app, host="0.0.0.0", port=8000)

    web_thread = threading.Thread(target=run_web, daemon=True)
    web_thread.start()
    logger.info("Web server started at http://0.0.0.0:8000")

    threads = []
    for channel in settings.CAMERA_CHANNELS:
        thread = threading.Thread(target=process_camera_stream, args=(channel,), daemon=True)
        threads.append(thread)
        thread.start()
        logger.info(f"Started processing for camera channel {channel}")

    try:
        # Keep the main thread alive while the worker threads are running
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down.")
        stop_event.set()

    finally:
        logger.info("Waiting for all threads to complete...")
        for thread in threads:
            thread.join()
        
        if settings.ENVIRONMENT == "development":
            cv2.destroyAllWindows()

        if device == "cuda":
            del model
            torch.cuda.empty_cache()
            logger.info("Model cleared from VRAM.")
        
        logger.info("Application shut down gracefully.")
