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

# --- CONFIG ---
COOLDOWN = 0  # seconds
CLASS_NAMES = ["car", "truck", "person", "cat"]
TARGET_FPS = 5

# --- Init YOLO ---
model = YOLO("yolo11l.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

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
    rtsp_url = settings.RTSP_URL_BASE + f"&channel={channel}&stream=0.sdp"
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logger.error(f"[Channel {channel}] Impossible d'ouvrir le flux RTSP : {rtsp_url}")
        return

    logger.info(f"[Channel {channel}] Stream opened successfully.")
    last_sent = 0
    window_name = f"stream_channel_{channel}"

    try:
        while not stop_event.is_set():
            loop_start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"[Channel {channel}] No frame received, retrying...")
                time.sleep(1) # Wait a bit before retrying
                cap.release()
                cap = cv2.VideoCapture(rtsp_url)
                continue

            # Détection
            results = model(frame, verbose=False, device=device)[0]
            annotated = results.plot()

            # Filtre classe
            for box in results.boxes:
                cls_name = model.names[int(box.cls)]
                confidence = float(box.conf)
                
                if cls_name in CLASS_NAMES:
                    logger.info(f"[Channel {channel}] Detected: {cls_name} with confidence {confidence:.2f}")
                    # Cooldown
                    now = time.time()
                    if now - last_sent >= COOLDOWN:
                        last_sent = now

                        # Encode image
                        _, buffer = cv2.imencode('.jpg', annotated)
                        b64_img = base64.b64encode(buffer).decode('utf-8')

                        # POST avec image en JSON
                        try:
                            payload = {"image": b64_img, "channel": channel}
                            response = requests.post(settings.WEBHOOK_URL, json=payload, timeout=10)
                            response.raise_for_status()
                            logger.info(f"[Channel {channel}] Webhook envoyé")
                        except requests.exceptions.RequestException as e:
                            logger.error(f"[Channel {channel}] Erreur webhook: {e}")

            # Optionnel: affichage local
            if settings.ENVIRONMENT == "development":
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break

            # --- FPS Limiter ---
            elapsed_time = time.time() - loop_start_time
            sleep_time = (1 / TARGET_FPS) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        logger.info(f"[Channel {channel}] Cleaning up resources...")
        cap.release()
        if settings.ENVIRONMENT == "development":
            cv2.destroyWindow(window_name)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

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
