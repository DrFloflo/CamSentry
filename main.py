import cv2
import time
import requests
import base64
from ultralytics import YOLO
import torch

from core.config import settings
from core.logger import logger

# --- CONFIG ---
CHANNEL = 5
RTSP_URL= settings.RTSP_URL+f"&channel={CHANNEL}&stream=0.sdp"
COOLDOWN = 5  # seconds
CLASS_NAME = "cat"
TARGET_FPS = 5

# --- Init YOLO ---
model = YOLO("yolo11l.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
last_sent = 0

# --- Open RTSP via GStreamer ---
cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    logger.error(f"Impossible d'ouvrir le flux RTSP : {RTSP_URL}")
    exit(1)

try:
    while True:
        loop_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            continue

        # Détection
        results = model(frame, verbose=False, device=device)[0]
        annotated = results.plot()

        # Filtre classe cat
        for box in results.boxes:
            cls_name = model.names[int(box.cls)]
            logger.info(f"Detected: {cls_name}")
            if cls_name == CLASS_NAME:
                # Cooldown
                now = time.time()
                if now - last_sent >= COOLDOWN:
                    last_sent = now

                    # Encode image
                    _, buffer = cv2.imencode('.jpg', annotated)
                    b64_img = base64.b64encode(buffer).decode('utf-8')

                    # GET avec image en param
                    try:
                        requests.get(settings.WEBHOOK_URL, params={"image": b64_img}, timeout=5)
                    except Exception as e:
                        print(f"Erreur webhook: {e}")

        # Optionnel: affichage local
        if settings.ENVIRONMENT == "development":
            cv2.imshow("stream", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # --- FPS Limiter ---
        elapsed_time = time.time() - loop_start_time
        sleep_time = (1 / TARGET_FPS) - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)
finally:
    logger.info("Cleaning up resources...")
    cap.release()
    cv2.destroyAllWindows()
    if device == "cuda":
        del model
        torch.cuda.empty_cache()
        logger.info("Model cleared from VRAM.")
