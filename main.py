import cv2
import time
import requests
import base64
from ultralytics import YOLO
import torch

from core.config import settings
from core.logger import logger

# --- CONFIG ---
CHANNEL = 3
RTSP_URL= settings.RTSP_URL+f"&channel={CHANNEL}&stream=0.sdp"
COOLDOWN = 15  # seconds
CLASS_NAME = "car"
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
            confidence = float(box.conf)
            logger.info(f"Detected: {cls_name} with confidence {confidence:.2f}")
            if cls_name == CLASS_NAME:
                # Cooldown
                now = time.time()
                if now - last_sent >= COOLDOWN:
                    last_sent = now

                    # Encode image
                    _, buffer = cv2.imencode('.jpg', annotated)
                    b64_img = base64.b64encode(buffer).decode('utf-8')

                    # POST avec image en JSON
                    try:
                        payload = {"image": b64_img}
                        response = requests.post(settings.WEBHOOK_URL, json=payload, timeout=10)
                        response.raise_for_status()
                        logger.info("Webhook envoyé")
                    except requests.exceptions.RequestException as e:
                        logger.error(f"Erreur webhook: {e}")

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
