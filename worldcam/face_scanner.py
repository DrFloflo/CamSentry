"""Local face scanner and zoom-window helpers for selected person photos."""

from __future__ import annotations

import cv2
import numpy as np

from worldcam.face_yunet import detect_faces_yunet
from worldcam.image_upscale import upscale_for_display

FACE_ZOOM_WINDOW_NAME = "Zoom visage selectionne"
FACE_ZOOM_MIN_WINDOW_WIDTH = 220
FACE_ZOOM_MIN_WINDOW_HEIGHT = 220
FACE_SCAN_COLOR = (0, 255, 255)
FACE_SCAN_TEXT_COLOR = (0, 0, 0)
FACE_SCAN_PADDING_RATIO = 0.35
_FACE_CASCADE: cv2.CascadeClassifier | None = None


def load_face_cascade() -> cv2.CascadeClassifier | None:
    """Load OpenCV's local Haar face detector once."""
    global _FACE_CASCADE
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print(f"Scanner visage indisponible: cascade introuvable ({cascade_path}).")
        return None

    _FACE_CASCADE = cascade
    return _FACE_CASCADE


def detect_faces_haar(photo: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Fallback face detection using OpenCV Haar cascade."""
    cascade = load_face_cascade()
    if cascade is None:
        return []

    gray_photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    gray_photo = cv2.equalizeHist(gray_photo)
    faces = cascade.detectMultiScale(
        gray_photo,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(24, 24),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def detect_faces(photo: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect faces with specialized YuNet first, then Haar as fallback."""
    faces = detect_faces_yunet(photo)
    if faces:
        return faces
    return detect_faces_haar(photo)


def choose_best_face(faces: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    """Select the largest detected face for the zoom window."""
    if not faces:
        return None
    return max(faces, key=lambda face: face[2] * face[3])


def draw_face_scanner_overlay(photo: np.ndarray, faces: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Draw scanner boxes and status text on a copy of the selected person photo."""
    scanned_photo = photo.copy()
    status = "VISAGE DETECTE" if faces else "AUCUN VISAGE"
    status_width = min(scanned_photo.shape[1] - 1, 190)
    cv2.rectangle(scanned_photo, (0, 0), (status_width, 30), FACE_SCAN_COLOR, -1)
    cv2.putText(scanned_photo, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, FACE_SCAN_TEXT_COLOR, 1)

    for face_index, (x, y, w, h) in enumerate(faces, start=1):
        cv2.rectangle(scanned_photo, (x, y), (x + w, y + h), FACE_SCAN_COLOR, 2)
        cv2.putText(
            scanned_photo,
            f"face #{face_index}",
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            FACE_SCAN_COLOR,
            1,
        )
    return scanned_photo


def build_face_zoom(photo: np.ndarray, face: tuple[int, int, int, int]) -> np.ndarray:
    """Crop and enlarge the selected face for a dedicated zoom window."""
    photo_height, photo_width = photo.shape[:2]
    x, y, w, h = face
    padding_x = int(w * FACE_SCAN_PADDING_RATIO)
    padding_y = int(h * FACE_SCAN_PADDING_RATIO)
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(photo_width, x + w + padding_x)
    y2 = min(photo_height, y + h + padding_y)
    face_crop = photo[y1:y2, x1:x2].copy()
    if face_crop.size == 0:
        return photo

    return upscale_for_display(
        face_crop,
        FACE_ZOOM_MIN_WINDOW_WIDTH,
        FACE_ZOOM_MIN_WINDOW_HEIGHT,
        max_scale=4.0,
        interpolation=cv2.INTER_LANCZOS4,
    )


def open_face_zoom_window(face_zoom: np.ndarray) -> None:
    """Open or refresh a separate window with the detected face zoom."""
    zoom_height, zoom_width = face_zoom.shape[:2]
    window_width = max(FACE_ZOOM_MIN_WINDOW_WIDTH, zoom_width)
    window_height = max(FACE_ZOOM_MIN_WINDOW_HEIGHT, zoom_height)
    cv2.namedWindow(FACE_ZOOM_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(FACE_ZOOM_WINDOW_NAME, window_width, window_height)
    cv2.moveWindow(FACE_ZOOM_WINDOW_NAME, 380, 80)
    cv2.imshow(FACE_ZOOM_WINDOW_NAME, face_zoom)


def close_face_zoom_window() -> None:
    """Close the face zoom window if it exists."""
    try:
        cv2.destroyWindow(FACE_ZOOM_WINDOW_NAME)
    except cv2.error:
        pass


def scan_and_open_face_zoom(photo: np.ndarray) -> np.ndarray:
    """Scan the person photo, annotate it, and open a face zoom window when a face is detected."""
    faces = detect_faces(photo)
    scanned_photo = draw_face_scanner_overlay(photo, faces)
    best_face = choose_best_face(faces)
    if best_face is None:
        close_face_zoom_window()
        return scanned_photo

    open_face_zoom_window(build_face_zoom(photo, best_face))
    return scanned_photo
