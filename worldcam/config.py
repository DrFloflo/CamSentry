"""Shared configuration for the WorldCam analysis pipeline."""

STREAM_URL = (
    "https://videos-3.earthcam.com/fecnetwork/24322.flv/playlist.m3u8?"
    "t=qAP3aum0UbcBtTuO%2Fx%2F7Lz9UytxcCWnrPDJyjgaIxep8QE4xtRu4RMqXEWHwdbnk&td=202605160341"
)

OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
TARGET_FPS = 20
FRAME_INTERVAL = 1.0 / TARGET_FPS
FRAME_SIZE = OUTPUT_WIDTH * OUTPUT_HEIGHT * 3
READ_WARN_SECONDS = 0.25
STATS_LOG_SECONDS = 5.0

MODEL_PT = "yolo26l.pt"
MODEL_ONNX = "yolo26l.onnx"
MODEL_ENGINE = "yolo26l.engine"
POSE_MODEL_PT = "yolo26m-pose.pt"
POSE_MODEL_ONNX = "yolo26m-pose.onnx"
POSE_MODEL_ENGINE = "yolo26m-pose.engine"
DEFAULT_CLASS_NAMES = {"person", "car", "bicycle", "motorcycle", "bus", "truck"}
INFERENCE_WIDTH = 640
FRAME_SKIP = 4

SAHI_ENABLED = True
SAHI_MODEL_PATH = MODEL_PT
SAHI_CONFIDENCE_THRESHOLD = 0.25
SAHI_SLICE_HEIGHT = 512
SAHI_SLICE_WIDTH = 512
SAHI_OVERLAP_HEIGHT_RATIO = 0.2
SAHI_OVERLAP_WIDTH_RATIO = 0.2

DETECTION_COLOR = (0, 255, 0)
DETECTION_CLASS_COLORS = {
    "person": (0, 255, 0),
    "bicycle": (255, 180, 0),
    "car": (0, 165, 255),
    "motorcycle": (255, 0, 0),
    "bus": (0, 255, 255),
    "truck": (255, 0, 255),
    "cat": (180, 105, 255),
}
DETECTION_FALLBACK_COLORS = [
    (0, 255, 0),
    (255, 180, 0),
    (0, 165, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (180, 105, 255),
    (255, 255, 0),
]
POSE_KEYPOINT_COLOR = (255, 0, 255)
POSE_SKELETON_COLOR = (255, 255, 0)
POSE_CONFIDENCE_THRESHOLD = 0.30
POSE_SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

MENU_BACKGROUND_COLOR = (35, 35, 35)
MENU_SELECTED_COLOR = (0, 255, 255)
MENU_TEXT_COLOR = (255, 255, 255)
MENU_ENABLED_COLOR = (0, 220, 0)
MENU_PAGE_SIZE = 12

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
REFERER = "https://www.earthcam.com/world/ireland/dublin/"
ORIGIN = "https://www.earthcam.com"

FFMPEG_HEADERS = "\r\n".join(
    [
        f"User-Agent: {USER_AGENT}",
        f"Referer: {REFERER}",
        f"Origin: {ORIGIN}",
        "Accept: */*",
        "",
    ]
)
