"""Shared configuration for the WorldCam analysis pipeline."""

STREAM_URLS = [
    (
        "https://videos-3.earthcam.com/fecnetwork/22172.flv/playlist.m3u8?"
        "t=pOGmzuDAmYSX7knfXlDLzYSXAgLsxTkQOagDzUD/U1NsjqCkLWUOMbS9PlyNFjKC&td=202605201348"
    ),
    (
        "https://videos-3.earthcam.com/fecnetwork/24322.flv/playlist.m3u8?"
        "t=qAP3aum0UbcBtTuO%2Fx%2F7Lz9UytxcCWnrPDJyjgaIxep8QE4xtRu4RMqXEWHwdbnk&td=202605160341"
    ),
    (
        "https://videos-3.earthcam.com/fecnetwork/24935.flv/playlist.m3u8?"
        "t=mexLmPaUse3br05vrUOgWmHCY3zSSIm2XkB8hKYeViSUXzq/1zkE4DlQHFYGtVec&td=202605161011"
    ),
    (
        "https://videos-3.earthcam.com/fecnetwork/14320.flv/playlist.m3u8?"
        "t=tBywzTYURGbEtGxF1LfyCkm5AvmWDkmIcwUf7arKv5SWW6J7OwYDRnqeU6TNfEfx&td=202605161018"
    ),
]

OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
TARGET_FPS = 24
FRAME_INTERVAL = 1.0 / TARGET_FPS
FRAME_SIZE = OUTPUT_WIDTH * OUTPUT_HEIGHT * 3
READ_WARN_SECONDS = 0.25
STATS_LOG_SECONDS = 5.0
STREAM_READ_TIMEOUT_SECONDS = 1.0
STREAM_STALE_SECONDS = 2.5
MAX_STREAM_READ_FAILURES = 5

FFMPEG_INPUT_REALTIME = True
FFMPEG_THREAD_QUEUE_SIZE = 512
FFMPEG_PROBESIZE = "512k"
FFMPEG_ANALYZEDURATION_US = 1_000_000
FFMPEG_MAX_DELAY_US = 500_000
FFMPEG_RECONNECT_DELAY_MAX_SECONDS = 2
FFMPEG_HWACCEL = ""
FFMPEG_REPEAT_DELTA_THRESHOLD = 1.0
FFMPEG_REPEAT_SAMPLE_SIZE = (64, 36)

MODEL_PT = "models/yolo26m.pt"
MODEL_ONNX = "models/yolo26m.onnx"
MODEL_ENGINE = "models/yolo26m.engine"
POSE_MODEL_PT = "models/yolo26m-pose.pt"
POSE_MODEL_ONNX = "models/yolo26m-pose.onnx"
POSE_MODEL_ENGINE = "models/yolo26m-pose.engine"
SEGMENTATION_MODEL_PT = "models/yolo26m-seg.pt"
SEGMENTATION_MODEL_ONNX = "models/yolo26m-seg.onnx"
SEGMENTATION_MODEL_ENGINE = "models/yolo26m-seg.engine"
DEFAULT_CLASS_NAMES = {"person", "car", "bicycle", "motorcycle", "bus", "truck", "airplane"}
INFERENCE_WIDTH = 640
FRAME_SKIP = 2

SAHI_ENABLED = True
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
    "airplane": (180, 105, 255),
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

SEGMENTATION_ENABLED = False
SEGMENTATION_ALPHA = 0.25
SEGMENTATION_CONTOUR_THICKNESS = 1
SEGMENTATION_Y_OFFSET = 7

COUNTING_ZONE_ENABLED = True
COUNTING_ZONE_EDIT_ENABLED = False
COUNTING_ZONE_POINTS = [(227, 494), (387, 527), (10, 637), (1, 558)]
COUNTING_ZONE_COLOR = (0, 255, 255)
COUNTING_ZONE_EDIT_COLOR = (0, 128, 255)
COUNTING_ZONE_HANDLE_COLOR = (0, 0, 255)
COUNTING_ZONE_HANDLE_RADIUS = 8
COUNTING_ZONE_MIN_SIZE = 10

PERSON_TRACK_ENABLED = False
PERSON_TRACK_COLOR = (0, 255, 255)
PERSON_TRACK_MIN_IOU = 0.20
PERSON_TRACK_MAX_DISTANCE = 80.0
PERSON_TRACK_MAX_AGE = 3
PERSON_TRACK_TRAIL_LENGTH = 6
PERSON_TRACK_DEBUG = False
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
