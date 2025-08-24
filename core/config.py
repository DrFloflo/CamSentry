import os

from dotenv import load_dotenv
from pydantic import field_validator, ValidationInfo
from pydantic_settings import BaseSettings

load_dotenv()
class Settings(BaseSettings):

    PLATFORM: str = os.getenv("PLATFORM", "linux")

    # RTSP URL
    RTSP_URL_BASE: str = os.getenv("RTSP_URL_BASE", "")
    CAMERA_CHANNELS: str = os.getenv("CAMERA_CHANNELS", "1")
    CAMERA_ROIS: str = os.getenv("CAMERA_ROIS", "")

    # Webhook URL
    WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "")

    @field_validator("RTSP_URL_BASE", "WEBHOOK_URL")
    def not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must not be empty")
        return v

    @field_validator("CAMERA_CHANNELS")
    def parse_camera_channels(cls, v: str) -> list[int]:
        if not v:
            raise ValueError("CAMERA_CHANNELS must not be empty")
        try:
            return [int(channel.strip()) for channel in v.split(',')]
        except ValueError:
            raise ValueError("CAMERA_CHANNELS must be a comma-separated list of integers")
    
    @field_validator("CAMERA_ROIS")
    def parse_camera_rois(cls, v: str, info: ValidationInfo) -> dict[int, tuple[int, int, int, int]]:
        rois = {}
        if not v:
            return rois

        channels = info.data.get('CAMERA_CHANNELS')
        if not channels:
            raise ValueError("CAMERA_CHANNELS must be defined to define ROIs")

        try:
            # Format: "channel1:x1,y1,w1,h1;channel2:x2,y2,w2,h2"
            for roi_entry in v.split(';'):
                if not roi_entry.strip():
                    continue
                channel_str, coords_str = roi_entry.split(':')
                channel = int(channel_str.strip())
                if channel not in channels:
                    raise ValueError(f"ROI defined for channel {channel}, which is not in CAMERA_CHANNELS")

                coords = [int(c.strip()) for c in coords_str.split(',')]
                if len(coords) != 4:
                    raise ValueError("ROI coordinates must be in x,y,w,h format.")
                rois[channel] = tuple(coords)
        except Exception as e:
            raise ValueError(f"Invalid format for CAMERA_ROIS. Expected 'channel:x,y,w,h;...'. Error: {e}")

        return rois
    
    # Performance
    INFERENCE_WIDTH: int = int(os.getenv("INFERENCE_WIDTH", "640"))
    FRAME_SKIP: int = int(os.getenv("FRAME_SKIP", "2"))

    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    if ENVIRONMENT == "development":
        PRETTY_PRINT: bool = True
    else:
        PRETTY_PRINT: bool = False

settings = Settings()