import os

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings

load_dotenv()
class Settings(BaseSettings):

    # RTSP URL
    RTSP_URL_BASE: str = os.getenv("RTSP_URL_BASE", "")
    CAMERA_CHANNELS: str = os.getenv("CAMERA_CHANNELS", "1")

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
    
    # Performance
    INFERENCE_WIDTH: int = int(os.getenv("INFERENCE_WIDTH", "640"))
    FRAME_SKIP: int = int(os.getenv("FRAME_SKIP", "2"))

    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    if ENVIRONMENT == "development":
        PRETTY_PRINT: bool = True
    else:
        PRETTY_PRINT: bool = False

settings = Settings()