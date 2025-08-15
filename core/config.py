import os

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings

load_dotenv()
class Settings(BaseSettings):

    # RTSP URL
    RTSP_URL: str = os.getenv("RTSP_URL","")

    # Webhook URL
    WEBHOOK_URL: str = os.getenv("WEBHOOK_URL","")

    @field_validator("RTSP_URL", "WEBHOOK_URL")
    def not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must not be empty")
        return v
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    if ENVIRONMENT == "development":
        PRETTY_PRINT: bool = True
    else:
        PRETTY_PRINT: bool = False

settings = Settings()