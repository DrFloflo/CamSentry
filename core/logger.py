import json
import logging
import logging.config
import sys
from typing import Any, Dict

from core.config import settings

class CustomFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%', pretty_print=False):
        super().__init__(fmt, datefmt, style)
        self.pretty_print = pretty_print

    def format(self, record: logging.LogRecord) -> str:
        # Create a copy of the record's extra data
        extra_data: Dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key not in ['args', 'asctime', 'created', 'exc_info', 'exc_text', 
                          'filename', 'funcName', 'id', 'levelname', 'levelno', 
                          'lineno', 'module', 'msecs', 'message', 'msg', 
                          'name', 'pathname', 'process', 'processName', 
                          'relativeCreated', 'stack_info', 'thread', 'threadName']:
                extra_data[key] = value

        # Format the basic log message
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            **extra_data
        }

        # Handle exceptions
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_entry["stack_info"] = self.formatStack(record.stack_info)

        # Pretty print for console, compact for files
        if self.pretty_print:
            return json.dumps(log_entry, indent=2, ensure_ascii=False)
        return json.dumps(log_entry, ensure_ascii=False)

class SuppressGoogleGenaiModelsFilter(logging.Filter):
    def filter(self, record):
        # Return False to suppress logs from google_genai.models
        return record.name != "google_genai.models"

class SuppressHttpxInfoFilter(logging.Filter):
    def filter(self, record):
        # Suppress only INFO-level logs from httpx
        return not (record.name == "httpx" and record.levelno == logging.INFO)
    
class SuppressUSP(logging.Filter):
    def filter(self, record):
        # Suppress logs from usp.
        return not record.name.startswith("usp.")

class StreamHandlerByLevel(logging.Handler):
    def emit(self, record):
        stream = sys.stderr if record.levelno >= logging.ERROR else sys.stdout
        formatter = self.formatter
        try:
            msg = formatter.format(record)
            stream.write(msg + "\n")
            stream.flush()
        except Exception:
            self.handleError(record)


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "console": {"()": CustomFormatter, "pretty_print": settings.PRETTY_PRINT},
    },
    "handlers": {
        "console": {
            "()": StreamHandlerByLevel,
            "formatter": "console",
            "filters": ["suppress_google_genai_models", "suppress_httpx_info", "suppress_USP"]
        },
    },
    "loggers": {
        "": {
            "level": "INFO",
            "handlers": ["console"],
        }
    },
    "filters": {
        "suppress_google_genai_models": {
            "()": SuppressGoogleGenaiModelsFilter
        },
        "suppress_httpx_info": {
            "()": SuppressHttpxInfoFilter
        },
        "suppress_USP": {
            "()": SuppressUSP
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)