
"""TRI-NAYANTRA backend configuration.

The main API owns MRI, language-model, webhook and data settings.
YOLO/PPE inference is delegated to a separate detection service.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# --------------------------------------------------
# BASE PATHS AND ENVIRONMENT
# --------------------------------------------------

BASE = Path(__file__).resolve().parent.parent

load_dotenv(BASE / ".env")


def _p(value):
    """Resolve relative paths against the backend directory."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE / path


def _env_bool(name, default=False):
    """Read a boolean environment variable."""
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name, default, minimum=None):
    """Read and validate a floating-point environment variable."""
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = float(default)

    if minimum is not None:
        value = max(minimum, value)

    return value


def _env_int(name, default, minimum=None):
    """Read and validate an integer environment variable."""
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = int(default)

    if minimum is not None:
        value = max(minimum, value)

    return value


# --------------------------------------------------
# REMOTE YOLO / PPE DETECTION SERVICE
# --------------------------------------------------

DETECTION_SERVICE_URL = os.getenv(
    "DETECTION_SERVICE_URL", ""
).strip().rstrip("/")

DETECTION_SERVICE_TOKEN = os.getenv(
    "DETECTION_SERVICE_TOKEN", ""
).strip()

DETECTION_CONNECT_TIMEOUT = _env_float(
    "DETECTION_CONNECT_TIMEOUT", 3, minimum=0.5
)

DETECTION_READ_TIMEOUT = _env_float(
    "DETECTION_READ_TIMEOUT", 20, minimum=1
)

# Reference only. The main API must not load YOLO/PPE weights.
YOLO_MODEL = os.getenv(
    "YOLO_MODEL", "yolov8n.pt"
)

PPE_MODEL = _p(
    os.getenv("PPE_MODEL", "models/ppe_best.pt")
)


# --------------------------------------------------
# MRI MODEL
# --------------------------------------------------

MRI_MODEL = _p(
    os.getenv("MRI_MODEL", "models/mri_classifier.pt")
)


# --------------------------------------------------
# SPEECH CONFIGURATION
# --------------------------------------------------

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL", "small"
)


# --------------------------------------------------
# LANGUAGE MODEL CONFIGURATION
# --------------------------------------------------

ANTHROPIC_API_KEY = os.getenv(
    "ANTHROPIC_API_KEY", ""
).strip()

LLM_MODEL = os.getenv(
    "LLM_MODEL", "claude-sonnet-5-5"
)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL", ""
).strip().rstrip("/")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL", "llama3.2"
)


# --------------------------------------------------
# ALERTS AND WEBHOOKS
# --------------------------------------------------

ALERT_WEBHOOK_URL = os.getenv(
    "ALERT_WEBHOOK_URL", ""
).strip()


# --------------------------------------------------
# CORS
# --------------------------------------------------

_cors_value = os.getenv("CORS_ORIGINS", "*")

CORS_ORIGINS = [
    origin.strip()
    for origin in _cors_value.split(",")
    if origin.strip()
]


# --------------------------------------------------
# STORAGE DIRECTORIES
# --------------------------------------------------

DATA = BASE / "data"
DATA.mkdir(parents=True, exist_ok=True)

KNOWLEDGE = BASE / "knowledge"
KNOWLEDGE.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# UPLOAD AND PROCESSING LIMITS
# --------------------------------------------------

MAX_IMAGE_UPLOAD_MB = _env_int(
    "MAX_IMAGE_UPLOAD_MB", 12, minimum=1
)

MAX_IMAGE_UPLOAD_BYTES = (
    MAX_IMAGE_UPLOAD_MB * 1024 * 1024
)

MAX_IMAGE_PIXELS = _env_int(
    "MAX_IMAGE_PIXELS", 16000000, minimum=100000
)


# --------------------------------------------------
# SERVICE SETTINGS
# --------------------------------------------------

DEBUG = _env_bool("DEBUG", False)

APP_NAME = "TRI-NAYANTRA"
APP_VERSION = "1.0.0"
