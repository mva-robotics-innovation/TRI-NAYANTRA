
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


# --------------------------------------------------
# REMOTE YOLO / PPE DETECTION SERVICE
# --------------------------------------------------

# Example:
# https://tri-nayantra-detection.onrender.com
DETECTION_SERVICE_URL = os.getenv(
    "DETECTION_SERVICE_URL", ""
).strip().rstrip("/")

# Shared secret between the main API and detection service.
# Keep this secret on the backend only.
DETECTION_SERVICE_TOKEN = os.getenv(
    "DETECTION_SERVICE_TOKEN", ""
).strip()

# Connection and inference request timeouts.
DETECTION_CONNECT_TIMEOUT = float(
    os.getenv("DETECTION_CONNECT_TIMEOUT", "3")
)

DETECTION_READ_TIMEOUT = float(
    os.getenv("DETECTION_READ_TIMEOUT", "20")
)

# The following paths are retained for compatibility/reference.
# The main backend must NOT load these models.
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")

PPE_MODEL = _p(
    os.getenv("PPE_MODEL", "models/ppe_best.pt")
)


# --------------------------------------------------
# MRI MODEL
# --------------------------------------------------

# MRI processing remains in the main backend.
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

MAX_IMAGE_UPLOAD_MB = int(
    os.getenv("MAX_IMAGE_UPLOAD_MB", "12")
)

MAX_IMAGE_UPLOAD_BYTES = (
    MAX_IMAGE_UPLOAD_MB * 1024 * 1024
)

MAX_IMAGE_PIXELS = int(
    os.getenv("MAX_IMAGE_PIXELS", "16000000")
)


# --------------------------------------------------
# SERVICE SETTINGS
# --------------------------------------------------

DEBUG = _env_bool("DEBUG", False)

APP_NAME = "TRI-NAYANTRA"
APP_VERSION = "1.0.0"
