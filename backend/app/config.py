import os
from pathlib import Path
from dotenv import load_dotenv
BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")
def _p(v): p = Path(v); return p if p.is_absolute() else BASE / p
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
PPE_MODEL = _p(os.getenv("PPE_MODEL", "models/ppe_best.pt"))
MRI_MODEL = _p(os.getenv("MRI_MODEL", "models/mri_classifier.pt"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
OLLAMA_URL = os.getenv("OLLAMA_URL", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
DATA = BASE / "data"; DATA.mkdir(exist_ok=True)
KNOWLEDGE = BASE / "knowledge"
