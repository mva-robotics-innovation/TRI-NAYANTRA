"""Download / warm-up every model TRI-NAYANTRA uses.
   python download_models.py            -> YOLOv8 (objects), Whisper, MiniLM embeddings
   python download_models.py --ppe      -> + PPE detector (helmet/gloves/goggles/mask/shoes) from Hugging Face
   python download_models.py --xtts     -> + Coqui XTTS-v2 voice-cloning model (~1.8 GB, non-commercial licence)
MediaPipe (face/pose/hands) models ship inside the pip package. MRI classifier: see train_mri.py."""
import argparse, shutil
from pathlib import Path
from app import config
ap = argparse.ArgumentParser(); ap.add_argument("--ppe", action="store_true"); ap.add_argument("--xtts", action="store_true"); a = ap.parse_args()
(config.BASE / "models").mkdir(exist_ok=True)

print("[1/3] YOLOv8n (COCO: person, cell phone, laptop, tools...)")
from ultralytics import YOLO; YOLO(config.YOLO_MODEL)
print("[2/3] Whisper:", config.WHISPER_MODEL)
from faster_whisper import WhisperModel; WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
print("[3/3] Sentence embeddings for RAG")
from sentence_transformers import SentenceTransformer; SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

if a.ppe:
    from huggingface_hub import hf_hub_download
    # Community PPE model (classes: glove, goggles, helmet, mask, no_* variants, shoes). If this repo is renamed/removed,
    # set PPE_MODEL in .env to any YOLOv8 PPE .pt you trust (e.g. trained on Roboflow 'Construction Site Safety').
    for repo in ("keremberke/yolov8m-protective-equipment-detection", "keremberke/yolov8n-protective-equipment-detection"):
        try:
            p = hf_hub_download(repo, "best.pt"); shutil.copy(p, config.PPE_MODEL); print("PPE model ->", config.PPE_MODEL); break
        except Exception as e: print("  could not fetch", repo, "-", type(e).__name__)
    else: print("!! PPE model download failed - place a YOLOv8 PPE weights file at", config.PPE_MODEL)
if a.xtts:
    from TTS.api import TTS; TTS("tts_models/multilingual/multi-dataset/xtts_v2"); print("XTTS-v2 ready")
print("Done.")
