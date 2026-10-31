
"""YOLO and PPE inference engine for TRI-NAYANTRA.

All heavy model dependencies stay inside this microservice.
Models are loaded lazily and inference runs on CPU.
"""

import os
import threading
from pathlib import Path

import cv2
import numpy as np


SERVICE_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(
    os.getenv("MODELS_DIR", str(SERVICE_ROOT / "models"))
)

YOLO_MODEL_PATH = Path(
    os.getenv("YOLO_MODEL_PATH", str(MODELS_DIR / "yolov8n.pt"))
)

PPE_MODEL_PATH = Path(
    os.getenv("PPE_MODEL_PATH", str(MODELS_DIR / "ppe_best.pt"))
)

DEVICE = os.getenv("DEVICE", "cpu")
IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "480"))
MAX_CONCURRENT_INFERENCES = int(
    os.getenv("MAX_CONCURRENT_INFERENCES", "1")
)

_model_lock = threading.Lock()
_inference_slots = threading.BoundedSemaphore(
    max(1, MAX_CONCURRENT_INFERENCES)
)

_yolo_model = None
_ppe_model = None
_yolo_attempted = False
_ppe_attempted = False
_yolo_error = None
_ppe_error = None


def _load_model(model_path: Path):
    """Load an Ultralytics YOLO checkpoint."""
    from ultralytics import YOLO

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path.name}"
        )

    return YOLO(str(model_path))


def get_yolo_model():
    """Load the general object detection model once."""
    global _yolo_model, _yolo_attempted, _yolo_error

    with _model_lock:
        if _yolo_model is not None:
            return _yolo_model

        if _yolo_attempted:
            return None

        _yolo_attempted = True

        try:
            _yolo_model = _load_model(YOLO_MODEL_PATH)
            _yolo_error = None
        except Exception as exc:
            _yolo_error = str(exc)
            _yolo_model = None

        return _yolo_model


def get_ppe_model():
    """Load a separately trained PPE checkpoint once."""
    global _ppe_model, _ppe_attempted, _ppe_error

    with _model_lock:
        if _ppe_model is not None:
            return _ppe_model

        if _ppe_attempted:
            return None

        _ppe_attempted = True

        try:
            _ppe_model = _load_model(PPE_MODEL_PATH)
            _ppe_error = None
        except Exception as exc:
            _ppe_error = str(exc)
            _ppe_model = None

        return _ppe_model


def _normalize_detections(results):
    """Convert YOLO results into normalized bounding boxes."""
    detections = []

    for result in results:
        if result.boxes is None:
            continue

        height, width = result.orig_shape

        if not width or not height:
            continue

        names = result.names or {}

        for box in result.boxes:
            try:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                x1, y1, x2, y2 = (
                    float(value)
                    for value in box.xyxy[0].tolist()
                )

                label = str(names.get(class_id, class_id))

                detections.append({
                    "label": label,
                    "conf": round(confidence, 4),
                    "box": [
                        round(max(0.0, min(1.0, x1 / width)), 6),
                        round(max(0.0, min(1.0, y1 / height)), 6),
                        round(max(0.0, min(1.0, x2 / width)), 6),
                        round(max(0.0, min(1.0, y2 / height)), 6),
                    ],
                })

            except (ValueError, TypeError, IndexError, AttributeError):
                continue

    return detections


def _run_inference(model, image: np.ndarray, confidence: float):
    """Run CPU inference with bounded concurrency."""
    acquired = _inference_slots.acquire(timeout=10)

    if not acquired:
        raise RuntimeError("Detection service is busy. Try again shortly.")

    try:
        results = model.predict(
            source=image,
            conf=confidence,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )

        return _normalize_detections(results)

    finally:
        _inference_slots.release()


def detect_objects(image: np.ndarray, confidence: float = 0.35):
    """Run general object detection."""
    model = get_yolo_model()

    if model is None:
        raise RuntimeError("General YOLO model is unavailable.")

    return _run_inference(model, image, confidence)


def detect_ppe(image: np.ndarray, confidence: float = 0.35):
    """Run PPE detection using the separately trained PPE checkpoint."""
    model = get_ppe_model()

    if model is None:
        raise RuntimeError(
            "Trained PPE model is unavailable. Configure PPE_MODEL_PATH."
        )

    return _run_inference(model, image, confidence)


def get_model_status():
    """Return truthful model availability and loading information."""
    yolo = get_yolo_model()
    ppe_model = get_ppe_model()

    return {
        "yolo_model_available": yolo is not None,
        "ppe_model_available": ppe_model is not None,
        "device": DEVICE,
        "image_size": IMAGE_SIZE,
    }
