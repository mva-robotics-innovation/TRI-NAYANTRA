
"""TRI-NAYANTRA YOLO/PPE Detection Microservice."""

import io
import os
import hmac
import logging

import cv2
import numpy as np

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Depends,
    HTTPException,
    status,
)
from fastapi.middleware.cors import CORSMiddleware

from .security import verify_detection_token
from . import detector


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tri_nayantra_detection")

APP_NAME = "TRI-NAYANTRA Detection Service"
MAX_UPLOAD_MB = int(os.getenv("MAX_IMAGE_UPLOAD_MB", "8"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "12000000"))

app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    description="Isolated YOLO object and PPE detection service.",
)

# This service is intended for authenticated server-to-server calls.
# Browser CORS is not needed for the normal architecture.
allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]

if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )


@app.get("/")
def root():
    return {
        "project": "TRI-NAYANTRA",
        "service": "detection",
        "status": "ok",
    }


@app.get("/health")
def health():
    """Lightweight liveness endpoint; does not load models."""
    return {
        "status": "ok",
        "service": "tri-nayantra-detection",
    }


@app.get(
    "/api/detection/status",
    dependencies=[Depends(verify_detection_token)],
)
def detection_status():
    """Report whether the configured models actually load."""
    try:
        model_status = detector.get_model_status()
    except Exception:
        logger.exception("Unable to inspect model status")
        model_status = {
            "yolo_model_available": False,
            "ppe_model_available": False,
        }

    return {
        "status": "ok",
        **model_status,
    }


async def read_image(file: UploadFile) -> np.ndarray:
    """Read and validate an uploaded image."""
    if file.content_type not in (
        "image/jpeg",
        "image/png",
        "image/webp",
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload a JPEG, PNG or WebP image.",
        )

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    content = await file.read(max_bytes + 1)

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty.",
        )

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds the {MAX_UPLOAD_MB} MB limit.",
        )

    try:
        image_array = np.frombuffer(content, dtype=np.uint8)
        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )
    except cv2.error:
        image = None

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or unsupported image.",
        )

    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image dimensions.",
        )

    if height * width > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image resolution exceeds the configured limit.",
        )

    return image


def validate_confidence(conf: float):
    if not 0.05 <= conf <= 0.95:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Confidence must be between 0.05 and 0.95.",
        )


@app.post(
    "/api/detection/objects",
    dependencies=[Depends(verify_detection_token)],
)
async def objects(
    file: UploadFile = File(...),
    conf: float = Form(0.35),
):
    """Run general YOLO object detection."""
    validate_confidence(conf)
    image = await read_image(file)

    try:
        detections = detector.detect_objects(image, conf)
        return {"detections": detections}

    except RuntimeError as exc:
        logger.warning("Object detection unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Object detection is currently unavailable.",
        ) from exc

    except Exception as exc:
        logger.exception("Object detection failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Object detection failed.",
        ) from exc

    finally:
        await file.close()


@app.post(
    "/api/detection/ppe",
    dependencies=[Depends(verify_detection_token)],
)
async def ppe(
    file: UploadFile = File(...),
    conf: float = Form(0.35),
):
    """Run the separately trained PPE model."""
    validate_confidence(conf)
    image = await read_image(file)

    try:
        detections = detector.detect_ppe(image, conf)
        return {"detections": detections}

    except RuntimeError as exc:
        logger.warning("PPE detection unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trained PPE detection model is unavailable.",
        ) from exc

    except Exception as exc:
        logger.exception("PPE detection failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PPE detection failed.",
        ) from exc

    finally:
        await file.close()
