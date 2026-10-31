
"""Remote YOLO/PPE detection client for TRI-NAYANTRA.

The main API does not import torch or ultralytics.
All model inference is delegated to the detection microservice.
"""

import threading
import requests
import cv2

from . import config

_session = requests.Session()
_request_lock = threading.Lock()

def _headers():
    token = getattr(config, "DETECTION_SERVICE_TOKEN", "")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _endpoint(path):
    base = getattr(config, "DETECTION_SERVICE_URL", "").rstrip("/")
    if not base:
        return None
    return f"{base}{path}"


def _send_frame(path, frame, conf=0.35):
    """Send a BGR frame to the detection service."""
    url = _endpoint(path)
    if not url or frame is None:
        return None

    try:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
        )
        if not ok:
            return None

        files = {
            "file": ("frame.jpg", encoded.tobytes(), "image/jpeg")
        }
        data = {"conf": str(conf)}

        # A single request at a time avoids unnecessary memory pressure.
        with _request_lock:
            response = _session.post(
                url,
                files=files,
                data=data,
                headers=_headers(),
                timeout=(3, 20),
            )

        response.raise_for_status()
        payload = response.json()

        detections = payload.get("detections")
        if detections is None:
            return None

        return detections

    except (requests.RequestException, ValueError, cv2.error):
        return None
    except Exception:
        return None


def objects(bgr, conf=0.35):
    """Return general object detections, or None if unavailable."""
    return _send_frame("/api/detection/objects", bgr, conf)


def ppe_model_loaded():
    """Check whether the remote PPE model is configured and loaded."""
    url = _endpoint("/api/detection/status")
    if not url:
        return False

    try:
        response = _session.get(
            url,
            headers=_headers(),
            timeout=(3, 5),
        )
        response.raise_for_status()
        data = response.json()
        return bool(data.get("ppe_model_available", False))
    except Exception:
        return False


def ppe(bgr, conf=0.35):
    """Return PPE detections, or None if no trained PPE model is available."""
    return _send_frame("/api/detection/ppe", bgr, conf)


PPE_ALIASES = {
    "glove": "gloves",
    "gloves": "gloves",
    "hardhat": "helmet",
    "helmet": "helmet",
    "hard_hat": "helmet",
    "hard-hat": "helmet",
    "goggles": "goggles",
    "glasses": "goggles",
    "mask": "mask",
    "shoes": "shoes",
    "boots": "shoes",
    "safety_shoes": "shoes",
    "safety-shoes": "shoes",
    "vest": "vest",
    "safety_vest": "vest",
    "safety-vest": "vest",
}


def ppe_status(dets, required):
    """Return PPE status: ok, missing, or not_visible."""
    best = {}

    for detection in dets or []:
        label = str(detection.get("label", "")).lower().replace(" ", "_")

        negative = label.startswith(("no_", "no-", "without"))

        normalized = (
            label.replace("no_", "")
            .replace("no-", "")
            .replace("without_", "")
        )

        key = PPE_ALIASES.get(normalized)
        if not key:
            continue

        try:
            confidence = float(detection.get("conf", 0))
        except (TypeError, ValueError):
            confidence = 0

        current = best.get(key)
        if current is None or confidence > current[1]:
            best[key] = (
                "missing" if negative else "ok",
                confidence,
            )

    return {
        item: best[item][0] if item in best else "not_visible"
        for item in required
    }


def phone_in_hand(objs, hands, face_box, thr=0.22):
    """Check whether a detected phone is near a hand or face."""
    phones = [
        item for item in (objs or [])
        if str(item.get("label", "")).lower() == "cell phone"
    ]

    if not phones:
        return False, False

    points = [
        hand["pts"][0]
        for hand in (hands or [])
        if hand.get("pts")
    ]

    if face_box:
        points.append([
            (face_box[0] + face_box[2]) / 2,
            (face_box[1] + face_box[3]) / 2,
        ])

    for phone in phones:
        x1, y1, x2, y2 = phone["box"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        for x, y in points:
            dx = cx - x
            dy = cy - y
            if (dx * dx + dy * dy) ** 0.5 < thr:
                return True, True

    return False, True


def yolo_available():
    """Report whether the remote detection service is reachable."""
    url = _endpoint("/health")
    if not url:
        return False

    try:
        response = _session.get(url, timeout=(3, 5))
        return response.ok
    except requests.RequestException:
        return False


def unload_models():
    """Compatibility function: models are owned by the remote service."""
    return None
