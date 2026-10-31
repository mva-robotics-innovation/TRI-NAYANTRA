
"""Remote YOLO/PPE detection client for TRI-NAYANTRA.

The main API does not import torch or ultralytics.
All model inference is delegated to the detection microservice.
"""

import threading
import math

import cv2
import requests

from . import config


_session = requests.Session()
_request_lock = threading.Lock()

PPE_ALIASES = {
    "glove": "gloves",
    "gloves": "gloves",
    "helmet": "helmet",
    "hardhat": "helmet",
    "hard_hat": "helmet",
    "goggles": "goggles",
    "glasses": "goggles",
    "mask": "mask",
    "shoe": "shoes",
    "shoes": "shoes",
    "boot": "shoes",
    "boots": "shoes",
    "safety_shoes": "shoes",
    "vest": "vest",
    "safety_vest": "vest",
}

PHONE_LABELS = {
    "cell phone",
    "cell_phone",
    "mobile phone",
    "mobile_phone",
    "phone",
}


def _base_url():
    return str(
        getattr(config, "DETECTION_SERVICE_URL", "") or ""
    ).strip().rstrip("/")


def _endpoint(path):
    base = _base_url()
    return f"{base}{path}" if base else None


def _headers():
    token = str(
        getattr(config, "DETECTION_SERVICE_TOKEN", "") or ""
    ).strip()

    if not token:
        return {}

    return {"Authorization": f"Bearer {token}"}


def _timeout():
    connect = float(
        getattr(config, "DETECTION_CONNECT_TIMEOUT", 3)
    )
    read = float(
        getattr(config, "DETECTION_READ_TIMEOUT", 20)
    )
    return max(0.5, connect), max(1, read)


def _normalize_detection(item):
    """Normalize a detection to label, conf and box format."""
    if not isinstance(item, dict):
        return None

    label = str(item.get("label", "")).strip()
    if not label:
        return None

    confidence = item.get("conf", item.get("confidence", 0))

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    if not math.isfinite(confidence):
        confidence = 0.0

    box = item.get("box")

    if box is None and all(
        key in item for key in ("x1", "y1", "x2", "y2")
    ):
        box = [
            item["x1"],
            item["y1"],
            item["x2"],
            item["y2"],
        ]

    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None

    try:
        box = [float(value) for value in box]
    except (TypeError, ValueError):
        return None

    if not all(math.isfinite(value) for value in box):
        return None

    x1, y1, x2, y2 = box

    # Detection service returns normalized coordinates.
    box = [
        max(0.0, min(1.0, x1)),
        max(0.0, min(1.0, y1)),
        max(0.0, min(1.0, x2)),
        max(0.0, min(1.0, y2)),
    ]

    return {
        "label": label,
        "conf": confidence,
        "box": box,
    }


def _send_frame(path, frame, conf=0.35):
    """Send a BGR frame to the detection service."""
    url = _endpoint(path)

    if not url or frame is None:
        return None

    if not hasattr(frame, "shape") or len(frame.shape) < 2:
        return None

    try:
        if frame.size == 0:
            return None

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
        )

        if not ok:
            return None

        files = {
            "file": (
                "frame.jpg",
                encoded.tobytes(),
                "image/jpeg",
            )
        }

        data = {"conf": str(float(conf))}

        # Serialize requests through the shared Session.
        with _request_lock:
            response = _session.post(
                url,
                files=files,
                data=data,
                headers=_headers(),
                timeout=_timeout(),
            )

        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, dict):
            return None

        detections = payload.get("detections")

        if not isinstance(detections, list):
            return None

        normalized = []

        for item in detections:
            detection = _normalize_detection(item)
            if detection is not None:
                normalized.append(detection)

        return normalized

    except (
        requests.RequestException,
        ValueError,
        TypeError,
        cv2.error,
    ):
        return None
    except Exception:
        return None


def objects(bgr, conf=0.35):
    """Return general object detections, or None if unavailable."""
    return _send_frame(
        "/api/detection/objects",
        bgr,
        conf,
    )


def ppe(bgr, conf=0.35):
    """Return PPE detections, or None if unavailable."""
    return _send_frame(
        "/api/detection/ppe",
        bgr,
        conf,
    )


def _get_json(path, timeout=None):
    url = _endpoint(path)

    if not url:
        return None

    try:
        with _request_lock:
            response = _session.get(
                url,
                headers=_headers(),
                timeout=timeout or _timeout(),
            )

        response.raise_for_status()

        data = response.json()
        return data if isinstance(data, dict) else None

    except Exception:
        return None


def ppe_model_loaded():
    """Check whether a PPE model file is configured on the service."""
    data = _get_json(
        "/api/detection/status",
        timeout=(3, 5),
    )

    if not data:
        return False

    return bool(data.get("ppe_model_available", False))


def yolo_available():
    """Check whether the remote detection service is reachable."""
    data = _get_json(
        "/health",
        timeout=(3, 5),
    )

    if not data:
        return False

    return bool(data.get("status") in ("ok", "healthy"))


def ppe_status(dets, required):
    """Return ok, missing or not_visible for each required PPE item.

    'missing' is only returned when the model explicitly emits a
    negative label such as no_helmet or without-vest.
    An absent positive detection is not treated as proof of non-compliance.
    """
    best = {}

    for detection in dets or []:
        if not isinstance(detection, dict):
            continue

        label = str(
            detection.get("label", "")
        ).strip().lower()

        label = label.replace(" ", "_")

        negative = False

        for prefix in ("no_", "no-", "without_", "without-"):
            if label.startswith(prefix):
                negative = True
                label = label[len(prefix):]
                break

        label = label.replace("-", "_")
        key = PPE_ALIASES.get(label)

        if not key:
            continue

        try:
            confidence = float(
                detection.get(
                    "conf",
                    detection.get("confidence", 0),
                )
            )
        except (TypeError, ValueError):
            confidence = 0.0

        if not math.isfinite(confidence):
            confidence = 0.0

        current = best.get(key)

        if current is None or confidence > current[1]:
            best[key] = (
                "missing" if negative else "ok",
                confidence,
            )

    result = {}

    for item in required or []:
        normalized_item = str(item).strip().lower()
        normalized_item = normalized_item.replace(" ", "_")
        normalized_item = PPE_ALIASES.get(
            normalized_item,
            normalized_item,
        )

        result[item] = (
            best[normalized_item][0]
            if normalized_item in best
            else "not_visible"
        )

    return result


def phone_in_hand(objs, hands, face_box, thr=0.22):
    """Check whether a detected phone is near a hand or face.

    Object boxes, hand points and face boxes must use normalized
    coordinates between 0 and 1.
    """
    phones = []

    for item in objs or []:
        label = str(item.get("label", "")).strip().lower()

        if label not in PHONE_LABELS:
            continue

        box = item.get("box")

        if isinstance(box, (list, tuple)) and len(box) == 4:
            phones.append(item)

    if not phones:
        return False, False

    points = []

    for hand in hands or []:
        hand_points = hand.get("pts") or []

        if not hand_points:
            continue

        wrist = hand_points[0]

        if (
            isinstance(wrist, (list, tuple))
            and len(wrist) >= 2
        ):
            try:
                x, y = float(wrist[0]), float(wrist[1])

                if math.isfinite(x) and math.isfinite(y):
                    points.append((x, y))
            except (TypeError, ValueError):
                pass

    if (
        isinstance(face_box, (list, tuple))
        and len(face_box) == 4
    ):
        try:
            x1, y1, x2, y2 = map(float, face_box)
            points.append((
                (x1 + x2) / 2,
                (y1 + y2) / 2,
            ))
        except (TypeError, ValueError):
            pass

    for phone in phones:
        x1, y1, x2, y2 = phone["box"]

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        for x, y in points:
            distance = math.sqrt(
                (cx - x) ** 2 + (cy - y) ** 2
            )

            if distance < thr:
                return True, True

    return False, True


def unload_models():
    """Compatibility function: models are owned by the remote service."""
    return None
