
"""
TRI-NAYANTRA — Remote YOLO/PPE Detection Client

The main FastAPI backend does not load PyTorch,
Ultralytics or YOLO checkpoints.

All object and PPE inference is delegated to the
separate authenticated detection microservice.

Return conventions:
    None = service unavailable or inference failed.
    []   = inference succeeded, no detections.
    [...] = successful detections.
"""

import math
import logging
import threading

import cv2
import requests

from . import config


logger = logging.getLogger("tri_nayantra.detect")


# --------------------------------------------------
# HTTP SESSION
# --------------------------------------------------

_session = requests.Session()
_request_lock = threading.Lock()

_session.headers.update({
    "User-Agent": "TRI-NAYANTRA-Main-Backend/2.1"
})


# --------------------------------------------------
# LABEL NORMALIZATION
# --------------------------------------------------

PPE_ALIASES = {
    "glove": "gloves",
    "gloves": "gloves",
    "helmet": "helmet",
    "hardhat": "helmet",
    "hard_hat": "helmet",
    "hard_hat_helmet": "helmet",
    "goggles": "goggles",
    "glass": "goggles",
    "glasses": "goggles",
    "safety_glasses": "goggles",
    "mask": "mask",
    "face_mask": "mask",
    "shoe": "shoes",
    "shoes": "shoes",
    "boot": "shoes",
    "boots": "shoes",
    "safety_shoes": "shoes",
    "safety_boots": "shoes",
    "vest": "vest",
    "safety_vest": "vest",
    "reflective_vest": "vest",
}

PHONE_LABELS = {
    "cell phone",
    "cell_phone",
    "mobile phone",
    "mobile_phone",
    "phone",
}


def _clean_label(value):
    """Normalize a model class label."""
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


# --------------------------------------------------
# SERVICE CONFIGURATION
# --------------------------------------------------

def _base_url():
    return str(
        getattr(config, "DETECTION_SERVICE_URL", "") or ""
    ).strip().rstrip("/")


def _endpoint(path):
    base = _base_url()

    if not base:
        return None

    return f"{base}{path}"


def _headers():
    token = str(
        getattr(config, "DETECTION_SERVICE_TOKEN", "") or ""
    ).strip()

    if not token:
        return {}

    return {
        "Authorization": f"Bearer {token}"
    }


def _timeout():
    try:
        connect = float(
            getattr(config, "DETECTION_CONNECT_TIMEOUT", 3)
        )

        read = float(
            getattr(config, "DETECTION_READ_TIMEOUT", 20)
        )

    except (TypeError, ValueError):
        connect, read = 3.0, 20.0

    return (
        max(0.5, connect),
        max(1.0, read),
    )


def _valid_json_response(response):
    """Raise for HTTP errors and return a JSON object."""
    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload, dict):
        raise ValueError(
            "Detection service returned an invalid JSON response."
        )

    return payload


# --------------------------------------------------
# DETECTION NORMALIZATION
# --------------------------------------------------

def _normalize_detection(item):
    """Normalize one detection to label, confidence and box."""

    if not isinstance(item, dict):
        return None

    label = str(
        item.get("label", "")
    ).strip()

    if not label:
        return None

    try:
        confidence = float(
            item.get(
                "conf",
                item.get("confidence", 0),
            )
        )

    except (TypeError, ValueError):
        return None

    if not math.isfinite(confidence):
        return None

    box = item.get("box")

    if box is None and all(
        key in item
        for key in ("x1", "y1", "x2", "y2")
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
        box = [
            float(value)
            for value in box
        ]

    except (TypeError, ValueError):
        return None

    if not all(math.isfinite(value) for value in box):
        return None

    x1, y1, x2, y2 = box

    if x2 <= x1 or y2 <= y1:
        return None

    return {
        "label": label,
        "conf": max(
            0.0,
            min(1.0, confidence),
        ),
        "box": [
            max(0.0, min(1.0, x1)),
            max(0.0, min(1.0, y1)),
            max(0.0, min(1.0, x2)),
            max(0.0, min(1.0, y2)),
        ],
    }


def _normalize_detections(items):
    """Normalize a list of detection objects."""

    if not isinstance(items, list):
        return None

    normalized = []

    for item in items:
        detection = _normalize_detection(item)

        if detection is not None:
            normalized.append(detection)

    return normalized


# --------------------------------------------------
# REMOTE INFERENCE
# --------------------------------------------------

def _send_frame(path, frame, conf=0.35):
    """
    Encode a BGR image and send it to the remote service.

    Returns None on failure and [] on successful empty inference.
    """

    url = _endpoint(path)

    if not url or frame is None:
        return None

    if (
        not hasattr(frame, "shape")
        or not hasattr(frame, "size")
    ):
        return None

    try:
        if frame.size == 0 or len(frame.shape) < 2:
            return None

        confidence = float(conf)

        if not math.isfinite(confidence):
            return None

        confidence = max(
            0.05,
            min(0.95, confidence),
        )

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                80,
            ],
        )

        if not ok:
            logger.warning(
                "Could not encode camera frame."
            )
            return None

        files = {
            "file": (
                "frame.jpg",
                encoded.tobytes(),
                "image/jpeg",
            )
        }

        data = {
            "conf": str(confidence)
        }

        # A shared requests.Session is protected by the lock.
        with _request_lock:
            response = _session.post(
                url,
                files=files,
                data=data,
                headers=_headers(),
                timeout=_timeout(),
            )

        payload = _valid_json_response(response)

        detections = payload.get("detections")

        if not isinstance(detections, list):
            logger.warning(
                "Invalid detections response from %s",
                path,
            )
            return None

        return _normalize_detections(detections)

    except requests.RequestException:
        logger.warning(
            "Remote inference request failed: %s",
            path,
        )
        return None

    except (
        ValueError,
        TypeError,
        cv2.error,
    ):
        logger.exception(
            "Could not process remote inference response: %s",
            path,
        )
        return None

    except Exception:
        logger.exception(
            "Unexpected remote inference failure: %s",
            path,
        )
        return None


def objects(bgr, conf=0.35):
    """Run general YOLO object detection remotely."""

    return _send_frame(
        "/api/detection/objects",
        bgr,
        conf,
    )


def ppe(bgr, conf=0.35):
    """Run trained PPE detection remotely."""

    return _send_frame(
        "/api/detection/ppe",
        bgr,
        conf,
    )


# --------------------------------------------------
# REMOTE STATUS
# --------------------------------------------------

def _get_json(path, timeout=None):
    """Make an authenticated GET request."""

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

        return _valid_json_response(response)

    except requests.RequestException:
        logger.warning(
            "Detection service GET failed: %s",
            path,
        )
        return None

    except (
        ValueError,
        TypeError,
    ):
        logger.warning(
            "Invalid detection service response: %s",
            path,
        )
        return None

    except Exception:
        logger.exception(
            "Unexpected detection service status error."
        )
        return None


def detection_service_available():
    """Check whether the remote microservice is reachable."""

    data = _get_json(
        "/health",
        timeout=(3, 5),
    )

    return bool(
        data
        and data.get("status") in ("ok", "healthy")
    )


def model_status():
    """
    Return actual remote service and model availability.

    The status endpoint must be authenticated and should report
    whether each configured model successfully loaded.
    """

    unavailable = {
        "service_available": False,
        "yolo_model_available": False,
        "ppe_model_available": False,
        "device": "unknown",
    }

    data = _get_json(
        "/api/detection/status",
        timeout=(3, 10),
    )

    if data is None:
        return unavailable.copy()

    return {
        "service_available": True,
        "yolo_model_available": bool(
            data.get("yolo_model_available", False)
        ),
        "ppe_model_available": bool(
            data.get("ppe_model_available", False)
        ),
        "device": str(
            data.get("device", "unknown")
        ),
    }


def yolo_available():
    """Check whether the remote YOLO model is available."""

    return model_status()["yolo_model_available"]


def ppe_model_loaded():
    """Check whether the remote PPE model is available."""

    return model_status()["ppe_model_available"]


# --------------------------------------------------
# PPE INTERPRETATION
# --------------------------------------------------

def ppe_status(dets, required):
    """
    Evaluate PPE detections conservatively.

    Explicit negative labels such as no_helmet can indicate
    missing PPE.

    An item that is not detected is marked not_visible.

    Missing detections are never automatically treated as
    proof of PPE compliance.
    """

    best = {}

    for detection in dets or []:

        if not isinstance(detection, dict):
            continue

        label = _clean_label(
            detection.get("label", "")
        )

        if not label:
            continue

        negative = False

        for prefix in (
            "no_",
            "without_",
        ):
            if label.startswith(prefix):
                negative = True
                label = label[len(prefix):]
                break

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
            continue

        if (
            not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            continue

        current = best.get(key)

        if (
            current is None
            or confidence > current[1]
        ):
            best[key] = (
                "missing" if negative else "ok",
                confidence,
            )

    result = {}

    # The pipeline currently supplies required as a dictionary.
    # Its keys are the required PPE item names.
    if isinstance(required, dict):
        required_items = list(required.keys())

    elif isinstance(required, (list, tuple, set)):
        required_items = list(required)

    else:
        required_items = []

    for item in required_items:

        normalized = _clean_label(item)

        normalized = PPE_ALIASES.get(
            normalized,
            normalized,
        )

        result[str(item)] = (
            best[normalized][0]
            if normalized in best
            else "not_visible"
        )

    return result


# --------------------------------------------------
# PHONE PROXIMITY
# --------------------------------------------------

def phone_in_hand(
    objs,
    hands,
    face_box,
    thr=0.22,
):
    """
    Estimate phone proximity to a hand or face.

    This is a geometric proximity heuristic, not proof that
    a phone is being held or used.
    """

    phones = []

    for item in objs or []:

        if not isinstance(item, dict):
            continue

        label = str(
            item.get("label", "")
        ).strip().lower()

        if label not in PHONE_LABELS:
            continue

        box = item.get("box")

        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
        ):
            continue

        try:
            values = [
                float(value)
                for value in box
            ]

            if not all(
                math.isfinite(value)
                for value in values
            ):
                continue

            x1, y1, x2, y2 = values

            if x2 <= x1 or y2 <= y1:
                continue

            phones.append({
                "box": values
            })

        except (TypeError, ValueError):
            continue

    if not phones:
        return False, False

    points = []

    # Hand wrist positions.
    for hand in hands or []:

        if not isinstance(hand, dict):
            continue

        hand_points = hand.get("pts") or []

        if not hand_points:
            continue

        wrist = hand_points[0]

        if (
            not isinstance(wrist, (list, tuple))
            or len(wrist) < 2
        ):
            continue

        try:
            x = float(wrist[0])
            y = float(wrist[1])

            if (
                math.isfinite(x)
                and math.isfinite(y)
                and 0 <= x <= 1
                and 0 <= y <= 1
            ):
                points.append((x, y))

        except (TypeError, ValueError):
            continue

    # Face centre.
    if (
        isinstance(face_box, (list, tuple))
        and len(face_box) == 4
    ):
        try:
            x1, y1, x2, y2 = map(
                float,
                face_box,
            )

            if all(
                math.isfinite(v)
                for v in (x1, y1, x2, y2)
            ):
                points.append((
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                ))

        except (TypeError, ValueError):
            pass

    try:
        threshold = max(
            0.0,
            float(thr),
        )

    except (TypeError, ValueError):
        threshold = 0.22

    for phone in phones:

        x1, y1, x2, y2 = phone["box"]

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        for x, y in points:

            distance = math.sqrt(
                (cx - x) ** 2
                + (cy - y) ** 2
            )

            if distance < threshold:
                return True, True

    return False, True


# --------------------------------------------------
# COMPATIBILITY AND CLEANUP
# --------------------------------------------------

def unload_models():
    """
    Compatibility function.

    YOLO/PPE models belong to the remote service.
    No local model unloading is required.
    """
    return None


def close():
    """Close the HTTP session during application shutdown."""

    with _request_lock:
        _session.close()
