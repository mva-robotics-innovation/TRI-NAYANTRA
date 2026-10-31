
"""Remote YOLO/PPE detection client for TRI-NAYANTRA.

The main API does not load torch or ultralytics.
All YOLO and PPE inference is delegated to the detection microservice.
"""

import math
import threading

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
    try:
        connect = float(
            getattr(config, "DETECTION_CONNECT_TIMEOUT", 3)
        )
        read = float(
            getattr(config, "DETECTION_READ_TIMEOUT", 20)
        )
    except (TypeError, ValueError):
        connect, read = 3.0, 20.0

    return max(0.5, connect), max(1.0, read)


def _normalize_detection(item):
    """Normalize one detection to label, confidence and box."""
    if not isinstance(item, dict):
        return None

    label = str(item.get("label", "")).strip()

    if not label:
        return None

    try:
        confidence = float(
            item.get("conf", item.get("confidence", 0))
        )
    except (TypeError, ValueError):
        return None

    if not math.isfinite(confidence):
        return None

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

    if x2 <= x1 or y2 <= y1:
        return None

    box = [
        max(0.0, min(1.0, x1)),
        max(0.0, min(1.0, y1)),
        max(0.0, min(1.0, x2)),
        max(0.0, min(1.0, y2)),
    ]

    return {
        "label": label,
        "conf": max(0.0, min(1.0, confidence)),
        "box": box,
    }


def _send_frame(path, frame, conf=0.35):
    """Encode a BGR frame and send it to the detection service."""
    url = _endpoint(path)

    if not url or frame is None:
        return None

    if not hasattr(frame, "shape") or not hasattr(frame, "size"):
        return None

    try:
        if frame.size == 0 or len(frame.shape) < 2:
            return None

        confidence = float(conf)

        if not math.isfinite(confidence):
            return None

        confidence = max(0.05, min(0.95, confidence))

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

        data = {"conf": str(confidence)}

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
    """Run remote general object detection."""
    return _send_frame(
        "/api/detection/objects",
        bgr,
        conf,
    )


def ppe(bgr, conf=0.35):
    """Run remote PPE detection."""
    return _send_frame(
        "/api/detection/ppe",
        bgr,
        conf,
    )


def _get_json(path, timeout=None):
    """Make an authenticated GET request to the detection service."""
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


def detection_service_available():
    """Check whether the detection microservice is reachable."""
    data = _get_json(
        "/health",
        timeout=(3, 5),
    )

    return bool(
        data and data.get("status") in ("ok", "healthy")
    )


def model_status():
    """Return the actual YOLO and PPE model availability."""
    data = _get_json(
        "/api/detection/status",
        timeout=(3, 10),
    )

    if not data:
        return {
            "service_available": False,
            "yolo_model_available": False,
            "ppe_model_available": False,
        }

    return {
        "service_available": True,
        "yolo_model_available": bool(
            data.get("yolo_model_available", False)
        ),
        "ppe_model_available": bool(
            data.get("ppe_model_available", False)
        ),
    }


def yolo_available():
    """Check whether the YOLO model is actually loaded and available."""
    return model_status()["yolo_model_available"]


def ppe_model_loaded():
    """Check whether the separately trained PPE model is available."""
    return model_status()["ppe_model_available"]


def ppe_status(dets, required):
    """Evaluate PPE detections conservatively.

    An item is marked missing only when the model explicitly returns
    a negative label such as no_helmet or without_vest.

    An item that is not detected is marked not_visible, not missing.
    """
    best = {}

    for detection in dets or []:
        if not isinstance(detection, dict):
            continue

        label = str(
            detection.get("label", "")
        ).strip().lower().replace(" ", "_")

        if not label:
            continue

        negative = False

        for prefix in (
            "no_",
            "no-",
            "without_",
            "without-",
        ):
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
            continue

        if not math.isfinite(confidence):
            continue

        current = best.get(key)

        if current is None or confidence > current[1]:
            best[key] = (
                "missing" if negative else "ok",
                confidence,
            )

    result = {}

    for item in required or []:
        normalized = str(item).strip().lower()
        normalized = normalized.replace(" ", "_")

        normalized = PPE_ALIASES.get(
            normalized,
            normalized,
        )

        result[item] = (
            best[normalized][0]
            if normalized in best
            else "not_visible"
        )

    return result


def phone_in_hand(objs, hands, face_box, thr=0.22):
    """Estimate whether a detected phone is close to a hand or face.

    This is a geometric proximity heuristic, not proof that a phone
    is being held.
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

        if isinstance(box, (list, tuple)) and len(box) == 4:
            try:
                values = [float(v) for v in box]

                if all(math.isfinite(v) for v in values):
                    phones.append({
                        "box": values,
                    })
            except (TypeError, ValueError):
                continue

    if not phones:
        return False, False

    points = []

    for hand in hands or []:
        if not isinstance(hand, dict):
            continue

        hand_points = hand.get("pts") or []

        if not hand_points:
            continue

        wrist = hand_points[0]

        if isinstance(wrist, (list, tuple)) and len(wrist) >= 2:
            try:
                x, y = float(wrist[0]), float(wrist[1])

                if (
                    math.isfinite(x)
                    and math.isfinite(y)
                    and 0 <= x <= 1
                    and 0 <= y <= 1
                ):
                    points.append((x, y))
            except (TypeError, ValueError):
                continue

    if isinstance(face_box, (list, tuple)) and len(face_box) == 4:
        try:
            x1, y1, x2, y2 = map(float, face_box)

            if all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                points.append((
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                ))
        except (TypeError, ValueError):
            pass

    try:
        threshold = max(0.0, float(thr))
    except (TypeError, ValueError):
        threshold = 0.22

    for phone in phones:
        x1, y1, x2, y2 = phone["box"]

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        for x, y in points:
            distance = math.sqrt(
                (cx - x) ** 2 + (cy - y) ** 2
            )

            if distance < threshold:
                return True, True

    return False, True


def unload_models():
    """Compatibility function: models belong to the remote service."""
    return None


def close():
    """Close the HTTP session during application shutdown."""
    with _request_lock:
        _session.close()
