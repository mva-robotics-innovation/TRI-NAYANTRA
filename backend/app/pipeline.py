
"""
TRI-NAYANTRA — Mode Processing Pipeline

Main backend responsibilities:
- MediaPipe vision
- Remote YOLO object detection
- Remote PPE detection
- Workplace monitoring
- Student attention monitoring
- Gesture and care assistance
- Advanced pseudo-thermal visualization

YOLO/PPE inference runs exclusively in the separate
detection microservice.

PPE compliance is never assumed when detection is unavailable.
"""

import json
import time
import threading
import logging

import requests

from . import config, vision, detect, bodyscan


logger = logging.getLogger("tri_nayantra.pipeline")


# --------------------------------------------------
# EVENT NOTIFICATIONS
# --------------------------------------------------

_EVENT_LOCK = threading.Lock()


def notify(kind, detail, mode="", severe=False):
    """Record an event and optionally send a webhook."""

    record = {
        "ts": round(time.time(), 1),
        "mode": str(mode),
        "kind": str(kind),
        "detail": str(detail),
        "severe": bool(severe),
    }

    try:
        config.DATA.mkdir(parents=True, exist_ok=True)

        with _EVENT_LOCK:
            with open(
                config.DATA / "events.jsonl",
                "a",
                encoding="utf8",
            ) as file:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

    except OSError:
        logger.exception("Could not write event record")

    if severe and config.ALERT_WEBHOOK_URL:
        threading.Thread(
            target=_send_webhook,
            args=(config.ALERT_WEBHOOK_URL, record),
            daemon=True,
        ).start()

    return record


def _send_webhook(url, record):
    """Send an event to the configured webhook."""

    try:
        response = requests.post(url, json=record, timeout=5)
        response.raise_for_status()

    except requests.RequestException:
        logger.exception("Alert webhook delivery failed")


# --------------------------------------------------
# ALERT THROTTLING
# --------------------------------------------------

def _throttle(sess, kind, gap=8):
    """Prevent repeated notifications for the same event."""

    now = time.time()

    if not hasattr(sess, "last_event"):
        sess.last_event = {}

    previous = sess.last_event.get(kind, 0)

    if now - previous < gap:
        return False

    sess.last_event[kind] = now
    return True


def _alert(
    sess,
    alerts,
    kind,
    text,
    level="warn",
    severe=False,
):
    """Add an alert and record it when not throttled."""

    alerts.append({
        "kind": kind,
        "text": text,
        "level": level,
    })

    if _throttle(sess, kind):
        notify(
            kind,
            text,
            getattr(sess, "mode", ""),
            severe,
        )


# --------------------------------------------------
# SESSION CACHE
# --------------------------------------------------

def _session_cache(sess):
    if not hasattr(sess, "cache") or not isinstance(sess.cache, dict):
        sess.cache = {}

    return sess.cache


def _cached_remote_detection(
    sess,
    frame,
    cache_key,
    detector_function,
    interval,
):
    """
    Cache remote inference using elapsed time.

    interval:
        Minimum seconds between successful inference attempts.

    Successful empty result:
        [] — inference completed, no detections.

    Unavailable result:
        None — inference could not be completed.

    Failed requests are retried after a short cooldown.
    """

    cache = _session_cache(sess)
    now = time.monotonic()

    cached = cache.get(cache_key)

    if cached is not None:
        retry_at = cached.get("retry_at", 0)

        if now < retry_at:
            return cached.get("result")

    try:
        result = detector_function(frame)

        if result is not None and not isinstance(result, list):
            logger.warning(
                "Unexpected remote detection response for %s",
                cache_key,
            )
            result = None

        # Successful results are refreshed at the configured interval.
        cooldown = interval if result is not None else 1.0

        cache[cache_key] = {
            "result": result,
            "retry_at": now + max(0.1, cooldown),
        }

        return result

    except Exception:
        logger.exception(
            "Remote detection failed: %s",
            cache_key,
        )

        cache[cache_key] = {
            "result": None,
            "retry_at": now + 1.0,
        }

        return None


def _cached_objects(sess, frame):
    """Run remote object detection at most twice per second."""

    return _cached_remote_detection(
        sess,
        frame,
        "remote_objects",
        detect.objects,
        0.5,
    )


def _cached_ppe(sess, frame):
    """Run remote PPE inference at most once every 0.8 seconds."""

    return _cached_remote_detection(
        sess,
        frame,
        "remote_ppe",
        detect.ppe,
        0.8,
    )


# --------------------------------------------------
# DETECTION SERVICE STATUS
# --------------------------------------------------

def _service_status(sess, force=False):
    """
    Retrieve and cache remote service/model status.

    The status endpoint is not called on every camera frame.
    """

    cache = _session_cache(sess)
    now = time.monotonic()

    cached = cache.get("remote_service_status")

    if (
        not force
        and cached is not None
        and now < cached.get("expires_at", 0)
    ):
        return cached["result"]

    default = {
        "service_available": False,
        "yolo_model_available": False,
        "ppe_model_available": False,
    }

    try:
        result = detect.model_status()

        if not isinstance(result, dict):
            result = default.copy()
        else:
            result = {
                "service_available": bool(
                    result.get("service_available", False)
                ),
                "yolo_model_available": bool(
                    result.get("yolo_model_available", False)
                ),
                "ppe_model_available": bool(
                    result.get("ppe_model_available", False)
                ),
            }

    except Exception:
        logger.exception(
            "Could not retrieve detection service status"
        )
        result = default.copy()

    cache["remote_service_status"] = {
        "result": result,
        "expires_at": now + 5.0,
    }

    return result


# --------------------------------------------------
# PHONE DETECTION
# --------------------------------------------------

def _phone_analysis(objects, hands, face_box):
    """Estimate phone proximity to hands or face."""

    if objects is None:
        return {
            "in_use": False,
            "present": False,
            "available": False,
        }

    try:
        in_hand, present = detect.phone_in_hand(
            objects,
            hands,
            face_box,
        )

        return {
            "in_use": bool(in_hand),
            "present": bool(present),
            "available": True,
        }

    except Exception:
        logger.exception("Phone analysis failed")

        return {
            "in_use": False,
            "present": False,
            "available": False,
        }


# --------------------------------------------------
# PPE STATUS
# --------------------------------------------------

def _ppe_analysis(sess, raw):
    """
    Calculate PPE status only after successful PPE inference.

    None means unavailable.
    An empty list means inference completed with no detections.
    Neither is automatically treated as PPE compliance.
    """

    if raw is None:
        return {
            "model_available": False,
            "raw": None,
            "status": None,
        }

    try:
        cfg = getattr(sess, "cfg", {}) or {}
        required = cfg.get("required", {})

        status = detect.ppe_status(raw, required)

        if not isinstance(status, dict):
            status = None

        return {
            "model_available": status is not None,
            "raw": raw,
            "status": status,
        }

    except Exception:
        logger.exception("PPE status calculation failed")

        return {
            "model_available": False,
            "raw": None,
            "status": None,
        }


# --------------------------------------------------
# WORKPLACE MODE
# --------------------------------------------------

def _workplace(
    sess,
    frame,
    out,
    alerts,
    face,
    hands,
):
    """Workplace monitoring logic."""

    objects = _cached_objects(sess, frame)

    out["objects"] = objects
    out["object_detection_available"] = objects is not None

    service = _service_status(sess)

    out["detection_service_available"] = service[
        "service_available"
    ]
    out["yolo_model_available"] = service[
        "yolo_model_available"
    ]

    # Person count
    if objects is None:
        out["persons"] = None
    else:
        out["persons"] = sum(
            1
            for obj in objects
            if isinstance(obj, dict)
            and str(obj.get("label", "")).lower() == "person"
        )

    # Phone detection
    phone = _phone_analysis(
        objects,
        hands,
        out.get("face_box"),
    )

    out["phone"] = phone

    if phone["in_use"]:
        _alert(
            sess,
            alerts,
            "phone",
            "Mobile phone proximity detected - review required",
            "danger",
        )

    # Possible fall
    if out.get("fall"):
        _alert(
            sess,
            alerts,
            "fall",
            "Possible fall detected. Please verify immediately.",
            "danger",
            True,
        )

    # Help signal
    if out.get("help_signal"):
        _alert(
            sess,
            alerts,
            "help",
            "HELP signal detected.",
            "danger",
            True,
        )

    # PPE detection
    raw_ppe = _cached_ppe(sess, frame)

    ppe_result = _ppe_analysis(sess, raw_ppe)

    out["ppe_model"] = ppe_result["model_available"]
    out["ppe_raw"] = ppe_result["raw"]
    out["ppe"] = ppe_result["status"]

    out["ppe_message"] = (
        "PPE inference completed."
        if ppe_result["model_available"]
        else "PPE detection unavailable. Configure a compatible trained PPE model."
    )

    # Only report missing PPE when a valid PPE status exists.
    if ppe_result["status"]:
        missing = [
            name
            for name, value in ppe_result["status"].items()
            if value == "missing"
        ]

        if missing:
            _alert(
                sess,
                alerts,
                "ppe",
                "Potential missing PPE: " + ", ".join(missing),
                "danger",
                True,
            )

    # Worker state
    if objects is None:
        state = "detection_unavailable"

    elif not out.get("person"):
        state = "no_worker"

    elif phone["in_use"]:
        state = "distracted"

    elif out.get("away_for_s", 0) > 3:
        state = "distracted"

    elif face.get("expression") in ("drowsy", "yawning"):
        state = "fatigued"

    elif out.get("idle_s", 0) > 10:
        state = "idle"

    else:
        state = "working"

    out["state"] = state

    if state == "fatigued":
        _alert(
            sess,
            alerts,
            "fatigue",
            "Possible fatigue indicators detected.",
            "warn",
        )

    if state == "idle" and out.get("idle_s", 0) > 30:
        _alert(
            sess,
            alerts,
            "idle",
            "No activity detected for 30 seconds.",
            "warn",
        )


# --------------------------------------------------
# STUDENT MODE
# --------------------------------------------------

def _student(
    sess,
    frame,
    out,
    alerts,
    face,
    hands,
):
    """Student attention monitoring logic."""

    objects = _cached_objects(sess, frame)

    out["objects"] = objects
    out["object_detection_available"] = objects is not None

    phone = _phone_analysis(
        objects,
        hands,
        out.get("face_box"),
    )

    out["phone"] = phone

    person_missing = (
        objects is not None
        and not out.get("person")
    )

    eyes_distracted = (
        face.get("eyes_closed", False)
        and face.get("perclos", 0) > 0.3
    )

    looking_away = out.get("away_for_s", 0) > 2

    bad = (
        phone["in_use"]
        or looking_away
        or eyes_distracted
        or person_missing
    )

    # Do not claim 100% attention when detection is unavailable.
    if objects is None:
        out["attention"] = None
        out["state"] = "detection_unavailable"

    else:
        if phone["in_use"]:
            target = 0
        elif bad:
            target = 15
        else:
            target = 100

        if not hasattr(sess, "attn"):
            sess.attn = 100.0

        sess.attn += (target - sess.attn) * 0.08
        sess.attn = max(0.0, min(100.0, sess.attn))

        out["attention"] = round(sess.attn)

        out["state"] = (
            "attentive"
            if sess.attn >= 70
            else "partially attentive"
            if sess.attn >= 40
            else "distracted"
        )

    if out["state"] == "distracted":
        _alert(
            sess,
            alerts,
            "inattentive",
            "Possible student distraction detected.",
            "warn",
        )

    if out.get("fall"):
        _alert(
            sess,
            alerts,
            "fall",
            "Possible fall detected. Please verify immediately.",
            "danger",
            True,
        )

    if out.get("help_signal"):
        _alert(
            sess,
            alerts,
            "help",
            "HELP signal detected.",
            "danger",
            True,
        )


# --------------------------------------------------
# GESTURE AND CARE MODES
# --------------------------------------------------

def _care_and_gesture(sess, out, alerts):
    """Gesture and care assistance logic."""

    if out.get("help_signal"):
        _alert(
            sess,
            alerts,
            "help",
            "HELP signal detected.",
            "danger",
            True,
        )

    if out.get("fall"):
        _alert(
            sess,
            alerts,
            "fall",
            "Possible fall detected.",
            "danger",
            True,
        )


# --------------------------------------------------
# ADVANCED MODE
# --------------------------------------------------

def _advanced(frame, out, mask):
    """Generate an illustrative pseudo-thermal visualization."""

    thermal, gray = bodyscan.thermal_view(frame, mask)

    out["thermal"] = thermal

    out["organs"] = bodyscan.organ_map(
        out.get("pose"),
        out["w"],
        out["h"],
        gray,
    )

    out["thermal_type"] = "pseudo_thermal_visualization"
    out["medical_diagnostic"] = False


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def process(sess, frame):
    """Process one camera frame for the selected mode."""

    if getattr(sess, "closed", False):
        raise RuntimeError(
            "Cannot process a closed camera session."
        )

    if frame is None:
        raise ValueError("Camera frame is empty.")

    if not hasattr(sess, "frame_i"):
        sess.frame_i = 0

    sess.frame_i += 1

    mode = getattr(sess, "mode", "")

    # MediaPipe processing remains local.
    out = vision.analyze(sess, frame)

    if not isinstance(out, dict):
        raise RuntimeError(
            "Vision pipeline returned an invalid response."
        )

    mask = out.pop("mask", None)

    alerts = []

    face = out.get("face") or {}
    hands = out.get("hands") or []

    if out.get("error"):
        out["alerts"] = []
        out["pipeline_status"] = "vision_unavailable"
        return out

    # Workplace monitoring
    if mode == "workplace":
        _workplace(
            sess,
            frame,
            out,
            alerts,
            face,
            hands,
        )

    # Student attention monitoring
    elif mode == "student":
        _student(
            sess,
            frame,
            out,
            alerts,
            face,
            hands,
        )

    # Gesture / Care
    elif mode in ("gesture", "care"):
        _care_and_gesture(
            sess,
            out,
            alerts,
        )

    # Advanced visualization
    elif mode == "advanced":
        try:
            _advanced(frame, out, mask)

        except Exception:
            logger.exception(
                "Advanced visualization failed"
            )

            out["thermal"] = None
            out["organs"] = []
            out["thermal_error"] = (
                "Advanced visualization unavailable."
            )

    # Health indicators
    elif mode == "health":
        out["health_indicators"] = {
            "face_visible": bool(face.get("visible", False)),
            "eyes_closed": bool(face.get("eyes_closed", False)),
            "possible_yawning": bool(face.get("yawning", False)),
            "experimental_heart_rate": face.get(
                "heart_rate_estimate"
            ),
            "heart_rate_type": face.get(
                "heart_rate_type",
                "unavailable",
            ),
        }

        out["medical_diagnostic"] = False

    # One cached status response for monitoring modes.
    if mode in ("workplace", "student"):
        out["detection_service"] = _service_status(sess)

    out["alerts"] = alerts
    out["pipeline_status"] = "ok"

    return out
