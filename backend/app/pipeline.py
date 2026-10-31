
"""
TRI-NAYANTRA — Mode Processing Pipeline

Connects:
- MediaPipe vision engine
- Remote YOLO object detection service
- Remote PPE detection service
- Workplace monitoring
- Student attention monitoring
- Gesture and care assistance
- Advanced pseudo-thermal visualization

YOLO/PPE inference is performed by detect.py,
which communicates with the separate detection service.

PPE compliance is never reported as safe when
the PPE model is unavailable.
"""

import json
import time
import threading
import logging

import requests

from . import (
    config,
    vision,
    detect,
    bodyscan,
)


logger = logging.getLogger("tri-nayantra.pipeline")


# --------------------------------------------------
# EVENT NOTIFICATIONS
# --------------------------------------------------

_EVENT_LOCK = threading.Lock()


def notify(
    kind,
    detail,
    mode="",
    severe=False,
):
    """
    Record an event and optionally send a webhook.

    Event recording is local.
    Webhook delivery runs in a background thread.
    """

    rec = {
        "ts": round(
            time.time(),
            1,
        ),
        "mode": str(mode),
        "kind": str(kind),
        "detail": str(detail),
        "severe": bool(severe),
    }

    try:

        config.DATA.mkdir(
            parents=True,
            exist_ok=True,
        )

        with _EVENT_LOCK:

            with open(
                config.DATA / "events.jsonl",
                "a",
                encoding="utf8",
            ) as file:

                file.write(
                    json.dumps(
                        rec,
                        ensure_ascii=False,
                    ) + "\n"
                )

    except OSError:
        logger.exception(
            "Could not write event record"
        )

    # Send severe events to configured webhook.
    if (
        severe
        and config.ALERT_WEBHOOK_URL
    ):

        thread = threading.Thread(
            target=_send_webhook,
            args=(
                config.ALERT_WEBHOOK_URL,
                rec,
            ),
            daemon=True,
        )

        thread.start()

    return rec


def _send_webhook(url, record):
    """Send an event to the configured webhook."""

    try:

        response = requests.post(
            url,
            json=record,
            timeout=5,
        )

        response.raise_for_status()

    except requests.RequestException:
        logger.exception(
            "Alert webhook delivery failed"
        )


# --------------------------------------------------
# ALERT THROTTLING
# --------------------------------------------------

def _throttle(
    sess,
    kind,
    gap=8,
):
    """
    Prevent the same alert from being recorded
    on every camera frame.
    """

    now = time.time()

    previous = sess.last_event.get(
        kind,
        0,
    )

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
    """Add an alert to the current frame."""

    alerts.append({
        "kind": kind,
        "text": text,
        "level": level,
    })

    if _throttle(
        sess,
        kind,
    ):
        notify(
            kind,
            text,
            sess.mode,
            severe,
        )


# --------------------------------------------------
# DETECTION CACHE
# --------------------------------------------------

def _cached_objects(
    sess,
    frame,
):
    """
    Run remote object detection every second frame.

    None means the service is unavailable or the
    inference failed. An empty list means inference
    completed successfully but no objects were found.
    """

    if (
        sess.frame_i % 2 == 1
        or "objs" not in sess.cache
    ):

        try:
            sess.cache["objs"] = (
                detect.objects(frame)
            )

        except Exception:
            logger.exception(
                "Remote object detection failed"
            )

            sess.cache["objs"] = None

    return sess.cache.get(
        "objs"
    )


def _cached_ppe(
    sess,
    frame,
):
    """
    Run remote PPE detection every third frame.

    None means PPE inference is unavailable.
    """

    if (
        sess.frame_i % 3 == 1
        or "ppe" not in sess.cache
    ):

        try:
            sess.cache["ppe"] = (
                detect.ppe(frame)
            )

        except Exception:
            logger.exception(
                "Remote PPE detection failed"
            )

            sess.cache["ppe"] = None

    return sess.cache.get(
        "ppe"
    )


# --------------------------------------------------
# PHONE DETECTION
# --------------------------------------------------

def _phone_analysis(
    objects,
    hands,
    face_box,
):
    """
    Analyze phone detections relative to hand positions.

    The result is unknown when object detection fails.
    """

    if objects is None:

        return {
            "in_use": False,
            "present": False,
            "available": False,
        }

    try:

        in_hand, present = (
            detect.phone_in_hand(
                objects,
                hands,
                face_box,
            )
        )

        return {
            "in_use": bool(in_hand),
            "present": bool(present),
            "available": True,
        }

    except Exception:
        logger.exception(
            "Phone analysis failed"
        )

        return {
            "in_use": False,
            "present": False,
            "available": False,
        }


# --------------------------------------------------
# PPE STATUS
# --------------------------------------------------

def _ppe_analysis(
    sess,
    raw,
):
    """
    Return PPE results only when inference succeeds.

    Do not convert an unavailable model into an
    apparently compliant PPE result.
    """

    if raw is None:

        return {
            "model_available": False,
            "raw": None,
            "status": None,
        }

    try:

        required = sess.cfg.get(
            "required",
            {},
        )

        status = detect.ppe_status(
            raw,
            required,
        )

        return {
            "model_available": True,
            "raw": raw,
            "status": status,
        }

    except Exception:
        logger.exception(
            "PPE status calculation failed"
        )

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

    objects = _cached_objects(
        sess,
        frame,
    )

    out["objects"] = objects

    # ----------------------------------------------
    # Object detection availability
    # ----------------------------------------------

    out["object_detection_available"] = (
        objects is not None
    )

    # ----------------------------------------------
    # Person count
    # ----------------------------------------------

    if objects is None:

        out["persons"] = None

    else:

        out["persons"] = sum(
            1
            for obj in objects
            if obj.get("label") == "person"
        )

    # ----------------------------------------------
    # Phone detection
    # ----------------------------------------------

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
            "Mobile phone in use - DISTRACTED",
            "danger",
        )

    # ----------------------------------------------
    # Possible fall and help signal
    # ----------------------------------------------

    if out.get("fall"):

        _alert(
            sess,
            alerts,
            "fall",
            "Possible fall detected",
            "danger",
            True,
        )

    if out.get("help_signal"):

        _alert(
            sess,
            alerts,
            "help",
            "HELP signal shown",
            "danger",
            True,
        )

    # ----------------------------------------------
    # PPE detection
    # ----------------------------------------------

    raw_ppe = _cached_ppe(
        sess,
        frame,
    )

    ppe_result = _ppe_analysis(
        sess,
        raw_ppe,
    )

    out["ppe_model"] = (
        ppe_result["model_available"]
    )

    out["ppe_raw"] = (
        ppe_result["raw"]
    )

    out["ppe"] = (
        ppe_result["status"]
    )

    if not ppe_result["model_available"]:

        out["ppe_message"] = (
            "PPE detection unavailable. "
            "Configure a compatible trained PPE model."
        )

    else:

        out["ppe_message"] = (
            "PPE inference completed."
        )

        ppe_status = (
            ppe_result["status"]
        )

        if ppe_status:

            missing = [
                name
                for name, value in ppe_status.items()
                if value == "missing"
            ]

            if missing:

                _alert(
                    sess,
                    alerts,
                    "ppe",
                    "Missing PPE: " + ", ".join(missing),
                    "danger",
                    True,
                )

    # ----------------------------------------------
    # Worker state
    # ----------------------------------------------

    # Do not claim a worker is absent when object
    # detection is unavailable.
    if objects is None:

        state = "detection_unavailable"

    elif not out.get("person"):

        state = "no_worker"

    elif phone["in_use"]:

        state = "distracted"

    elif (
        out.get("away_for_s", 0) > 3
    ):

        state = "distracted"

    elif (
        face.get("expression")
        in ("drowsy", "yawning")
    ):

        state = "fatigued"

    elif out.get("idle_s", 0) > 10:

        state = "idle"

    else:

        state = "working"

    out["state"] = state

    # ----------------------------------------------
    # State alerts
    # ----------------------------------------------

    if state == "fatigued":

        _alert(
            sess,
            alerts,
            "fatigue",
            "Possible fatigue indicators detected",
            "warn",
        )

    if (
        state == "idle"
        and out.get("idle_s", 0) > 30
    ):

        _alert(
            sess,
            alerts,
            "idle",
            "No activity detected for 30 seconds",
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

    objects = _cached_objects(
        sess,
        frame,
    )

    out["objects"] = objects

    out["object_detection_available"] = (
        objects is not None
    )

    phone = _phone_analysis(
        objects,
        hands,
        out.get("face_box"),
    )

    out["phone"] = phone

    # Only use absence of person as a distraction
    # signal when object detection is available.
    person_missing = (
        objects is not None
        and not out.get("person")
    )

    eyes_distracted = (
        face.get("eyes_closed", False)
        and face.get("perclos", 0) > 0.3
    )

    looking_away = (
        out.get("away_for_s", 0) > 2
    )

    bad = (
        phone["in_use"]
        or looking_away
        or eyes_distracted
        or person_missing
    )

    # Unknown detection should not be presented as
    # confirmed distraction.
    if objects is None:

        target = 100

    elif phone["in_use"]:

        target = 0

    elif bad:

        target = 15

    else:

        target = 100

    sess.attn += (
        target - sess.attn
    ) * 0.08

    sess.attn = max(
        0.0,
        min(100.0, sess.attn),
    )

    out["attention"] = round(
        sess.attn
    )

    if objects is None:

        out["state"] = (
            "detection_unavailable"
        )

    else:

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
            "Possible student distraction detected",
            "warn",
        )

    if out.get("fall"):

        _alert(
            sess,
            alerts,
            "fall",
            "Possible fall detected",
            "danger",
            True,
        )

    if out.get("help_signal"):

        _alert(
            sess,
            alerts,
            "help",
            "HELP signal shown",
            "danger",
            True,
        )


# --------------------------------------------------
# GESTURE AND CARE MODES
# --------------------------------------------------

def _care_and_gesture(
    sess,
    out,
    alerts,
):
    """Gesture and care assistance logic."""

    if out.get("help_signal"):

        _alert(
            sess,
            alerts,
            "help",
            "HELP signal shown",
            "danger",
            True,
        )

    if out.get("fall"):

        _alert(
            sess,
            alerts,
            "fall",
            "Possible fall detected",
            "danger",
            True,
        )


# --------------------------------------------------
# ADVANCED MODE
# --------------------------------------------------

def _advanced(
    frame,
    out,
    mask,
):
    """
    Generate an illustrative pseudo-thermal image.

    This is not a thermal camera measurement or
    medical visualization of internal organs.
    """

    thermal, gray = (
        bodyscan.thermal_view(
            frame,
            mask,
        )
    )

    out["thermal"] = thermal

    out["organs"] = (
        bodyscan.organ_map(
            out.get("pose"),
            out["w"],
            out["h"],
            gray,
        )
    )

    out["thermal_type"] = (
        "pseudo_thermal_visualization"
    )

    out["medical_diagnostic"] = False


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def process(
    sess,
    frame,
):
    """
    Process one camera frame for the selected mode.

    Returns the frontend-compatible response dictionary.
    """

    if sess.closed:
        raise RuntimeError(
            "Cannot process a closed camera session."
        )

    mode = sess.mode

    # ----------------------------------------------
    # MediaPipe core
    # ----------------------------------------------

    out = vision.analyze(
        sess,
        frame,
    )

    # Keep mask internal for advanced mode.
    mask = out.pop(
        "mask",
        None,
    )

    alerts = []

    face = (
        out.get("face")
        or {}
    )

    hands = (
        out.get("hands")
        or []
    )

    # ----------------------------------------------
    # Handle MediaPipe failure
    # ----------------------------------------------

    if out.get("error"):

        out["alerts"] = []

        out["pipeline_status"] = (
            "vision_unavailable"
        )

        return out

    # ----------------------------------------------
    # Workplace mode
    # ----------------------------------------------

    if mode == "workplace":

        _workplace(
            sess,
            frame,
            out,
            alerts,
            face,
            hands,
        )

    # ----------------------------------------------
    # Student mode
    # ----------------------------------------------

    elif mode == "student":

        _student(
            sess,
            frame,
            out,
            alerts,
            face,
            hands,
        )

    # ----------------------------------------------
    # Gesture and care
    # ----------------------------------------------

    elif mode in (
        "gesture",
        "care",
    ):

        _care_and_gesture(
            sess,
            out,
            alerts,
        )

    # ----------------------------------------------
    # Advanced mode
    # ----------------------------------------------

    elif mode == "advanced":

        try:

            _advanced(
                frame,
                out,
                mask,
            )

        except Exception:
            logger.exception(
                "Advanced visualization failed"
            )

            out["thermal"] = None
            out["organs"] = []
            out["thermal_error"] = (
                "Advanced visualization unavailable."
            )

    # ----------------------------------------------
    # Health mode
    # ----------------------------------------------

    elif mode == "health":

        out["health_indicators"] = {
            "face_visible": bool(
                face.get("visible", False)
            ),
            "eyes_closed": bool(
                face.get("eyes_closed", False)
            ),
            "possible_yawning": bool(
                face.get("yawning", False)
            ),
            "experimental_heart_rate": (
                face.get("heart_rate_estimate")
            ),
            "heart_rate_type": (
                face.get("heart_rate_type", "unavailable")
            ),
        }

        out["medical_diagnostic"] = False

    # ----------------------------------------------
    # Final response
    # ----------------------------------------------

    out["alerts"] = alerts

    out["pipeline_status"] = "ok"

    return out
