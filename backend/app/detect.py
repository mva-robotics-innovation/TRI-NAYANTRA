
"""Object and PPE detection.

Ultralytics is imported only when detection is requested.
Missing packages or model files degrade gracefully.
"""

import threading
import numpy as np

from . import config

# Do not import ultralytics or torch during application startup.
YOLO = None

_obj = None
_ppe = None

_yolo_import_tried = False
_yolo_import_lock = threading.Lock()
_model_lock = threading.Lock()
_ppe_tried = False

PPE_ALIASES = {
    "glove": "gloves",
    "gloves": "gloves",
    "hardhat": "helmet",
    "helmet": "helmet",
    "hard-hat": "helmet",
    "goggles": "goggles",
    "glasses": "goggles",
    "mask": "mask",
    "shoes": "shoes",
    "boots": "shoes",
    "safety-shoes": "shoes",
    "vest": "vest",
    "safety-vest": "vest",
}


def _load_yolo():
    """Import Ultralytics only when detection is requested."""
    global YOLO, _yolo_import_tried

    if YOLO is not None:
        return YOLO

    if _yolo_import_tried:
        return None

    with _yolo_import_lock:
        if YOLO is not None:
            return YOLO

        if _yolo_import_tried:
            return None

        _yolo_import_tried = True

        try:
            from ultralytics import YOLO as YOLOClass
            YOLO = YOLOClass
        except Exception:
            YOLO = None

    return YOLO


def yolo_available():
    """Report whether Ultralytics has been imported successfully."""
    "yolo": detect.yolo_loaded(),


def _norm(result, names):
    """Convert model detections to normalized coordinates."""
    h, w = result.orig_shape
    output = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls)

        if isinstance(names, dict):
            label = names.get(class_id, str(class_id))
        else:
            label = names[class_id]

        output.append({
            "label": str(label),
            "conf": round(float(box.conf), 2),
            "box": [
                round(x1 / w, 3),
                round(y1 / h, 3),
                round(x2 / w, 3),
                round(y2 / h, 3),
            ],
        })

    return output


def objects(bgr, conf=0.35):
    """Run general object detection. Returns None if unavailable."""
    global _obj

    YOLOClass = _load_yolo()

    if YOLOClass is None:
        return None

    try:
        with _model_lock:
            if _obj is None:
                _obj = YOLOClass(config.YOLO_MODEL)

            result = _obj.predict(
                bgr,
                conf=conf,
                imgsz=480,
                device="cpu",
                verbose=False,
            )[0]

            return _norm(result, _obj.names)

    except Exception:
        return None


def ppe_model_loaded():
    return _ppe is not None


def ppe(bgr, conf=0.35):
    """Run PPE detection if a trained PPE model exists."""
    global _ppe, _ppe_tried

    YOLOClass = _load_yolo()

    if YOLOClass is None:
        return None

    try:
        with _model_lock:
            if _ppe is None and not _ppe_tried:
                _ppe_tried = True

                if config.PPE_MODEL.exists():
                    _ppe = YOLOClass(
                        str(config.PPE_MODEL)
                    )

            if _ppe is None:
                return None

            result = _ppe.predict(
                bgr,
                conf=conf,
                imgsz=480,
                device="cpu",
                verbose=False,
            )[0]

            return _norm(result, _ppe.names)

    except Exception:
        return None


def ppe_status(dets, required):
    """Return PPE status: ok, missing, or not_visible."""
    best = {}

    for detection in dets or []:
        label = (
            detection["label"]
            .lower()
            .replace(" ", "_")
        )

        negative = label.startswith(
            ("no_", "no-", "without")
        )

        normalized = (
            label.replace("no_", "")
            .replace("no-", "")
            .replace("without_", "")
        )

        key = PPE_ALIASES.get(normalized)

        if not key:
            continue

        current = best.get(key)

        if (
            current is None
            or detection["conf"] > current[1]
        ):
            best[key] = (
                "missing" if negative else "ok",
                detection["conf"],
            )

    return {
        item: (
            best[item][0]
            if item in best
            else "not_visible"
        )
        for item in required
    }


def phone_in_hand(
    objs,
    hands,
    face_box,
    thr=0.22,
):
    """Check whether a detected phone is near a hand or face."""
    phones = [
        item
        for item in objs or []
        if item["label"].lower() == "cell phone"
    ]

    if not phones:
        return False, False

    points = [
        hand["pts"][0]
        for hand in hands
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

        if any(
            np.hypot(cx - x, cy - y) < thr
            for x, y in points
        ):
            return True, True

    return False, True


def unload_models():
    """Release references to loaded detection models."""
    global _obj, _ppe, _ppe_tried

    with _model_lock:
        _obj = None
        _ppe = None
        _ppe_tried = False
