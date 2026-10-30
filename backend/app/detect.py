"""Object + PPE detection (Ultralytics YOLO). Models load lazily; missing models degrade gracefully."""
import numpy as np
from . import config
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

_obj = _ppe = None; _ppe_tried = False
PPE_ALIASES = {"glove": "gloves", "gloves": "gloves", "hardhat": "helmet", "helmet": "helmet", "hard-hat": "helmet",
               "goggles": "goggles", "glasses": "goggles", "mask": "mask", "shoes": "shoes", "boots": "shoes",
               "safety-shoes": "shoes", "vest": "vest", "safety-vest": "vest"}

def _norm(r, names):
    h, w = r.orig_shape; out = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append({"label": names[int(b.cls)], "conf": round(float(b.conf), 2),
                    "box": [round(x1 / w, 3), round(y1 / h, 3), round(x2 / w, 3), round(y2 / h, 3)]})
    return out

def objects(bgr, conf=0.35):
    global _obj
    if YOLO is None: return None
    if _obj is None: _obj = YOLO(config.YOLO_MODEL)
    r = _obj.predict(bgr, conf=conf, imgsz=640, verbose=False)[0]
    return _norm(r, _obj.names)

def ppe_model_loaded():
    return _ppe is not None

def ppe(bgr, conf=0.35):
    """Returns raw PPE detections or None if no PPE model is installed (see download_models.py)."""
    global _ppe, _ppe_tried
    if YOLO is None: return None
    if _ppe is None and not _ppe_tried:
        _ppe_tried = True
        if config.PPE_MODEL.exists(): _ppe = YOLO(str(config.PPE_MODEL))
    if _ppe is None: return None
    r = _ppe.predict(bgr, conf=conf, imgsz=640, verbose=False)[0]
    return _norm(r, _ppe.names)

def ppe_status(dets, required):
    """item -> 'ok' | 'missing' | 'not_visible'. Handles classes like 'helmet' / 'no_helmet'."""
    best = {}
    for d in dets or []:
        lab = d["label"].lower().replace(" ", "_"); neg = lab.startswith(("no_", "no-", "without"))
        key = PPE_ALIASES.get(lab.replace("no_", "").replace("no-", "").replace("without_", ""))
        if not key: continue
        cur = best.get(key)
        if cur is None or d["conf"] > cur[1]: best[key] = ("missing" if neg else "ok", d["conf"])
    return {k: (best[k][0] if k in best else "not_visible") for k in required}

def phone_in_hand(objs, hands, face_box, thr=0.22):
    """True if a detected phone is close to a wrist/hand or the face."""
    phones = [o for o in objs or [] if o["label"] == "cell phone"]
    if not phones: return False, False
    pts = [h["pts"][0] for h in hands]
    if face_box: pts.append([(face_box[0] + face_box[2]) / 2, (face_box[1] + face_box[3]) / 2])
    for p in phones:
        cx, cy = (p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2
        if any(np.hypot(cx - x, cy - y) < thr for x, y in pts): return True, True
    return False, True   # phone present but not in hand
