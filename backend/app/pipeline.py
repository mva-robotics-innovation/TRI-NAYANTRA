"""Mode-specific logic on top of the vision core."""
import json, time, threading
import requests
from . import config, vision, detect, bodyscan

def notify(kind, detail, mode="", severe=False):
    rec = {"ts": round(time.time(), 1), "mode": mode, "kind": kind, "detail": detail}
    with open(config.DATA / "events.jsonl", "a", encoding="utf8") as f: f.write(json.dumps(rec) + "\n")
    if severe and config.ALERT_WEBHOOK_URL:
        threading.Thread(target=lambda: requests.post(config.ALERT_WEBHOOK_URL, json=rec, timeout=5), daemon=True).start()

def _throttle(sess, kind, gap=8):
    n = time.time()
    if n - sess.last_event.get(kind, 0) < gap: return False
    sess.last_event[kind] = n; return True

def _alert(sess, alerts, kind, text, level="warn", severe=False):
    alerts.append({"kind": kind, "text": text, "level": level})
    if _throttle(sess, kind): notify(kind, text, sess.mode, severe)

def process(sess, frame):
    mode = sess.mode; out = vision.analyze(sess, frame); mask = out.pop("mask", None); alerts = []
    f = out.get("face") or {}; hands = out.get("hands", [])
    if mode in ("workplace", "student"):
        if sess.frame_i % 2 == 1 or "objs" not in sess.cache: sess.cache["objs"] = detect.objects(frame)
        objs = sess.cache["objs"]; out["objects"] = objs
        in_hand, present = detect.phone_in_hand(objs, hands, out.get("face_box"))
        out["phone"] = {"in_use": in_hand, "present": present}
        if in_hand: _alert(sess, alerts, "phone", "Mobile phone in use - DISTRACTED", "danger")
        if out.get("fall"): _alert(sess, alerts, "fall", "Possible fall detected", "danger", True)
        if out.get("help_signal"): _alert(sess, alerts, "help", "HELP signal shown", "danger", True)
    if mode == "workplace":
        out["persons"] = sum(1 for o in (out["objects"] or []) if o["label"] == "person")
        if sess.frame_i % 3 == 1 or "ppe" not in sess.cache: sess.cache["ppe"] = detect.ppe(frame)
        raw = sess.cache["ppe"]; out["ppe_model"] = raw is not None
        out["ppe_raw"] = raw; out["ppe"] = detect.ppe_status(raw, sess.cfg["required"]) if raw is not None else None
        if out["ppe"]:
            miss = [k for k, v in out["ppe"].items() if v == "missing"]
            if miss: _alert(sess, alerts, "ppe", "Missing PPE: " + ", ".join(miss), "danger", True)
        if not out["person"]: state = "no_worker"
        elif out["phone"]["in_use"]: state = "distracted"
        elif out["away_for_s"] > 3: state = "distracted"
        elif f.get("expression") == "drowsy" or f.get("expression") == "yawning": state = "fatigued"
        elif out["idle_s"] > 10: state = "idle"
        else: state = "working"
        out["state"] = state
        if state == "fatigued": _alert(sess, alerts, "fatigue", "Worker looks fatigued", "warn")
        if state == "idle" and out["idle_s"] > 30: _alert(sess, alerts, "idle", "No activity for 30 s", "warn")
    elif mode == "student":
        bad = (out["phone"]["in_use"] or out["away_for_s"] > 2 or f.get("eyes_closed") and f.get("perclos", 0) > .3 or not out["person"])
        target = 0 if out["phone"]["in_use"] else 15 if bad else 100
        sess.attn += (target - sess.attn) * 0.08
        out["attention"] = round(sess.attn)
        out["state"] = "attentive" if sess.attn >= 70 else "partially attentive" if sess.attn >= 40 else "distracted"
        if out["state"] == "distracted": _alert(sess, alerts, "inattentive", "Student is distracted", "warn")
    elif mode in ("gesture", "care"):
        if out.get("help_signal"): _alert(sess, alerts, "help", "HELP signal shown", "danger", True)
        if out.get("fall"): _alert(sess, alerts, "fall", "Possible fall detected", "danger", True)
    elif mode == "advanced":
        img, gray = bodyscan.thermal_view(frame, mask)
        out["thermal"] = img; out["organs"] = bodyscan.organ_map(out.get("pose"), out["w"], out["h"], gray)
    out["alerts"] = alerts
    return out
