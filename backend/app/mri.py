"""Brain-MRI analysis for UPLOADED scans: dotted brain contour + bright-region candidates, a
sequence-type heuristic, and an optional trained classifier (models/mri_classifier.pt, see train_mri.py).
Research/education aid only - NOT a diagnostic device."""
import base64
import numpy as np, cv2
from . import config
_model = None; _tried = False

def _load():
    global _model, _tried
    if _tried: return _model
    _tried = True
    try:
        import torch, torchvision
        if config.MRI_MODEL.exists():
            ck = torch.load(config.MRI_MODEL, map_location="cpu")
            m = getattr(torchvision.models, ck["arch"])(weights=None)
            m.fc = torch.nn.Linear(m.fc.in_features, len(ck["classes"])); m.load_state_dict(ck["state"]); m.eval()
            _model = (m, ck["classes"])
    except Exception:
        _model = None
    return _model

def classify(gray):
    mm = _load()
    if not mm: return None
    import torch
    m, classes = mm
    x = cv2.resize(gray, (224, 224)).astype(np.float32) / 255
    x = np.stack([x] * 3); x = (x - np.array([.485, .456, .406])[:, None, None]) / np.array([.229, .224, .225])[:, None, None]
    with torch.no_grad(): p = torch.softmax(m(torch.tensor(x[None], dtype=torch.float32)), 1)[0].tolist()
    return {"label": classes[int(np.argmax(p))], "probs": {c: round(v, 3) for c, v in zip(classes, p)}}

def dotted(img, cnt, color, step=7, r=2):
    pts = cnt.reshape(-1, 2)
    for x, y in pts[::step]: cv2.circle(img, (int(x), int(y)), r, color, -1, cv2.LINE_AA)

def analyze(data: bytes):
    gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None: raise ValueError("Not a readable image")
    s = 512 / max(gray.shape); gray = cv2.resize(gray, None, fx=s, fy=s)
    blur = cv2.GaussianBlur(gray, (5, 5), 0); _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    over = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR); notes = []
    if not cnts: raise ValueError("No head/brain region found - is this an MRI slice?")
    c = max(cnts, key=cv2.contourArea); area = cv2.contourArea(c) / gray.size
    if area < 0.05: notes.append("Detected region is very small; result may be unreliable.")
    dotted(over, c, (255, 220, 0), 7, 2)
    mask = np.zeros_like(gray); cv2.drawContours(mask, [c], -1, 255, -1)
    inside = gray[mask > 0]; seq = "unknown"; cand = 0
    if inside.size:
        hi = np.percentile(inside, 96); lo = np.percentile(inside, 8); med = np.median(inside)
        bright = ((gray > hi) & (mask > 0)).astype(np.uint8) * 255
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        bc, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for b in bc:
            if cv2.contourArea(b) > 40: dotted(over, b, (255, 0, 200), 4, 2); cand += 1
        dark = ((gray < lo) & (mask > 0)).astype(np.uint8) * 255
        dc, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for d in dc:
            if cv2.contourArea(d) > 60: dotted(over, d, (0, 200, 255), 5, 1)
        seq = "T2/FLAIR-like (fluid bright)" if (hi - med) / (med + 1) > 1.1 and np.mean(inside > med * 1.6) > .08 else "T1-like (fluid dark)"
        notes.append("Sequence type is a crude intensity heuristic (low confidence).")
    ok, buf = cv2.imencode(".png", over)
    return {"overlay": "data:image/png;base64," + base64.b64encode(buf).decode(), "sequence_guess": seq,
            "bright_candidate_regions": cand, "brain_area_fraction": round(float(area), 3),
            "classification": classify(gray), "notes": notes + ["Not a medical diagnosis. Have a radiologist review the scan."]}
