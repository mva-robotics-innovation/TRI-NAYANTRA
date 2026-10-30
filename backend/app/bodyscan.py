"""Thermal-style view + anatomical organ guide.
HONEST NOTE: an ordinary webcam cannot see inside the body or measure temperature. This module
(1) renders a *pseudo-thermal* colour map of the camera image, and
(2) places an *estimated anatomical atlas* over the detected pose.
With a real radiometric camera (FLIR Lepton, MLX90640, Seek) feed its frame to `thermal_view`
as `heat` and the warmth values become genuine surface temperatures."""
import math, base64
import numpy as np, cv2

def thermal_view(bgr, mask, heat=None, width=480):
    g = heat if heat is not None else cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), 9)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    col = cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)
    if mask is not None:
        m = cv2.GaussianBlur((mask > 0.5).astype(np.float32), (0, 0), 4)[..., None]
        col = (col * m + col * 0.18 * (1 - m)).astype(np.uint8)
    s = width / col.shape[1]; col = cv2.resize(col, (width, int(col.shape[0] * s)))
    ok, buf = cv2.imencode(".jpg", col, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode(), g

def organ_map(pose, w, h, heat_gray=None):
    """pose: list of [x,y,vis] normalised. Returns ellipses (normalised) with a relative warmth index."""
    if not pose or pose[11][2] < .4 or pose[12][2] < .4: return []
    px = lambda i: np.array([pose[i][0] * w, pose[i][1] * h])
    sl, sr = px(11), px(12); mid = (sl + sr) / 2; sw = np.linalg.norm(sl - sr) + 1e-6
    hip = (px(23) + px(24)) / 2 if pose[23][2] > .3 and pose[24][2] > .3 else mid + np.array([0, sw * 1.9])
    down = hip - mid; tl = np.linalg.norm(down) + 1e-6; dn = down / tl
    left = (sl - sr) / sw      # unit vector toward the person's left shoulder (landmark 11)
    ang = math.degrees(math.atan2(dn[1], dn[0])) - 90
    def at(a, b): return mid + left * a * sw / 2 + dn * b * tl
    ears = (px(7) + px(8)) / 2; hw = np.linalg.norm(px(7) - px(8)) + 1e-6
    items = [("Brain", ears + np.array([0, -hw * .25]), hw * .55, hw * .45, 0),
             ("Heart", at(.35, .24), sw * .11, tl * .11, ang), ("Lung (L)", at(.55, .2), sw * .12, tl * .2, ang),
             ("Lung (R)", at(-.55, .2), sw * .12, tl * .2, ang), ("Liver", at(-.45, .45), sw * .17, tl * .1, ang),
             ("Stomach", at(.4, .46), sw * .11, tl * .1, ang), ("Intestines", at(0, .72), sw * .26, tl * .15, ang)]
    out = []
    for n, c, rx, ry, a in items:
        warm = None
        if heat_gray is not None:
            m = np.zeros(heat_gray.shape[:2], np.uint8)
            cv2.ellipse(m, (int(c[0] * heat_gray.shape[1] / w), int(c[1] * heat_gray.shape[0] / h)),
                        (max(1, int(rx * heat_gray.shape[1] / w)), max(1, int(ry * heat_gray.shape[0] / h))), a, 0, 360, 255, -1)
            if m.any(): warm = int(heat_gray[m > 0].mean() / 255 * 100)
        out.append({"name": n, "cx": round(c[0] / w, 3), "cy": round(c[1] / h, 3),
                    "rx": round(rx / w, 3), "ry": round(ry / h, 3), "angle": round(a, 1), "warmth_idx": warm})
    return out
