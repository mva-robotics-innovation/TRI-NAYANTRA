
"""Computer-vision core: MediaPipe Holistic -> face / pose / hands,
expression, head-pose attention, gestures, fall check and rPPG.

Memory optimizations:
- Lazy Holistic model initialization.
- Model complexity 0.
- Segmentation disabled.
- Reduced per-session history buffers.
- Explicit model cleanup.
"""

import time
import math
import collections

import numpy as np
import cv2

try:
    import mediapipe as mp
except Exception:
    mp = None


# --------------------------------------------------
# Detection thresholds
# --------------------------------------------------

EAR_CLOSED = 0.19
MAR_YAWN = 0.55
SMILE_T = 0.018
BROW_RAISE = 0.105

YAW_AWAY = 0.17
PITCH_DOWN = 0.66

MOTION_T = 0.004

L_EYE = [33, 160, 158, 133, 153, 144]
R_EYE = [362, 385, 387, 263, 373, 380]


# --------------------------------------------------
# Geometry helpers
# --------------------------------------------------

def _d(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1],
    )


def _ear(p, idx):
    q = [p[i] for i in idx]

    return (
        _d(q[1], q[5]) + _d(q[2], q[4])
    ) / (
        2 * _d(q[0], q[3]) + 1e-6
    )


# --------------------------------------------------
# Per-connection session
# --------------------------------------------------

class Session:
    """Per-connection state with lazy lightweight MediaPipe initialization."""

    def __init__(self, mode):
        self.mode = mode
        self.t0 = time.time()

        self.cfg = {
            "required": [
                "helmet",
                "gloves",
                "shoes",
            ]
        }

        # Smaller history buffers to reduce memory.
        self.ear_hist = collections.deque(maxlen=180)
        self.blinks = collections.deque(maxlen=60)
        self._was_closed = False

        self.rppg = collections.deque(maxlen=300)

        self.motion = 0.0
        self.last_motion = time.time()
        self.prev_wrists = None

        self.away_since = None
        self.fall_since = None

        self.attn = 100.0

        self.frame_i = 0
        self.cache = {}
        self.holistic = None

        self.last_event = {}
        self.closed = False

    def holo(self):
        """Initialize the MediaPipe model only when camera processing starts."""

        if self.closed:
            return None

        if (
            self.holistic is None
            and mp is not None
            and hasattr(mp, "solutions")
        ):
            self.holistic = mp.solutions.holistic.Holistic(
                model_complexity=0,
                enable_segmentation=False,
                refine_face_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        return self.holistic

    def close(self):
        """Release the MediaPipe model when the session ends."""

        self.closed = True

        if self.holistic is not None:
            try:
                self.holistic.close()
            finally:
                self.holistic = None


# --------------------------------------------------
# Remote PPG heart-rate estimation
# --------------------------------------------------

def rppg_bpm(buf):
    """Experimental camera-based pulse estimate."""

    if len(buf) < 60:
        return None, 0.0

    ts = np.array(
        [b[0] for b in buf]
    )

    v = np.array(
        [b[1] for b in buf]
    )

    if ts[-1] - ts[0] < 8:
        return None, 0.0

    fs = 30.0

    t = np.arange(
        ts[0],
        ts[-1],
        1 / fs,
    )

    x = np.interp(t, ts, v)

    window = np.ones(int(fs)) / fs

    x = x - np.convolve(
        x,
        window,
        "same",
    )

    x = (
        x - x.mean()
    ) / (
        x.std() + 1e-6
    )

    f = np.fft.rfftfreq(
        len(x),
        1 / fs,
    )

    p = np.abs(
        np.fft.rfft(
            x * np.hanning(len(x))
        )
    ) ** 2

    band = (
        (f >= 0.7)
        & (f <= 3.0)
    )

    if not band.any():
        return None, 0.0

    band_power = p[band]

    i = int(
        np.argmax(band_power)
    )

    snr = (
        band_power[i]
        / (band_power.mean() + 1e-9)
    )

    bpm = float(
        f[band][i] * 60
    )

    confidence = float(
        min(1.0, snr / 12)
    )

    return bpm, confidence


# --------------------------------------------------
# Hand gesture recognition
# --------------------------------------------------

def finger_states(h):
    """Estimate extended fingers from landmark geometry."""

    w = h[0]
    st = {}

    for name, tip, pip in (
        ("index", 8, 6),
        ("middle", 12, 10),
        ("ring", 16, 14),
        ("pinky", 20, 18),
    ):
        st[name] = bool(
            _d(h[tip], w)
            > _d(h[pip], w) * 1.08
        )

    st["thumb"] = bool(
        _d(h[4], h[17])
        > _d(h[3], h[17]) * 1.05
        and _d(h[4], h[5]) > 0.07
    )

    return st


SIGNS = {
    (1, 1, 1, 1, 1): "HELLO",
    (0, 0, 0, 0, 0): "STOP",
    (0, 1, 0, 0, 0): "I_NEED",
    (0, 1, 1, 0, 0): "WATER",
    (0, 1, 1, 1, 0): "FOOD",
    (0, 0, 0, 0, 1): "WASHROOM",
    (1, 1, 0, 0, 1): "THANK_YOU",
    (1, 0, 0, 0, 0): "YES",
    (0, 1, 1, 1, 1): "PAIN",
}


def classify_hand(h):
    states = finger_states(h)

    key = tuple(
        int(states[k])
        for k in (
            "thumb",
            "index",
            "middle",
            "ring",
            "pinky",
        )
    )

    token = SIGNS.get(key)

    if (
        key == (1, 0, 0, 0, 0)
        and h[4][1] > h[0][1]
    ):
        token = "NO"

    return token, states


# --------------------------------------------------
# Main frame analysis
# --------------------------------------------------

def analyze(sess, bgr):
    """Analyze one frame and return JSON-compatible results.

    The 'mask' key is retained for compatibility with the existing
    pipeline. It will be None because segmentation is disabled.
    """

    h, w = bgr.shape[:2]

    out = {
        "w": w,
        "h": h,
        "person": False,
        "t": time.time(),
    }

    holo = sess.holo()

    if holo is None:
        out["error"] = (
            "MediaPipe legacy solutions API unavailable. "
            "Check the installed MediaPipe version."
        )
        out["mask"] = None
        return out

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )

    res = holo.process(rgb)

    now = time.time()
    sess.frame_i += 1

    # Segmentation is disabled in the memory-optimized configuration.
    out["mask"] = None

    pose = (
        res.pose_landmarks.landmark
        if res.pose_landmarks
        else None
    )

    face = (
        res.face_landmarks.landmark
        if res.face_landmarks
        else None
    )

    out["person"] = (
        pose is not None
        or face is not None
    )

    # --------------------------------------------------
    # Pose and posture
    # --------------------------------------------------

    if pose:
        P = [
            (landmark.x, landmark.y)
            for landmark in pose
        ]

        vis = [
            landmark.visibility
            for landmark in pose
        ]

        out["pose"] = [
            [
                round(x, 3),
                round(y, 3),
                round(v, 2),
            ]
            for (x, y), v in zip(P, vis)
        ]

        sh_l = P[11]
        sh_r = P[12]

        hp_l = P[23]
        hp_r = P[24]

        sh_mid = (
            (sh_l[0] + sh_r[0]) / 2,
            (sh_l[1] + sh_r[1]) / 2,
        )

        hp_mid = (
            (hp_l[0] + hp_r[0]) / 2,
            (hp_l[1] + hp_r[1]) / 2,
        )

        tilt = math.degrees(
            math.atan2(
                (sh_l[1] - sh_r[1]) * h,
                abs(sh_l[0] - sh_r[0]) * w,
            )
        )

        shoulder_width = (
            _d(sh_l, sh_r) + 1e-6
        )

        head_drop = (
            sh_mid[1] - P[0][1]
        ) / shoulder_width

        out["posture"] = {
            "shoulder_tilt_deg": round(tilt, 1),
            "head_height_ratio": round(head_drop, 2),
            "slouching": bool(head_drop < 0.55),
        }

        torso = math.degrees(
            math.atan2(
                abs(hp_mid[0] - sh_mid[0]) * w,
                abs(hp_mid[1] - sh_mid[1]) * h + 1e-6,
            )
        )

        lying = (
            torso > 60
            and vis[11] > 0.5
            and vis[23] > 0.4
        )

        sess.fall_since = (
            sess.fall_since or now
        ) if lying else None

        out["fall"] = bool(
            sess.fall_since
            and now - sess.fall_since > 1.5
        )

        wrists = np.array([
            P[15],
            P[16],
        ])

        if sess.prev_wrists is not None:
            movement = float(
                np.mean(
                    np.linalg.norm(
                        wrists - sess.prev_wrists,
                        axis=1,
                    )
                )
            )

            sess.motion = (
                0.7 * sess.motion
                + 0.3 * movement
            )

            if sess.motion > MOTION_T:
                sess.last_motion = now

        sess.prev_wrists = wrists

    out["idle_s"] = round(
        now - sess.last_motion,
        1,
    )

    out["motion"] = round(
        sess.motion,
        4,
    )

    # --------------------------------------------------
    # Face analysis
    # --------------------------------------------------

    if face:
        F = [
            (landmark.x, landmark.y)
            for landmark in face
        ]

        fw = (
            _d(F[234], F[454]) + 1e-6
        )

        ear = (
            _ear(F, L_EYE)
            + _ear(F, R_EYE)
        ) / 2

        mar = (
            _d(F[13], F[14])
            / (
                _d(F[61], F[291]) + 1e-6
            )
        )

        closed = ear < EAR_CLOSED

        sess.ear_hist.append(
            (now, closed)
        )

        if closed and not sess._was_closed:
            sess.blinks.append(now)

        sess._was_closed = closed

        recent = [
            closed_value
            for timestamp, closed_value
            in sess.ear_hist
            if now - timestamp < 60
        ]

        perclos = (
            sum(recent)
            / max(1, len(recent))
        )

        blink_pm = sum(
            1
            for timestamp in sess.blinks
            if now - timestamp < 60
        )

        smile = (
            (
                F[13][1] + F[14][1]
            ) / 2
            - (
                F[61][1] + F[291][1]
            ) / 2
        ) / fw

        brow = (
            _d(F[105], F[159])
            + _d(F[334], F[386])
        ) / 2 / fw

        ratio = (
            F[1][0] - F[234][0]
        ) / (
            F[454][0] - F[234][0] + 1e-6
        )

        yaw = ratio - 0.5

        eye_mid = (
            F[33][1] + F[263][1]
        ) / 2

        pitch = (
            F[1][1] - eye_mid
        ) / (
            F[13][1] - eye_mid + 1e-6
        )

        away = abs(yaw) > YAW_AWAY
        down = pitch > PITCH_DOWN

        if mar > 0.7:
            expression = "yawning"

        elif mar > 0.45 and brow > BROW_RAISE:
            expression = "surprised"

        elif smile > SMILE_T:
            expression = "happy"

        elif (
            perclos > 0.35
            or (closed and perclos > 0.2)
        ):
            expression = "drowsy"

        elif brow < 0.07 and mar < 0.1:
            expression = "stressed/frowning"

        else:
            expression = "neutral"

        out["face"] = {
            "ear": round(ear, 3),
            "mar": round(mar, 3),
            "expression": expression,
            "yaw": round(yaw, 2),
            "pitch": round(pitch, 2),
            "looking_away": bool(away),
            "looking_down": bool(down),
            "eyes_closed": bool(closed),
            "perclos": round(perclos, 2),
            "blinks_per_min": blink_pm,
        }

        out["face_pts"] = [
            [
                round(F[i][0], 3),
                round(F[i][1], 3),
            ]
            for i in range(0, 468, 3)
        ]

        xs = [point[0] for point in F]
        ys = [point[1] for point in F]

        out["face_box"] = [
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        ]

        sess.away_since = (
            sess.away_since or now
        ) if (away or down) else None

        x1 = int(F[109][0] * w)
        x2 = int(F[338][0] * w)

        y1 = int(F[10][1] * h)
        y2 = int(F[105][1] * h)

        if (
            x2 - x1 > 8
            and y2 - y1 > 4
            and x1 >= 0
            and y1 >= 0
            and x2 <= w
            and y2 <= h
        ):
            roi = bgr[y1:y2, x1:x2]

            if roi.size:
                sess.rppg.append(
                    (
                        now,
                        float(roi[..., 1].mean()),
                    )
                )

        bpm, confidence = rppg_bpm(
            sess.rppg
        )

        warm = (
            max(
                0,
                round(
                    8 - (
                        sess.rppg[-1][0]
                        - sess.rppg[0][0]
                    )
                ),
            )
            if sess.rppg
            else 8
        )

        out["hr"] = {
            "bpm": round(bpm) if bpm else None,
            "confidence": round(confidence, 2),
            "warming_up_s": warm,
        }

    else:
        sess.away_since = None

    out["away_for_s"] = (
        round(now - sess.away_since, 1)
        if sess.away_since
        else 0
    )

    # --------------------------------------------------
    # Hand landmark and gesture analysis
    # --------------------------------------------------

    hands = []

    for side, hand_landmarks in (
        ("left", res.left_hand_landmarks),
        ("right", res.right_hand_landmarks),
    ):

        if hand_landmarks:
            H = [
                (
                    landmark.x,
                    landmark.y,
                    landmark.z,
                )
                for landmark in hand_landmarks.landmark
            ]

            token, states = classify_hand(H)

            raised = bool(
                pose
                and H[0][1] < out["pose"][11][1]
            )

            hands.append({
                "side": side,
                "pts": [
                    [
                        round(x, 3),
                        round(y, 3),
                    ]
                    for x, y, _ in H
                ],
                "gesture": token,
                "fingers": states,
                "raised": raised,
            })

    out["hands"] = hands

    out["help_signal"] = bool(
        len(hands) == 2
        and all(
            hand["raised"]
            and hand["gesture"] == "HELLO"
            for hand in hands
        )
    )

    return out
