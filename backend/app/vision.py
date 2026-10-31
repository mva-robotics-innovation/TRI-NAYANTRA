
"""
TRI-NAYANTRA — MediaPipe Vision Engine

Responsibilities:
- Human presence and pose estimation
- Face landmarks and facial expression indicators
- Eye closure, blink and PERCLOS estimation
- Head orientation and attention indicators
- Hand landmarks and approximate gesture classification
- Possible fall and movement heuristics
- Experimental camera-based rPPG estimate

YOLO/PPE inference is intentionally NOT implemented here.
It runs through the separate detection service via detect.py.

Important:
These outputs are computer-vision estimates, not medical diagnoses.
"""

import time
import math
import logging
from collections import deque

import cv2
import numpy as np


logger = logging.getLogger("tri-nayantra.vision")


# ==================================================
# OPTIONAL MEDIAPIPE IMPORT
# ==================================================

try:
    import mediapipe as mp
except Exception as exc:
    mp = None
    logger.warning("MediaPipe unavailable: %s", exc)


# ==================================================
# CONFIGURATION
# ==================================================

EAR_CLOSED = 0.19
MAR_YAWN = 0.55

SMILE_T = 0.018
YAW_AWAY = 0.17
PITCH_DOWN = 0.66

MOTION_T = 0.004
MIN_VISIBILITY = 0.35

FACE_HISTORY = 180
BLINK_HISTORY = 60
RPPG_HISTORY = 300

CAMERA_FPS_ESTIMATE = 30.0


# ==================================================
# LANDMARK INDICES
# ==================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH = {
    "left": 61,
    "right": 291,
    "top": 13,
    "bottom": 14,
}

POSE = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


# ==================================================
# BASIC GEOMETRY
# ==================================================

def _distance(a, b):
    return math.hypot(
        float(a[0]) - float(b[0]),
        float(a[1]) - float(b[1]),
    )


def _angle(a, b, c):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba)
        * np.linalg.norm(bc)
    )

    if denominator < 1e-7:
        return None

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def _safe_float(value, default=0.0):
    try:
        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except (TypeError, ValueError, OverflowError):
        return default


def _landmark_xy(landmark):
    return (
        _safe_float(landmark.x),
        _safe_float(landmark.y),
    )


def _visible(landmark, threshold=MIN_VISIBILITY):
    return (
        _safe_float(
            getattr(landmark, "visibility", 1.0),
            0.0,
        ) >= threshold
    )


# ==================================================
# FACE METRICS
# ==================================================

def _ear(points):
    """Eye Aspect Ratio."""

    if len(points) != 6:
        return 0.0

    vertical_1 = _distance(points[1], points[5])
    vertical_2 = _distance(points[2], points[4])
    horizontal = _distance(points[0], points[3])

    if horizontal < 1e-7:
        return 0.0

    return (
        vertical_1 + vertical_2
    ) / (
        2.0 * horizontal
    )


def _face_ear(face_landmarks):
    points = face_landmarks.landmark

    left = [_landmark_xy(points[i]) for i in LEFT_EYE]
    right = [_landmark_xy(points[i]) for i in RIGHT_EYE]

    return (_ear(left) + _ear(right)) / 2.0


def _mar(face_landmarks):
    points = face_landmarks.landmark

    left = _landmark_xy(points[MOUTH["left"]])
    right = _landmark_xy(points[MOUTH["right"]])
    top = _landmark_xy(points[MOUTH["top"]])
    bottom = _landmark_xy(points[MOUTH["bottom"]])

    width = _distance(left, right)

    if width < 1e-7:
        return 0.0

    return _distance(top, bottom) / width


def _face_box(face_landmarks):
    if face_landmarks is None:
        return None

    points = face_landmarks.landmark

    xs = [
        _safe_float(p.x, float("nan"))
        for p in points
    ]

    ys = [
        _safe_float(p.y, float("nan"))
        for p in points
    ]

    xs = [v for v in xs if math.isfinite(v)]
    ys = [v for v in ys if math.isfinite(v)]

    if not xs or not ys:
        return None

    return [
        max(0.0, min(xs)),
        max(0.0, min(ys)),
        min(1.0, max(xs)),
        min(1.0, max(ys)),
    ]


def _head_orientation(face_landmarks):
    if face_landmarks is None:
        return None

    try:
        points = face_landmarks.landmark

        left_eye = points[33]
        right_eye = points[263]
        nose = points[1]
        forehead = points[10]
        chin = points[152]

        eye_mid_x = (
            left_eye.x + right_eye.x
        ) / 2.0

        eye_width = abs(
            right_eye.x - left_eye.x
        )

        if eye_width < 1e-6:
            return None

        yaw = (
            nose.x - eye_mid_x
        ) / eye_width

        face_height = abs(
            chin.y - forehead.y
        )

        if face_height < 1e-6:
            return None

        pitch = (
            nose.y - forehead.y
        ) / face_height

        return {
            "yaw": round(float(yaw), 3),
            "pitch": round(float(pitch), 3),
        }

    except Exception:
        return None


def _forehead_signal(frame, face_landmarks):
    """Extract a basic green-channel signal from the forehead ROI."""

    if face_landmarks is None:
        return None

    h, w = frame.shape[:2]

    try:
        forehead = face_landmarks.landmark[10]

        cx = int(forehead.x * w)
        cy = int(forehead.y * h)

        x1 = max(0, cx - int(w * 0.025))
        x2 = min(w, cx + int(w * 0.025))

        y1 = max(0, cy - int(h * 0.015))
        y2 = min(h, cy + int(h * 0.025))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        value = float(np.mean(roi[:, :, 1]))

        return value if math.isfinite(value) else None

    except Exception:
        return None


# ==================================================
# POSE HELPERS
# ==================================================

def _pose_data(pose_landmarks):
    if pose_landmarks is None:
        return None

    result = []

    for landmark in pose_landmarks.landmark:
        result.append([
            _safe_float(landmark.x),
            _safe_float(landmark.y),
            _safe_float(
                getattr(landmark, "visibility", 1.0)
            ),
        ])

    return result


def _pose_point(pose, index):
    if pose is None or index >= len(pose):
        return None

    x, y, visibility = pose[index]

    if visibility < MIN_VISIBILITY:
        return None

    return float(x), float(y)


def _pose_angle(pose, a, b, c):
    p1 = _pose_point(pose, a)
    p2 = _pose_point(pose, b)
    p3 = _pose_point(pose, c)

    if p1 is None or p2 is None or p3 is None:
        return None

    return _angle(p1, p2, p3)


# ==================================================
# SESSION
# ==================================================

class Session:
    """
    State belonging to one camera WebSocket session.

    Create one Session per WebSocket connection.
    Do not share a Session between camera clients.
    """

    def __init__(self, mode="workplace"):
        self.mode = mode
        self.t0 = time.time()

        self.cfg = {
            "required": {
                "helmet": True,
                "gloves": True,
                "shoes": True,
            }
        }

        self.ear_hist = deque(maxlen=FACE_HISTORY)
        self.blinks = deque(maxlen=BLINK_HISTORY)
        self.rppg = deque(maxlen=RPPG_HISTORY)

        self.motion = 0.0
        self.last_motion = time.time()

        self.prev_wrists = None
        self.prev_nose = None

        self.away_since = None
        self.fall_since = None

        self.attn = 100.0

        self.frame_i = 0
        self.cache = {}

        self.holistic = None
        self.last_event = {}

        self.closed = False

    def holo(self):
        """Lazily initialize MediaPipe Holistic."""

        if self.closed or mp is None:
            return None

        if self.holistic is not None:
            return self.holistic

        try:
            self.holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                enable_segmentation=False,
                refine_face_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            return self.holistic

        except Exception:
            logger.exception(
                "Could not initialize MediaPipe Holistic"
            )

            self.holistic = None
            return None

    def close(self):
        """Release MediaPipe resources and session caches."""

        if self.closed:
            return

        self.closed = True

        if self.holistic is not None:
            try:
                self.holistic.close()

            except Exception:
                logger.exception(
                    "MediaPipe cleanup failed"
                )

            finally:
                self.holistic = None

        self.cache.clear()
        self.ear_hist.clear()
        self.blinks.clear()
        self.rppg.clear()


# ==================================================
# EXPERIMENTAL rPPG
# ==================================================

def rppg_bpm(history):
    """
    Experimental camera-based pulse estimate.

    Requires a sufficiently long signal. This is not a
    medical-grade heart-rate measurement.
    """

    if len(history) < 120:
        return None

    try:
        values = np.asarray(
            history,
            dtype=np.float64,
        )

        if (
            values.ndim != 1
            or not np.all(np.isfinite(values))
        ):
            return None

        values -= np.mean(values)

        if np.std(values) < 1e-5:
            return None

        source_t = (
            np.arange(len(values), dtype=np.float64)
            / CAMERA_FPS_ESTIMATE
        )

        target_t = np.arange(
            0,
            source_t[-1],
            1.0 / CAMERA_FPS_ESTIMATE,
        )

        if len(target_t) < 100:
            return None

        values = np.interp(
            target_t,
            source_t,
            values,
        )

        values -= np.mean(values)

        window = np.hanning(len(values))

        spectrum = np.abs(
            np.fft.rfft(values * window)
        )

        frequencies = np.fft.rfftfreq(
            len(values),
            d=1.0 / CAMERA_FPS_ESTIMATE,
        )

        valid = (
            (frequencies >= 0.7)
            & (frequencies <= 3.0)
        )

        if not np.any(valid):
            return None

        selected = spectrum[valid]
        selected_freq = frequencies[valid]

        peak = selected_freq[
            int(np.argmax(selected))
        ]

        bpm = float(peak * 60.0)

        if not 42 <= bpm <= 180:
            return None

        return round(bpm, 1)

    except Exception:
        logger.exception("rPPG calculation failed")
        return None


# ==================================================
# HAND GESTURES
# ==================================================

def finger_states(hand_landmarks):
    """
    Approximate finger extension.

    The vertical comparison is most reliable for
    upright hands facing the camera.
    """

    points = hand_landmarks.landmark

    finger_pairs = {
        "index": (8, 6),
        "middle": (12, 10),
        "ring": (16, 14),
        "pinky": (20, 18),
    }

    states = {}

    for name, (tip, joint) in finger_pairs.items():
        states[name] = (
            points[tip].y < points[joint].y
        )

    thumb_tip = points[4]
    thumb_joint = points[3]
    reference = points[5]

    states["thumb"] = (
        abs(thumb_tip.x - reference.x)
        > abs(thumb_joint.x - reference.x)
    )

    return states


def classify_hand(hand_landmarks):
    """Classify a limited set of static hand poses."""

    states = finger_states(hand_landmarks)

    count = sum(bool(v) for v in states.values())

    if count == 0:
        return "FIST"

    if count == 5:
        return "OPEN_PALM"

    if (
        states["index"]
        and states["middle"]
        and not states["ring"]
        and not states["pinky"]
    ):
        return "PEACE"

    if (
        states["thumb"]
        and not states["index"]
        and not states["middle"]
        and not states["ring"]
        and not states["pinky"]
    ):
        return "THUMBS_UP"

    if (
        states["index"]
        and not states["middle"]
        and not states["ring"]
        and not states["pinky"]
    ):
        return "POINT"

    if (
        states["thumb"]
        and states["index"]
        and not states["middle"]
        and not states["ring"]
        and not states["pinky"]
    ):
        return "PINCH"

    return "UNKNOWN"


# ==================================================
# MOVEMENT
# ==================================================

def _update_movement(sess, pose):
    left = _pose_point(pose, POSE["left_wrist"])
    right = _pose_point(pose, POSE["right_wrist"])

    current = (left, right)
    movement = 0.0

    if sess.prev_wrists is not None:
        old_left, old_right = sess.prev_wrists

        if left is not None and old_left is not None:
            movement += _distance(left, old_left)

        if right is not None and old_right is not None:
            movement += _distance(right, old_right)

    sess.prev_wrists = current

    sess.motion = (
        0.75 * sess.motion
        + 0.25 * movement
    )

    if sess.motion > MOTION_T:
        sess.last_motion = time.time()

    idle_s = max(
        0.0,
        time.time() - sess.last_motion,
    )

    return round(sess.motion, 5), round(idle_s, 1)


# ==================================================
# FALL HEURISTIC
# ==================================================

def _fall_heuristic(sess, pose, posture, tilt):
    """
    Basic possible-fall heuristic.

    This is not a validated fall-detection model.
    """

    shoulder_left = _pose_point(
        pose,
        POSE["left_shoulder"],
    )

    shoulder_right = _pose_point(
        pose,
        POSE["right_shoulder"],
    )

    hip_left = _pose_point(
        pose,
        POSE["left_hip"],
    )

    hip_right = _pose_point(
        pose,
        POSE["right_hip"],
    )

    if not all([
        shoulder_left,
        shoulder_right,
        hip_left,
        hip_right,
    ]):
        sess.fall_since = None
        return False

    shoulder_mid = (
        (shoulder_left[0] + shoulder_right[0]) / 2.0,
        (shoulder_left[1] + shoulder_right[1]) / 2.0,
    )

    hip_mid = (
        (hip_left[0] + hip_right[0]) / 2.0,
        (hip_left[1] + hip_right[1]) / 2.0,
    )

    torso_dx = abs(
        shoulder_mid[0] - hip_mid[0]
    )

    torso_dy = abs(
        shoulder_mid[1] - hip_mid[1]
    )

    horizontal_torso = torso_dx > torso_dy

    low_posture = (
        posture == "lying"
        or (
            posture == "bent"
            and tilt > 55
        )
    )

    candidate = horizontal_torso or low_posture
    now = time.time()

    if candidate:
        if sess.fall_since is None:
            sess.fall_since = now

        return (
            now - sess.fall_since
        ) >= 0.8

    sess.fall_since = None
    return False


# ==================================================
# POSTURE ANALYSIS
# ==================================================

def _analyze_posture(sess, pose, out):
    if pose is None:
        sess.fall_since = None
        return

    left_shoulder = _pose_point(
        pose,
        POSE["left_shoulder"],
    )

    right_shoulder = _pose_point(
        pose,
        POSE["right_shoulder"],
    )

    left_hip = _pose_point(
        pose,
        POSE["left_hip"],
    )

    right_hip = _pose_point(
        pose,
        POSE["right_hip"],
    )

    nose = _pose_point(
        pose,
        POSE["nose"],
    )

    if left_shoulder and right_shoulder:
        shoulder_mid = (
            (left_shoulder[0] + right_shoulder[0]) / 2.0,
            (left_shoulder[1] + right_shoulder[1]) / 2.0,
        )

        shoulder_tilt = math.degrees(
            math.atan2(
                right_shoulder[1] - left_shoulder[1],
                right_shoulder[0] - left_shoulder[0],
            )
        )

        out["tilt"] = round(
            abs(shoulder_tilt),
            1,
        )

        if left_hip and right_hip:
            hip_mid = (
                (left_hip[0] + right_hip[0]) / 2.0,
                (left_hip[1] + right_hip[1]) / 2.0,
            )

            torso_dx = abs(
                shoulder_mid[0] - hip_mid[0]
            )

            torso_dy = abs(
                shoulder_mid[1] - hip_mid[1]
            )

            if (
                torso_dx > torso_dy
                and torso_dx > 0.08
            ):
                out["posture"] = "lying"

            elif (
                torso_dy > 0.12
                and shoulder_mid[1] > hip_mid[1]
            ):
                out["posture"] = "upright"

            elif torso_dy > 0.08:
                out["posture"] = "bent"

        if nose is not None:
            out["head_drop"] = (
                nose[1] > shoulder_mid[1] + 0.12
            )

    out["fall"] = _fall_heuristic(
        sess,
        pose,
        out["posture"],
        out["tilt"],
    )


# ==================================================
# FACE ANALYSIS
# ==================================================

def _analyze_face(sess, bgr, face_landmarks, out, now):
    if face_landmarks is None:
        sess.away_since = None

        out["face"] = {
            "visible": False,
            "eyes_closed": False,
            "expression": "unavailable",
            "perclos": None,
            "blink_min": None,
            "heart_rate_estimate": None,
            "heart_rate_type": "unavailable",
        }

        out["face_box"] = None
        out["away_for_s"] = 0.0
        return

    ear = _face_ear(face_landmarks)
    mar = _mar(face_landmarks)

    sess.ear_hist.append(ear)

    eyes_closed = ear < EAR_CLOSED

    if len(sess.ear_hist) >= 2:
        previous = sess.ear_hist[-2]

        if previous >= EAR_CLOSED and eyes_closed:
            sess.blinks.append(now)

    closed_count = sum(
        value < EAR_CLOSED
        for value in sess.ear_hist
    )

    perclos = (
        closed_count / max(1, len(sess.ear_hist))
    )

    recent_blinks = [
        value
        for value in sess.blinks
        if now - value <= 60
    ]

    blink_rate = len(recent_blinks)
    yawning = mar > MAR_YAWN

    orientation = _head_orientation(face_landmarks)

    yaw = orientation["yaw"] if orientation else 0.0
    pitch = orientation["pitch"] if orientation else 0.0

    points = face_landmarks.landmark

    mouth_left = points[61]
    mouth_right = points[291]
    mouth_top = points[13]
    mouth_bottom = points[14]

    smile = (
        (mouth_left.y + mouth_right.y) / 2.0
        - (mouth_top.y + mouth_bottom.y) / 2.0
    )

    expression = "neutral"

    if eyes_closed and perclos > 0.3:
        expression = "drowsy"

    elif yawning:
        expression = "yawning"

    elif smile > SMILE_T:
        expression = "smiling"

    elif orientation and abs(yaw) > YAW_AWAY:
        expression = "looking_away"

    elif orientation and pitch > PITCH_DOWN:
        expression = "looking_down"

    looking_away = (
        orientation is not None
        and abs(yaw) > YAW_AWAY
    )

    if looking_away:
        if sess.away_since is None:
            sess.away_since = now
    else:
        sess.away_since = None

    away_for = (
        now - sess.away_since
        if sess.away_since is not None
        else 0.0
    )

    signal = _forehead_signal(
        bgr,
        face_landmarks,
    )

    if signal is not None:
        sess.rppg.append(signal)

    heart_rate = rppg_bpm(sess.rppg)

    out["face"] = {
        "visible": True,
        "ear": round(ear, 3),
        "mar": round(mar, 3),
        "eyes_closed": bool(eyes_closed),
        "perclos": round(perclos, 3),
        "blink_min": blink_rate,
        "yawning": bool(yawning),
        "expression": expression,
        "yaw": round(yaw, 3),
        "pitch": round(pitch, 3),
        "heart_rate_estimate": heart_rate,
        "heart_rate_type": (
            "experimental_camera_rppg"
            if heart_rate is not None
            else "unavailable"
        ),
    }

    out["face_box"] = _face_box(face_landmarks)
    out["away_for_s"] = round(away_for, 1)


# ==================================================
# HAND ANALYSIS
# ==================================================

def _analyze_hands(results):
    hands = []

    for side, hand_landmarks in (
        ("left", results.left_hand_landmarks),
        ("right", results.right_hand_landmarks),
    ):
        if hand_landmarks is None:
            continue

        try:
            states = finger_states(hand_landmarks)
            gesture = classify_hand(hand_landmarks)
            wrist = hand_landmarks.landmark[0]

            hands.append({
                "side": side,
                "gesture": gesture,
                "fingers": states,
                "x": round(float(wrist.x), 4),
                "y": round(float(wrist.y), 4),
            })

        except Exception:
            logger.exception(
                "Hand classification failed"
            )

    return hands


# ==================================================
# RAISED HAND / HELP SIGNAL
# ==================================================

def _help_signal(pose, hands):
    raised = []

    if pose is not None:
        left_shoulder = _pose_point(
            pose,
            POSE["left_shoulder"],
        )

        right_shoulder = _pose_point(
            pose,
            POSE["right_shoulder"],
        )

        left_wrist = _pose_point(
            pose,
            POSE["left_wrist"],
        )

        right_wrist = _pose_point(
            pose,
            POSE["right_wrist"],
        )

        left_raised = (
            left_shoulder is not None
            and left_wrist is not None
            and left_wrist[1] < left_shoulder[1] - 0.04
        )

        right_raised = (
            right_shoulder is not None
            and right_wrist is not None
            and right_wrist[1] < right_shoulder[1] - 0.04
        )

        raised = [
            bool(left_raised),
            bool(right_raised),
        ]

    open_palms = sum(
        1
        for hand in hands
        if hand["gesture"] == "OPEN_PALM"
    )

    help_signal = (
        len(raised) == 2
        and all(raised)
        and open_palms >= 1
    )

    return raised, bool(help_signal)


# ==================================================
# MAIN VISION ANALYSIS
# ==================================================

def analyze(sess, bgr):
    """
    Analyze one BGR camera frame.

    Returns JSON-compatible output and an optional
    internal mask. pipeline.py should remove `mask`
    before sending the result to the frontend.

    `pose` is included for advanced body-map processing.
    """

    if sess.closed:
        raise RuntimeError(
            "Vision session is already closed."
        )

    if (
        bgr is None
        or not isinstance(bgr, np.ndarray)
        or bgr.ndim != 3
        or bgr.shape[2] != 3
    ):
        raise ValueError(
            "Expected a valid BGR camera frame."
        )

    h, w = bgr.shape[:2]

    if h <= 0 or w <= 0:
        raise ValueError(
            "Invalid camera frame dimensions."
        )

    now = time.time()

    out = {
        "w": int(w),
        "h": int(h),
        "person": False,
        "t": round(now - sess.t0, 1),
        "mode": sess.mode,

        "face": {},
        "face_box": None,
        "hands": [],

        "pose": None,

        "posture": "unknown",
        "tilt": 0.0,
        "head_drop": False,

        "fall": False,
        "help_signal": False,
        "raised": [],

        "motion": 0.0,
        "idle_s": 0.0,
        "away_for_s": 0.0,

        "error": None,
        "mask": None,
    }

    # ----------------------------------------------
    # Initialize MediaPipe
    # ----------------------------------------------

    holistic = sess.holo()

    if holistic is None:
        out["error"] = (
            "MediaPipe is unavailable. "
            "Install a compatible mediapipe package."
        )
        return out

    # ----------------------------------------------
    # Process frame
    # ----------------------------------------------

    rgb = None

    try:
        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

        rgb.flags.writeable = False

        results = holistic.process(rgb)

    except Exception:
        logger.exception(
            "MediaPipe frame processing failed"
        )

        out["error"] = (
            "MediaPipe could not process the camera frame."
        )

        return out

    finally:
        if rgb is not None:
            try:
                rgb.flags.writeable = True
            except Exception:
                pass

    sess.frame_i += 1

    # ----------------------------------------------
    # Pose
    # ----------------------------------------------

    pose = _pose_data(
        results.pose_landmarks
    )

    # Important: pipeline.py advanced mode needs this.
    out["pose"] = pose

    out["person"] = (
        pose is not None
        and any(
            visibility >= MIN_VISIBILITY
            for _, _, visibility in pose
        )
    )

    # ----------------------------------------------
    # Movement
    # ----------------------------------------------

    motion, idle_s = _update_movement(
        sess,
        pose,
    )

    out["motion"] = motion
    out["idle_s"] = idle_s

    # ----------------------------------------------
    # Posture and possible fall
    # ----------------------------------------------

    _analyze_posture(
        sess,
        pose,
        out,
    )

    # ----------------------------------------------
    # Face
    # ----------------------------------------------

    _analyze_face(
        sess,
        bgr,
        results.face_landmarks,
        out,
        now,
    )

    # ----------------------------------------------
    # Hands
    # ----------------------------------------------

    hands = _analyze_hands(results)
    out["hands"] = hands

    # ----------------------------------------------
    # Raised hands / HELP signal
    # ----------------------------------------------

    raised, help_signal = _help_signal(
        pose,
        hands,
    )

    out["raised"] = raised
    out["help_signal"] = help_signal

    return out
