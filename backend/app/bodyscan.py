
"""
TRI-NAYANTRA — Pseudo-Thermal Visualization and Anatomical Guide

IMPORTANT:
An ordinary webcam cannot see inside the human body or measure
actual body temperature.

This module provides:
1. A pseudo-thermal colour visualization of camera frames.
2. An estimated anatomical atlas positioned using MediaPipe pose landmarks.
3. A relative image-intensity index for visualization.

For actual surface-temperature measurements, provide calibrated
radiometric thermal-camera data, such as data from a compatible
FLIR, MLX90640 or Seek device.

The anatomical guide is educational and is NOT a medical diagnostic tool.
"""

import math
import base64

import numpy as np
import cv2


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DEFAULT_WIDTH = 480
JPEG_QUALITY = 60

MIN_LANDMARK_VISIBILITY = 0.4
MIN_HIP_VISIBILITY = 0.3


# --------------------------------------------------
# IMAGE VALIDATION
# --------------------------------------------------

def _validate_frame(bgr):
    """Validate an incoming BGR camera frame."""

    if bgr is None:
        raise ValueError("Camera frame is empty")

    if not isinstance(bgr, np.ndarray):
        raise TypeError("Camera frame must be a NumPy array")

    if bgr.size == 0:
        raise ValueError("Camera frame contains no image data")

    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR image")


def _prepare_heat(bgr, heat=None):
    """Prepare a grayscale image for pseudo-thermal rendering."""

    if heat is None:
        gray = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2GRAY
        )

        return cv2.GaussianBlur(
            gray,
            (0, 0),
            9
        )

    gray = np.asarray(heat)

    if gray.size == 0:
        raise ValueError("Thermal input is empty")

    if gray.ndim == 3:
        if gray.shape[2] == 3:
            gray = cv2.cvtColor(
                gray,
                cv2.COLOR_BGR2GRAY
            )
        elif gray.shape[2] == 1:
            gray = gray[:, :, 0]
        else:
            raise ValueError("Unsupported thermal image format")

    if gray.ndim != 2:
        raise ValueError("Thermal input must be a grayscale or colour image")

    if gray.shape[:2] != bgr.shape[:2]:
        gray = cv2.resize(
            gray,
            (bgr.shape[1], bgr.shape[0]),
            interpolation=cv2.INTER_AREA
        )

    return gray


# --------------------------------------------------
# PSEUDO-THERMAL VISUALIZATION
# --------------------------------------------------

def thermal_view(bgr, mask=None, heat=None, width=DEFAULT_WIDTH):
    """
    Generate a pseudo-thermal visualization.

    Args:
        bgr:
            Original BGR webcam frame.

        mask:
            Optional MediaPipe person segmentation mask.
            Expected values are approximately 0 to 1.

        heat:
            Optional radiometric thermal-camera frame.
            Without calibrated radiometric input, the output is
            only a visual intensity representation.

        width:
            Output image width.

    Returns:
        (data_uri, gray_image)

        data_uri:
            Base64 JPEG data URI for frontend display.

        gray_image:
            Grayscale intensity image for relative visualization.
    """

    _validate_frame(bgr)

    if not isinstance(width, int) or width <= 0:
        raise ValueError("Output width must be a positive integer")

    # Prepare the image used for colour mapping.
    gray = _prepare_heat(bgr, heat)

    # Normalize intensity to 0–255.
    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Apply an inferno-style colour map.
    colour = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_INFERNO
    )

    # Optional person segmentation.
    if mask is not None:

        mask = np.asarray(mask)

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        if mask.shape[:2] != bgr.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.float32),
                (bgr.shape[1], bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        person_mask = (
            mask > 0.5
        ).astype(np.float32)

        person_mask = cv2.GaussianBlur(
            person_mask,
            (0, 0),
            4
        )[..., None]

        # Keep the person visually prominent.
        colour = (
            colour * person_mask
            + colour * 0.18 * (1 - person_mask)
        ).astype(np.uint8)

    # Resize while preserving aspect ratio.
    scale = width / colour.shape[1]

    output_height = max(
        1,
        int(colour.shape[0] * scale)
    )

    colour = cv2.resize(
        colour,
        (width, output_height),
        interpolation=cv2.INTER_AREA
    )

    # Encode JPEG.
    success, buffer = cv2.imencode(
        ".jpg",
        colour,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            JPEG_QUALITY
        ]
    )

    if not success:
        raise RuntimeError(
            "Unable to encode pseudo-thermal image"
        )

    encoded = base64.b64encode(
        buffer
    ).decode("ascii")

    data_uri = (
        "data:image/jpeg;base64,"
        + encoded
    )

    return data_uri, gray


# --------------------------------------------------
# POSE HELPERS
# --------------------------------------------------

def _landmark_visible(pose, index, threshold=MIN_LANDMARK_VISIBILITY):
    """Check whether a pose landmark is sufficiently visible."""

    try:
        return (
            pose is not None
            and len(pose) > index
            and len(pose[index]) >= 3
            and float(pose[index][2]) >= threshold
        )
    except (TypeError, ValueError, IndexError):
        return False


def _point(pose, index, width, height):
    """Convert normalized pose coordinates to image coordinates."""

    return np.array(
        [
            float(pose[index][0]) * width,
            float(pose[index][1]) * height
        ],
        dtype=np.float32
    )


def _safe_mean(values):
    """Calculate a safe mean."""

    if values is None:
        return None

    if len(values) == 0:
        return None

    return float(np.mean(values))


# --------------------------------------------------
# ANATOMICAL ORGAN GUIDE
# --------------------------------------------------

def organ_map(pose, w, h, heat_gray=None):
    """
    Estimate anatomical guide positions from MediaPipe pose landmarks.

    Args:
        pose:
            List of [x, y, visibility] normalized landmarks.

        w:
            Original camera frame width.

        h:
            Original camera frame height.

        heat_gray:
            Optional grayscale intensity image.

    Returns:
        List of estimated anatomical guide positions.

    NOTE:
        These positions are illustrative overlays.
        They do not identify actual internal organs or abnormalities.
    """

    if not pose:
        return []

    if w <= 0 or h <= 0:
        return []

    # Require visible shoulders.
    if not (
        _landmark_visible(pose, 11)
        and _landmark_visible(pose, 12)
    ):
        return []

    try:
        # Shoulder landmarks.
        shoulder_left = _point(
            pose, 11, w, h
        )

        shoulder_right = _point(
            pose, 12, w, h
        )

        middle = (
            shoulder_left + shoulder_right
        ) / 2

        shoulder_width = (
            np.linalg.norm(
                shoulder_left - shoulder_right
            ) + 1e-6
        )

        # Hip landmarks.
        if (
            _landmark_visible(
                pose,
                23,
                MIN_HIP_VISIBILITY
            )
            and
            _landmark_visible(
                pose,
                24,
                MIN_HIP_VISIBILITY
            )
        ):

            hip_left = _point(
                pose, 23, w, h
            )

            hip_right = _point(
                pose, 24, w, h
            )

            hip = (
                hip_left + hip_right
            ) / 2

        else:
            # Estimate hip position if hips are not visible.
            hip = (
                middle
                + np.array(
                    [0, shoulder_width * 1.9],
                    dtype=np.float32
                )
            )

        # Body direction.
        down = hip - middle

        torso_length = (
            np.linalg.norm(down) + 1e-6
        )

        down_direction = (
            down / torso_length
        )

        # Direction toward the person's left shoulder.
        left_direction = (
            shoulder_left - shoulder_right
        ) / shoulder_width

        angle = (
            math.degrees(
                math.atan2(
                    down_direction[1],
                    down_direction[0]
                )
            ) - 90
        )

        def at(horizontal, vertical):
            """Return a point relative to the torso."""

            return (
                middle
                + left_direction
                * horizontal
                * shoulder_width
                / 2
                + down_direction
                * vertical
                * torso_length
            )

        # Head landmarks.
        if (
            _landmark_visible(pose, 7, 0.2)
            and
            _landmark_visible(pose, 8, 0.2)
        ):

            left_ear = _point(
                pose, 7, w, h
            )

            right_ear = _point(
                pose, 8, w, h
            )

            ears = (
                left_ear + right_ear
            ) / 2

            head_width = (
                np.linalg.norm(
                    left_ear - right_ear
                ) + 1e-6
            )

        else:
            ears = (
                middle
                - down_direction
                * shoulder_width
                * 0.55
            )

            head_width = shoulder_width * 0.45

        # Estimated anatomical regions.
        items = [

            (
                "Brain",
                ears - down_direction * head_width * 0.25,
                head_width * 0.55,
                head_width * 0.45,
                angle
            ),

            (
                "Heart",
                at(0.35, 0.24),
                shoulder_width * 0.11,
                torso_length * 0.11,
                angle
            ),

            (
                "Lung (L)",
                at(0.55, 0.20),
                shoulder_width * 0.12,
                torso_length * 0.20,
                angle
            ),

            (
                "Lung (R)",
                at(-0.55, 0.20),
                shoulder_width * 0.12,
                torso_length * 0.20,
                angle
            ),

            (
                "Liver",
                at(-0.45, 0.45),
                shoulder_width * 0.17,
                torso_length * 0.10,
                angle
            ),

            (
                "Stomach",
                at(0.40, 0.46),
                shoulder_width * 0.11,
                torso_length * 0.10,
                angle
            ),

            (
                "Intestines",
                at(0, 0.72),
                shoulder_width * 0.26,
                torso_length * 0.15,
                angle
            )
        ]

        output = []

        for name, center, rx, ry, rotation in items:

            # Relative image-intensity index.
            warmth_index = None

            if heat_gray is not None:

                heat_gray = np.asarray(
                    heat_gray
                )

                if (
                    heat_gray.ndim == 2
                    and heat_gray.size > 0
                ):

                    heat_height, heat_width = (
                        heat_gray.shape[:2]
                    )

                    center_x = int(
                        center[0]
                        * heat_width
                        / w
                    )

                    center_y = int(
                        center[1]
                        * heat_height
                        / h
                    )

                    radius_x = max(
                        1,
                        int(rx * heat_width / w)
                    )

                    radius_y = max(
                        1,
                        int(ry * heat_height / h)
                    )

                    region_mask = np.zeros(
                        (heat_height, heat_width),
                        dtype=np.uint8
                    )

                    cv2.ellipse(
                        region_mask,
                        (center_x, center_y),
                        (radius_x, radius_y),
                        rotation,
                        0,
                        360,
                        255,
                        -1
                    )

                    values = heat_gray[
                        region_mask > 0
                    ]

                    mean_value = _safe_mean(
                        values
                    )

                    if mean_value is not None:

                        warmth_index = int(
                            np.clip(
                                mean_value / 255 * 100,
                                0,
                                100
                            )
                        )

            # Normalized coordinates for frontend rendering.
            output.append({

                "name": name,

                "cx": round(
                    float(center[0] / w),
                    3
                ),

                "cy": round(
                    float(center[1] / h),
                    3
                ),

                "rx": round(
                    float(rx / w),
                    3
                ),

                "ry": round(
                    float(ry / h),
                    3
                ),

                "angle": round(
                    float(rotation),
                    1
                ),

                "warmth_idx": warmth_index,

                "measurement_type": (
                    "relative_image_intensity"
                    if warmth_index is not None
                    else "unavailable"
                )

            })

        return output

    except (
        ValueError,
        TypeError,
        IndexError,
        ZeroDivisionError,
        FloatingPointError
    ):
        return []
