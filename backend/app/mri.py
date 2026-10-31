
"""Brain-MRI image analysis.

Research/education aid only — NOT a diagnostic device.
The optional PyTorch classifier is loaded only when a model file exists.
"""

import base64
import threading

import cv2
import numpy as np

from . import config


_model = None
_tried = False
_model_lock = threading.Lock()

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_IMAGE_SIDE = 512


def _load():
    """Load the optional MRI classifier once, only when weights exist."""
    global _model, _tried

    if _tried:
        return _model

    with _model_lock:
        if _tried:
            return _model

        # Do not import torch/torchvision if no trained model is installed.
        if not config.MRI_MODEL.exists():
            _tried = True
            return None

        try:
            import torch
            import torchvision

            ck = torch.load(
                config.MRI_MODEL,
                map_location="cpu",
                weights_only=False
            )

            required = {"arch", "classes", "state"}
            if not isinstance(ck, dict) or not required.issubset(ck):
                raise ValueError("MRI model checkpoint has an invalid format.")

            if not hasattr(torchvision.models, ck["arch"]):
                raise ValueError("Unsupported MRI model architecture.")

            model = getattr(torchvision.models, ck["arch"])(weights=None)
            model.fc = torch.nn.Linear(
                model.fc.in_features,
                len(ck["classes"])
            )
            model.load_state_dict(ck["state"])
            model.eval()

            _model = (model, ck["classes"])

        except Exception as exc:
            # Keep the API operational if optional classifier loading fails.
            print(f"MRI classifier unavailable: {type(exc).__name__}: {exc}")
            _model = None

        _tried = True
        return _model


def classify(gray):
    """Run optional classifier. Returns None when no model is available."""
    mm = _load()

    if mm is None:
        return None

    import torch

    model, classes = mm

    x = cv2.resize(gray, (224, 224)).astype(np.float32) / 255.0

    x = np.stack([x] * 3)
    mean = np.array([.485, .456, .406], dtype=np.float32)[:, None, None]
    std = np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
    x = (x - mean) / std

    tensor = torch.tensor(x[None], dtype=torch.float32)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].tolist()

    index = int(np.argmax(probabilities))

    return {
        "label": classes[index],
        "probs": {
            label: round(float(value), 3)
            for label, value in zip(classes, probabilities)
        }
    }


def dotted(img, cnt, color, step=7, r=2):
    points = cnt.reshape(-1, 2)

    for x, y in points[::step]:
        cv2.circle(
            img,
            (int(x), int(y)),
            r,
            color,
            -1,
            cv2.LINE_AA
        )


def analyze(data: bytes):
    """Analyze an uploaded MRI image and produce a visual overlay."""

    if not data:
        raise ValueError("The uploaded file is empty.")

    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("MRI image exceeds the 12 MB upload limit.")

    # Decode once and validate dimensions before resizing.
    gray = cv2.imdecode(
        np.frombuffer(data, np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    if gray is None:
        raise ValueError("Not a readable image.")

    height, width = gray.shape

    if height * width > MAX_IMAGE_PIXELS:
        raise ValueError("Image resolution is too large. Maximum is 16 megapixels.")

    scale = min(1.0, MAX_IMAGE_SIDE / max(height, width))

    if scale < 1.0:
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, threshold = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8)
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    notes = []

    if not contours:
        raise ValueError(
            "No head/brain region found. Please check the uploaded MRI slice."
        )

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour) / gray.size

    if area < 0.05:
        notes.append(
            "Detected region is very small; the image may be unsuitable."
        )

    dotted(overlay, contour, (255, 220, 0), 7, 2)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    inside = gray[mask > 0]
    sequence_guess = "unknown"
    candidate_count = 0

    if inside.size:
        high = np.percentile(inside, 96)
        low = np.percentile(inside, 8)
        median = np.median(inside)

        bright = (
            (gray > high) & (mask > 0)
        ).astype(np.uint8) * 255

        bright = cv2.morphologyEx(
            bright,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8)
        )

        bright_contours, _ = cv2.findContours(
            bright,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        for candidate in bright_contours:
            if cv2.contourArea(candidate) > 40:
                dotted(overlay, candidate, (255, 0, 200), 4, 2)
                candidate_count += 1

        dark = (
            (gray < low) & (mask > 0)
        ).astype(np.uint8) * 255

        dark_contours, _ = cv2.findContours(
            dark,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        for candidate in dark_contours:
            if cv2.contourArea(candidate) > 60:
                dotted(overlay, candidate, (0, 200, 255), 5, 1)

        if (
            (high - median) / (median + 1) > 1.1
            and np.mean(inside > median * 1.6) > 0.08
        ):
            sequence_guess = "T2/FLAIR-like (fluid bright)"
        else:
            sequence_guess = "T1-like (fluid dark)"

        notes.append(
            "Sequence estimate is based on image intensity only and has low confidence."
        )

    success, buffer = cv2.imencode(".png", overlay)

    if not success:
        raise ValueError("Could not generate the MRI overlay.")

    # Optional model inference. A missing model does not stop image analysis.
    try:
        classification = classify(gray)
    except Exception as exc:
        classification = None
        notes.append(
            f"Optional classifier unavailable: {type(exc).__name__}."
        )

    return {
        "overlay": (
            "data:image/png;base64,"
            + base64.b64encode(buffer).decode("ascii")
        ),
        "sequence_guess": sequence_guess,
        "bright_candidate_regions": candidate_count,
        "brain_area_fraction": round(float(area), 3),
        "classification": classification,
        "notes": notes + [
            "This is an experimental image-processing result, not a medical diagnosis.",
            "Bright regions are image-intensity candidates, not confirmed lesions or tumors.",
            "A qualified radiologist must interpret the original MRI scan."
        ]
    }
