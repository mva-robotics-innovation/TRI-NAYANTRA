
"""
TRI-NAYANTRA — Main FastAPI Backend

YOLO/PPE inference runs in a separate detection microservice.
The main backend handles:
- MediaPipe vision and gesture processing
- Workplace and student monitoring logic
- MRI image analysis
- Language model and RAG endpoints
- Speech and voice endpoints
- Alerts and event reporting
- Advanced pseudo-thermal visualization

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import csv
import io
import threading
import logging

import numpy as np
import cv2

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    Form,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, PlainTextResponse
from pydantic import BaseModel, Field

from . import config, vision, pipeline, llm, mri, detect


# --------------------------------------------------
# APPLICATION
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tri-nayantra")

app = FastAPI(
    title="TRI-NAYANTRA API",
    description="Multimodal computer vision and assistance platform",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODES = {
    "workplace",
    "student",
    "health",
    "gesture",
    "advanced",
    "care",
}

# Limit camera sessions on small Render instances.
CAMERA_LIMIT = threading.BoundedSemaphore(value=1)

# Maximum WebSocket frame size: 5 MB.
MAX_FRAME_BYTES = 5 * 1024 * 1024


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

async def read_limited_upload(
    file: UploadFile,
    max_bytes=None,
):
    """Read an upload with a maximum allowed size."""

    if max_bytes is None:
        max_bytes = config.MAX_IMAGE_UPLOAD_BYTES

    content = await file.read(max_bytes + 1)

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail="Uploaded file exceeds the allowed size.",
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    return content


def detection_status():
    """Get remote detection service status without loading local models."""

    yolo_online = detect.yolo_available()
    ppe_available = detect.ppe_model_loaded()

    return {
        "service_online": yolo_online,
        "ppe_model_available": ppe_available,
    }


# --------------------------------------------------
# ROOT AND HEALTH
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "project": "TRI-NAYANTRA",
        "status": "online",
        "version": "2.0",
        "docs": "/docs",
        "health": "/api/status",
    }


@app.get("/health")
def health():
    """Lightweight platform health check."""

    return {
        "status": "ok",
        "project": "TRI-NAYANTRA",
    }


@app.get("/api/status")
def status():
    """Return backend and remote detection service availability."""

    try:
        remote = detection_status()
    except Exception as exc:
        logger.warning(
            "Detection service status unavailable: %s",
            exc,
        )
        remote = {
            "service_online": False,
            "ppe_model_available": False,
        }

    return {
        "project": "TRI-NAYANTRA",
        "version": "2.0",

        # Import availability does not guarantee successful inference.
        "mediapipe": vision.mp is not None,

        # These fields now describe the remote detection service.
        "yolo": remote["service_online"],
        "detection_service_online": remote["service_online"],
        "ppe_model_file": remote["ppe_model_available"],
        "ppe_model_available": remote["ppe_model_available"],

        # MRI model remains local to the main backend.
        "mri_model_file": config.MRI_MODEL.exists(),

        "llm": (
            "anthropic"
            if config.ANTHROPIC_API_KEY
            else "ollama"
            if config.OLLAMA_URL
            else "offline-fallback"
        ),

        "camera_session_limit": 1,

        "languages": llm.LANGS,
    }


# --------------------------------------------------
# WEBSOCKET CAMERA PROCESSING
# --------------------------------------------------

@app.websocket("/ws/{mode}")
async def ws(sock: WebSocket, mode: str):

    if mode not in MODES:
        await sock.close(code=4000)
        return

    await sock.accept()

    acquired = CAMERA_LIMIT.acquire(
        blocking=False
    )

    if not acquired:
        try:
            await sock.send_json({
                "error": (
                    "Camera is busy. "
                    "Another camera session is active. "
                    "Please try again."
                )
            })
            await sock.close(code=1013)
        except Exception:
            pass

        return

    sess = None

    try:
        sess = vision.Session(mode)

        while True:

            msg = await sock.receive()

            if msg["type"] == "websocket.disconnect":
                break

            # --------------------------------------
            # Session configuration
            # --------------------------------------

            if msg.get("text"):

                try:
                    settings = json.loads(
                        msg["text"]
                    )

                    if isinstance(settings, dict):
                        sess.cfg.update(settings)

                except (
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ):
                    await sock.send_json({
                        "error": "Invalid session configuration."
                    })

                continue

            # --------------------------------------
            # Camera frame
            # --------------------------------------

            raw = msg.get("bytes")

            if not raw:
                continue

            if len(raw) > MAX_FRAME_BYTES:
                await sock.send_json({
                    "error": "Camera frame exceeds the 5 MB limit."
                })
                continue

            try:
                frame = cv2.imdecode(
                    np.frombuffer(
                        raw,
                        dtype=np.uint8,
                    ),
                    cv2.IMREAD_COLOR,
                )

                if frame is None:
                    await sock.send_json({
                        "error": "Invalid camera image."
                    })
                    continue

                # Avoid processing unusually large frames.
                height, width = frame.shape[:2]

                if (
                    width <= 0
                    or height <= 0
                    or width * height > config.MAX_IMAGE_PIXELS
                ):
                    await sock.send_json({
                        "error": "Camera image dimensions exceed the allowed limit."
                    })
                    continue

                # Process outside the event loop.
                result = await asyncio.to_thread(
                    pipeline.process,
                    sess,
                    frame,
                )

                await sock.send_json(result)

            except WebSocketDisconnect:
                break

            except Exception as exc:
                logger.exception(
                    "Camera processing failed"
                )

                try:
                    await sock.send_json({
                        "error": (
                            f"{type(exc).__name__}: {exc}"
                        )
                    })
                except Exception:
                    break

    except WebSocketDisconnect:
        logger.info(
            "Camera disconnected: mode=%s",
            mode,
        )

    except Exception:
        logger.exception(
            "WebSocket session failed: mode=%s",
            mode,
        )

        try:
            await sock.send_json({
                "error": "Camera session encountered an internal error."
            })
        except Exception:
            pass

    finally:

        if sess is not None:
            try:
                await asyncio.to_thread(
                    sess.close
                )
            except Exception:
                logger.exception(
                    "Could not close vision session"
                )

        CAMERA_LIMIT.release()


# --------------------------------------------------
# GESTURE COMPOSITION
# --------------------------------------------------

class Compose(BaseModel):
    tokens: list[str] = Field(
        min_length=1,
        max_length=50,
    )
    lang: str = "en"


@app.post("/api/gesture/compose")
def gesture_compose(b: Compose):

    english = llm.compose(
        b.tokens
    )

    return {
        "english": english,
        "translated": llm.translate(
            english,
            b.lang,
        ),
        "lang": b.lang,
    }


# --------------------------------------------------
# SPEECH COMMANDS
# --------------------------------------------------

@app.post("/api/speech/command")
async def speech_command(
    file: UploadFile = File(...),
):
    from . import speech

    try:
        audio = await read_limited_upload(
            file
        )

        loaded = speech.load_audio(
            audio
        )

        result = await asyncio.to_thread(
            speech.transcribe,
            loaded,
        )

        on, off = speech.is_command_mode(
            result["text"]
        )

        return {
            "text": result["text"],
            "command_mode": on,
            "exit": off,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Speech command processing failed"
        )

        raise HTTPException(
            status_code=503,
            detail=f"ASR unavailable: {exc}",
        )


# --------------------------------------------------
# TRANSLATION
# --------------------------------------------------

class Tr(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=10000,
    )
    lang: str = "hi"


@app.post("/api/translate")
def translate(b: Tr):

    return {
        "text": llm.translate(
            b.text,
            b.lang,
        )
    }


# --------------------------------------------------
# ALERTS
# --------------------------------------------------

class Alert(BaseModel):
    kind: str = Field(
        min_length=1,
        max_length=100,
    )
    detail: str = Field(
        default="",
        max_length=2000,
    )


@app.post("/api/alert")
def alert(b: Alert):

    pipeline.notify(
        b.kind,
        b.detail,
        "manual",
        severe=True,
    )

    return {
        "ok": True,
        "message": "Alert recorded.",
    }


# --------------------------------------------------
# AI ASSISTANT
# --------------------------------------------------

class Ask(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=5000,
    )
    context: str = Field(
        default="",
        max_length=20000,
    )


@app.post("/api/assistant/ask")
def ask(b: Ask):

    return llm.rag_answer(
        b.question,
        b.context,
    )


# --------------------------------------------------
# REPORT GENERATION
# --------------------------------------------------

class Rep(BaseModel):
    kind: str = Field(
        min_length=1,
        max_length=100,
    )
    data: dict


@app.post("/api/report")
def report(b: Rep):

    return {
        "report": llm.report(
            b.kind,
            b.data,
        )
    }


# --------------------------------------------------
# EVENTS
# --------------------------------------------------

@app.get("/api/events")
def events(limit: int = 100):

    p = config.DATA / "events.jsonl"

    if not p.exists():
        return []

    limit = max(
        1,
        min(limit, 10000),
    )

    try:
        lines = p.read_text(
            encoding="utf8"
        ).splitlines()

    except OSError as exc:
        logger.exception(
            "Could not read event file"
        )
        raise HTTPException(
            status_code=500,
            detail="Could not read event history.",
        )

    result = []

    for line in lines[-limit:]:

        try:
            result.append(
                json.loads(line)
            )

        except json.JSONDecodeError:
            continue

    return result[::-1]


@app.get("/api/events.csv")
def events_csv():

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "ts",
        "mode",
        "kind",
        "detail",
    ])

    for event in reversed(
        events(10000)
    ):

        writer.writerow([
            event.get("ts", ""),
            event.get("mode", ""),
            event.get("kind", ""),
            event.get("detail", ""),
        ])

    return PlainTextResponse(
        output.getvalue(),
        media_type="text/csv",
    )


# --------------------------------------------------
# MRI ANALYSIS
# --------------------------------------------------

@app.post("/api/mri/analyze")
async def mri_analyze(
    file: UploadFile = File(...),
):

    try:
        content = await read_limited_upload(
            file
        )

        result = await asyncio.to_thread(
            mri.analyze,
            content,
        )

        return result

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        logger.exception(
            "MRI analysis failed"
        )

        raise HTTPException(
            status_code=503,
            detail=f"MRI analysis unavailable: {exc}",
        )


# --------------------------------------------------
# VOICE ENROLLMENT
# --------------------------------------------------

@app.post("/api/voice/enroll")
async def voice_enroll(
    user: str = Form("default"),
    label: str = Form(...),
    phrase: str = Form(...),
    file: UploadFile = File(...),
):

    from . import speech

    content = await read_limited_upload(
        file
    )

    try:
        return await asyncio.to_thread(
            speech.enroll_intent,
            user,
            label,
            phrase,
            content,
        )

    except Exception as exc:
        logger.exception(
            "Voice enrollment failed"
        )

        raise HTTPException(
            status_code=503,
            detail=f"Voice enrollment unavailable: {exc}",
        )


# --------------------------------------------------
# VOICE CLONING ENROLLMENT
# --------------------------------------------------

@app.post("/api/voice/clone-enroll")
async def clone_enroll(
    user: str = Form("default"),
    consent: bool = Form(False),
    file: UploadFile = File(...),
):

    from . import speech

    if not consent:
        raise HTTPException(
            status_code=400,
            detail=(
                "Voice cloning requires the speaker's "
                "explicit consent."
            ),
        )

    content = await read_limited_upload(
        file
    )

    try:
        return await asyncio.to_thread(
            speech.clone_enroll,
            user,
            content,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        logger.exception(
            "Voice cloning enrollment failed"
        )

        raise HTTPException(
            status_code=503,
            detail=f"Voice cloning unavailable: {exc}",
        )


# --------------------------------------------------
# VOICE INTERPRETATION
# --------------------------------------------------

@app.post("/api/voice/interpret")
async def voice_interpret(
    user: str = Form("default"),
    file: UploadFile = File(...),
):

    from . import speech

    content = await read_limited_upload(
        file
    )

    try:
        return await asyncio.to_thread(
            speech.interpret,
            user,
            content,
        )

    except Exception as exc:
        logger.exception(
            "Voice interpretation failed"
        )

        raise HTTPException(
            status_code=503,
            detail=f"Voice interpretation unavailable: {exc}",
        )


# --------------------------------------------------
# VOICE OUTPUT
# --------------------------------------------------

class Speak(BaseModel):
    user: str = "default"

    text: str = Field(
        min_length=1,
        max_length=10000,
    )

    lang: str = "en"


@app.post("/api/voice/speak")
async def voice_speak(b: Speak):

    from . import speech

    try:
        wav = await asyncio.to_thread(
            speech.speak,
            b.user,
            b.text,
            b.lang,
        )

        if not wav:
            return Response(
                status_code=204
            )

        return Response(
            content=wav,
            media_type="audio/wav",
        )

    except Exception as exc:
        logger.exception(
            "Voice output failed"
        )

        raise HTTPException(
            status_code=503,
            detail=f"Voice output unavailable: {exc}",
        )
