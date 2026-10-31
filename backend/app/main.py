
"""TRI-NAYANTRA backend (FastAPI). Run: python run.py"""

import asyncio
import json
import csv
import io
import threading

import numpy as np
import cv2

from fastapi import (
    FastAPI,
    WebSocket,
    UploadFile,
    File,
    Form,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, PlainTextResponse
from pydantic import BaseModel

from . import config, vision, pipeline, llm, mri, detect


app = FastAPI(title="TRI-NAYANTRA API", version="1.1")

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

# Free Render instance: permit only one active camera session.
CAMERA_LIMIT = threading.BoundedSemaphore(value=1)


@app.get("/")
def root():
    return {
        "project": "TRI-NAYANTRA",
        "status": "online",
        "docs": "/docs",
        "health": "/api/status",
    }


@app.get("/api/status")
def status():
    return {
        "project": "TRI-NAYANTRA",
        "mediapipe": vision.mp is not None,
        "yolo": detect.YOLO is not None,
        "ppe_model_file": config.PPE_MODEL.exists(),
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


@app.websocket("/ws/{mode}")
async def ws(sock: WebSocket, mode: str):

    if mode not in MODES:
        await sock.close(code=4000)
        return

    await sock.accept()

    # Do not allow multiple camera sessions to exhaust Render RAM.
    acquired = CAMERA_LIMIT.acquire(blocking=False)

    if not acquired:
        await sock.send_json({
            "error": "Camera is busy. Another camera session is active. Please try again."
        })
        await sock.close(code=1013)
        return

    sess = None

    try:
        sess = vision.Session(mode)

        while True:
            msg = await sock.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("text"):
                try:
                    sess.cfg.update(json.loads(msg["text"]))
                except (ValueError, TypeError):
                    pass
                continue

            if msg.get("bytes"):
                frame = cv2.imdecode(
                    np.frombuffer(msg["bytes"], np.uint8),
                    cv2.IMREAD_COLOR,
                )

                if frame is None:
                    continue

                try:
                    out = await asyncio.to_thread(
                        pipeline.process,
                        sess,
                        frame,
                    )
                except Exception as exc:
                    out = {
                        "error": f"{type(exc).__name__}: {exc}"
                    }

                await sock.send_json(out)

    except Exception as exc:
        # Keep unexpected WebSocket failures from leaking the session.
        try:
            await sock.send_json({
                "error": f"{type(exc).__name__}: {exc}"
            })
        except Exception:
            pass

    finally:
        if sess is not None:
            try:
                sess.close()
            except Exception:
                pass

        CAMERA_LIMIT.release()


# --------------------------------------------------
# Gesture composition
# --------------------------------------------------

class Compose(BaseModel):
    tokens: list[str]
    lang: str = "en"


@app.post("/api/gesture/compose")
def gesture_compose(b: Compose):
    english = llm.compose(b.tokens)

    return {
        "english": english,
        "translated": llm.translate(english, b.lang),
        "lang": b.lang,
    }


# --------------------------------------------------
# Speech commands
# --------------------------------------------------

@app.post("/api/speech/command")
async def speech_command(file: UploadFile = File(...)):
    from . import speech

    try:
        audio = speech.load_audio(await file.read())
        result = speech.transcribe(audio)

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"ASR unavailable: {exc}",
        )

    on, off = speech.is_command_mode(result["text"])

    return {
        "text": result["text"],
        "command_mode": on,
        "exit": off,
    }


# --------------------------------------------------
# Translation
# --------------------------------------------------

class Tr(BaseModel):
    text: str
    lang: str = "hi"


@app.post("/api/translate")
def translate(b: Tr):
    return {
        "text": llm.translate(b.text, b.lang)
    }


# --------------------------------------------------
# Alerts
# --------------------------------------------------

class Alert(BaseModel):
    kind: str
    detail: str = ""


@app.post("/api/alert")
def alert(b: Alert):
    pipeline.notify(
        b.kind,
        b.detail,
        "manual",
        severe=True,
    )

    return {"ok": True}


# --------------------------------------------------
# AI assistant
# --------------------------------------------------

class Ask(BaseModel):
    question: str
    context: str = ""


@app.post("/api/assistant/ask")
def ask(b: Ask):
    return llm.rag_answer(
        b.question,
        b.context,
    )


# --------------------------------------------------
# Reports
# --------------------------------------------------

class Rep(BaseModel):
    kind: str
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
# Events
# --------------------------------------------------

@app.get("/api/events")
def events(limit: int = 100):
    p = config.DATA / "events.jsonl"

    if not p.exists():
        return []

    limit = max(1, min(limit, 10000))

    lines = p.read_text(
        encoding="utf8"
    ).splitlines()

    result = []

    for line in lines[-limit:]:
        try:
            result.append(json.loads(line))
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

    for event in reversed(events(10000)):
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
# MRI analysis
# --------------------------------------------------

@app.post("/api/mri/analyze")
async def mri_analyze(file: UploadFile = File(...)):
    try:
        content = await file.read()

        return await asyncio.to_thread(
            mri.analyze,
            content,
        )

    except ValueError as exc:
        raise HTTPException(400, str(exc))

    except Exception as exc:
        raise HTTPException(
            503,
            f"MRI analysis unavailable: {exc}",
        )


# --------------------------------------------------
# Voice enrollment
# --------------------------------------------------

@app.post("/api/voice/enroll")
async def voice_enroll(
    user: str = Form("default"),
    label: str = Form(...),
    phrase: str = Form(...),
    file: UploadFile = File(...),
):
    from . import speech

    content = await file.read()

    return await asyncio.to_thread(
        speech.enroll_intent,
        user,
        label,
        phrase,
        content,
    )


@app.post("/api/voice/clone-enroll")
async def clone_enroll(
    user: str = Form("default"),
    consent: bool = Form(False),
    file: UploadFile = File(...),
):
    from . import speech

    if not consent:
        raise HTTPException(
            400,
            "Voice cloning requires the speaker's explicit consent.",
        )

    try:
        content = await file.read()

        return await asyncio.to_thread(
            speech.clone_enroll,
            user,
            content,
        )

    except ValueError as exc:
        raise HTTPException(400, str(exc))

    except Exception as exc:
        raise HTTPException(
            503,
            f"Voice cloning unavailable: {exc}",
        )


# --------------------------------------------------
# Voice interpretation
# --------------------------------------------------

@app.post("/api/voice/interpret")
async def voice_interpret(
    user: str = Form("default"),
    file: UploadFile = File(...),
):
    from . import speech

    try:
        content = await file.read()

        return await asyncio.to_thread(
            speech.interpret,
            user,
            content,
        )

    except Exception as exc:
        raise HTTPException(
            503,
            f"Voice interpretation unavailable: {exc}",
        )


# --------------------------------------------------
# Voice output
# --------------------------------------------------

class Speak(BaseModel):
    user: str = "default"
    text: str
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
            return Response(status_code=204)

        return Response(
            content=wav,
            media_type="audio/wav",
        )

    except Exception as exc:
        raise HTTPException(
            503,
            f"Voice output unavailable: {exc}",
        )
