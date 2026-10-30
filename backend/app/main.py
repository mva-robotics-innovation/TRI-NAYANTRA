"""TRI-NAYANTRA backend (FastAPI). Run:  python run.py"""
import asyncio, json, csv, io
import numpy as np, cv2
from fastapi import FastAPI, WebSocket, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, PlainTextResponse
from pydantic import BaseModel
from . import config, vision, pipeline, llm, mri, detect

app = FastAPI(title="TRI-NAYANTRA API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
MODES = {"workplace", "student", "health", "gesture", "advanced", "care"}

@app.get("/api/status")
def status():
    return {"project": "TRI-NAYANTRA", "mediapipe": vision.mp is not None, "yolo": detect.YOLO is not None,
            "ppe_model_file": config.PPE_MODEL.exists(), "mri_model_file": config.MRI_MODEL.exists(),
            "llm": "anthropic" if config.ANTHROPIC_API_KEY else "ollama" if config.OLLAMA_URL else "offline-fallback",
            "languages": llm.LANGS}

@app.websocket("/ws/{mode}")
async def ws(sock: WebSocket, mode: str):
    if mode not in MODES: await sock.close(code=4000); return
    await sock.accept(); sess = vision.Session(mode)
    try:
        while True:
            msg = await sock.receive()
            if msg["type"] == "websocket.disconnect": break
            if msg.get("text"):
                try: sess.cfg.update(json.loads(msg["text"]))
                except Exception: pass
                continue
            if msg.get("bytes"):
                frame = cv2.imdecode(np.frombuffer(msg["bytes"], np.uint8), cv2.IMREAD_COLOR)
                if frame is None: continue
                try: out = await asyncio.to_thread(pipeline.process, sess, frame)
                except Exception as e: out = {"error": f"{type(e).__name__}: {e}"}
                await sock.send_json(out)
    finally:
        sess.close()

class Compose(BaseModel): tokens: list[str]; lang: str = "en"
@app.post("/api/gesture/compose")
def gesture_compose(b: Compose):
    en = llm.compose(b.tokens); return {"english": en, "translated": llm.translate(en, b.lang), "lang": b.lang}

@app.post("/api/speech/command")
async def speech_command(file: UploadFile = File(...)):
    from . import speech
    try: x = speech.load_audio(await file.read()); r = speech.transcribe(x)
    except Exception as e: raise HTTPException(500, f"ASR unavailable: {e}")
    on, off = speech.is_command_mode(r["text"]); return {"text": r["text"], "command_mode": on, "exit": off}

class Tr(BaseModel): text: str; lang: str = "hi"
@app.post("/api/translate")
def tr(b: Tr): return {"text": llm.translate(b.text, b.lang)}

class Alert(BaseModel): kind: str; detail: str = ""
@app.post("/api/alert")
def alert(b: Alert): pipeline.notify(b.kind, b.detail, "manual", severe=True); return {"ok": True}

class Ask(BaseModel): question: str; context: str = ""
@app.post("/api/assistant/ask")
def ask(b: Ask): return llm.rag_answer(b.question, b.context)

class Rep(BaseModel): kind: str; data: dict
@app.post("/api/report")
def report(b: Rep): return {"report": llm.report(b.kind, b.data)}

@app.get("/api/events")
def events(limit: int = 100):
    p = config.DATA / "events.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf8").splitlines()[-limit:]][::-1] if p.exists() else []

@app.get("/api/events.csv")
def events_csv():
    s = io.StringIO(); w = csv.writer(s); w.writerow(["ts", "mode", "kind", "detail"])
    for e in reversed(events(100000)): w.writerow([e["ts"], e["mode"], e["kind"], e["detail"]])
    return PlainTextResponse(s.getvalue(), media_type="text/csv")

@app.post("/api/mri/analyze")
async def mri_analyze(file: UploadFile = File(...)):
    try: return await asyncio.to_thread(mri.analyze, await file.read())
    except ValueError as e: raise HTTPException(400, str(e))

@app.post("/api/voice/enroll")
async def voice_enroll(user: str = Form("default"), label: str = Form(...), phrase: str = Form(...), file: UploadFile = File(...)):
    from . import speech; return await asyncio.to_thread(speech.enroll_intent, user, label, phrase, await file.read())

@app.post("/api/voice/clone-enroll")
async def clone_enroll(user: str = Form("default"), consent: bool = Form(False), file: UploadFile = File(...)):
    from . import speech
    if not consent: raise HTTPException(400, "Voice cloning requires the speaker's explicit consent.")
    try: return await asyncio.to_thread(speech.clone_enroll, user, await file.read())
    except ValueError as e: raise HTTPException(400, str(e))

@app.post("/api/voice/interpret")
async def voice_interpret(user: str = Form("default"), file: UploadFile = File(...)):
    from . import speech
    try: return await asyncio.to_thread(speech.interpret, user, await file.read())
    except Exception as e: raise HTTPException(500, f"{type(e).__name__}: {e}")

class Speak(BaseModel): user: str = "default"; text: str; lang: str = "en"
@app.post("/api/voice/speak")
async def voice_speak(b: Speak):
    from . import speech
    wav = await asyncio.to_thread(speech.speak, b.user, b.text, b.lang)
    return Response(wav, media_type="audio/wav") if wav else Response(status_code=204)
