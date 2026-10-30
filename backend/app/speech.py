"""Speech: Whisper ASR (+ enhancement for faint/unclear voices), personalised intent matching for
unclear speech (MFCC nearest-neighbour on the user's own enrolled phrases), and voice cloning (XTTS v2).
Voice cloning needs the speaker's explicit consent - enforced at the API layer."""
import io, json, re
import numpy as np
from scipy import signal
from . import config, llm

_whisper = None; _tts = None
VOICE_DIR = config.DATA / "voice"; VOICE_DIR.mkdir(exist_ok=True)

def load_audio(data: bytes):
    from faster_whisper.audio import decode_audio   # decodes webm/ogg/wav/mp3 via bundled PyAV
    return decode_audio(io.BytesIO(data), sampling_rate=16000)

def enhance(x, sr=16000):
    """Pre-emphasis + spectral-gate noise reduction + loudness normalisation, to help faint 'whisper' speech."""
    x = np.append(x[0], x[1:] - 0.95 * x[:-1]).astype(np.float32)
    f, t, Z = signal.stft(x, sr, nperseg=512); mag = np.abs(Z)
    noise = np.percentile(mag, 10, axis=1, keepdims=True)
    gain = np.clip((mag - 1.5 * noise) / (mag + 1e-8), 0.05, 1.0)
    _, y = signal.istft(Z * gain, sr, nperseg=512)
    y = y / (np.max(np.abs(y)) + 1e-6) * 0.9
    return y.astype(np.float32)

def transcribe(x, prompt=None):
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    segs, info = _whisper.transcribe(x, beam_size=5, best_of=5, temperature=[0.0, 0.2, 0.4], word_timestamps=True,
                                     initial_prompt=prompt, vad_filter=False, condition_on_previous_text=False)
    words, text = [], []
    for s in segs:
        text.append(s.text.strip())
        for w in s.words or []: words.append({"w": w.word.strip(), "p": round(w.probability, 2)})
    conf = float(np.mean([w["p"] for w in words])) if words else 0.0
    return {"text": " ".join(text).strip(), "language": info.language, "confidence": round(conf, 2), "words": words}

def is_command_mode(text):
    t = re.sub(r"[^a-z ]", "", text.lower())
    return ("command mode" in t or "commandmode" in t or "command more" in t), ("exit command" in t or "stop command" in t)

def embed(x, sr=16000):
    import librosa
    m = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20); d = librosa.feature.delta(m)
    return np.concatenate([m.mean(1), m.std(1), d.std(1)]).tolist()

def _store(user): p = VOICE_DIR / re.sub(r"\W", "_", user or "default"); p.mkdir(exist_ok=True, parents=True); return p
def _db(user):
    f = _store(user) / "intents.json"; return json.loads(f.read_text()) if f.exists() else {}
def enroll_intent(user, label, phrase, audio: bytes):
    x = load_audio(audio); db = _db(user); e = db.setdefault(label, {"phrase": phrase, "emb": []})
    e["phrase"] = phrase; e["emb"].append(embed(x)); (_store(user) / "intents.json").write_text(json.dumps(db))
    return {"label": label, "samples": len(e["emb"])}
def match_intent(user, x):
    db = _db(user)
    if not db: return None
    allv = np.array([v for e in db.values() for v in e["emb"]]); mu, sd = allv.mean(0), allv.std(0) + 1e-6
    q = (np.array(embed(x)) - mu) / sd; best = None
    for label, e in db.items():
        for v in e["emb"]:
            v = (np.array(v) - mu) / sd; c = float(q @ v / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))
            if best is None or c > best[1]: best = (label, c, e["phrase"])
    return {"label": best[0], "score": round(best[1], 2), "phrase": best[2]}

def interpret(user, audio: bytes):
    x = load_audio(audio); xe = enhance(x); db = _db(user)
    vocab = ", ".join(e["phrase"] for e in db.values())
    asr = transcribe(xe, prompt=vocab or None); mi = match_intent(user, x)
    decision = None; how = None
    if mi and mi["score"] >= 0.55 and sum(len(e["emb"]) for e in db.values()) >= 2:
        decision, how = mi["phrase"], f"personal voice-pattern match ({mi['score']})"
    if asr["confidence"] >= 0.6 and asr["text"]: decision, how = asr["text"], f"Whisper transcript ({asr['confidence']})"
    if decision is None or asr["confidence"] < 0.6:
        g = llm.ask_llm("A person with unclear speech said something. Using the noisy transcript and their known phrases, give the single most likely intended "
                        "message in clear first-person English. Output only the message, or UNSURE.",
                        f"Noisy transcript: {asr['text']!r}\nKnown phrases: {vocab or 'none'}\nVoice-pattern best match: {mi}", 60)
        if g and "UNSURE" not in g: decision, how = g.strip().strip('"'), "LLM decoding of unclear speech"
    return {"asr": asr, "personal_match": mi, "intended": decision or "(could not understand - please repeat)", "method": how,
            "loudness_rms": round(float(np.sqrt(np.mean(x ** 2))), 4)}

def clone_enroll(user, audio: bytes):
    import soundfile as sf
    x = load_audio(audio)
    if len(x) < 16000 * 6: raise ValueError("Please record at least 6 seconds (10-30 s is ideal).")
    sf.write(_store(user) / "voice_sample.wav", x, 16000); return {"seconds": round(len(x) / 16000, 1)}

def speak(user, text, lang="en"):
    """Returns WAV bytes in the enrolled voice, or None if cloning is unavailable (frontend falls back to browser TTS)."""
    global _tts
    ref = _store(user) / "voice_sample.wav"
    if not ref.exists(): return None
    try:
        if _tts is None:
            from TTS.api import TTS
            _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        out = _store(user) / "out.wav"; _tts.tts_to_file(text=text, speaker_wav=str(ref), language=lang, file_path=str(out))
        return out.read_bytes()
    except Exception:
        return None
