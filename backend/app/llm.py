"""LLM + RAG + translation. Works offline with a rule-based fallback."""
import re, math, json, collections
import requests
from . import config

_chunks = None; _emb = None; _st = None
def _load_kb():
    global _chunks, _emb, _st
    if _chunks is not None: return
    _chunks = []
    for p in sorted(config.KNOWLEDGE.glob("*.md")):
        for para in re.split(r"\n\s*\n", p.read_text(encoding="utf8")):
            if len(para.strip()) > 30: _chunks.append({"src": p.name, "text": para.strip()})
    try:
        from sentence_transformers import SentenceTransformer
        _st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _emb = _st.encode([c["text"] for c in _chunks], normalize_embeddings=True)
    except Exception:
        _st = None

def _bow(t): return collections.Counter(re.findall(r"[a-z]+", t.lower()))
def retrieve(q, k=3):
    _load_kb()
    if not _chunks: return []
    if _st is not None:
        import numpy as np
        s = _emb @ _st.encode([q], normalize_embeddings=True)[0]; idx = np.argsort(-s)[:k]
        return [_chunks[i] for i in idx if s[i] > 0.15]
    qv = _bow(q)
    def cos(a, b):
        n = sum(a[t] * b[t] for t in a); return n / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())) + 1e-9)
    sc = sorted(((cos(qv, _bow(c["text"])), c) for c in _chunks), key=lambda x: -x[0])[:k]
    return [c for s, c in sc if s > 0.05]

def ask_llm(system, prompt, max_tokens=400):
    """Anthropic API -> Ollama -> None."""
    if config.ANTHROPIC_API_KEY:
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=40,
                headers={"x-api-key": config.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": config.LLM_MODEL, "max_tokens": max_tokens, "system": system,
                      "messages": [{"role": "user", "content": prompt}]})
            return r.json()["content"][0]["text"].strip()
        except Exception: pass
    if config.OLLAMA_URL:
        try:
            r = requests.post(config.OLLAMA_URL.rstrip("/") + "/api/chat", timeout=60, json={"model": config.OLLAMA_MODEL, "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]})
            return r.json()["message"]["content"].strip()
        except Exception: pass
    return None

def rag_answer(question, context_extra=""):
    docs = retrieve(question)
    ctx = "\n\n".join(f"[{d['src']}] {d['text']}" for d in docs)
    sys = ("You are TRI-NAYANTRA Safety & Wellness Copilot. Answer briefly using ONLY the provided policy context and live "
           "data. If the answer is not in them, say so. You are not a doctor.")
    ans = ask_llm(sys, f"Context:\n{ctx}\n\nLive data:\n{context_extra}\n\nQuestion: {question}")
    if ans is None:
        ans = ("(LLM offline - showing retrieved policy)\n" + "\n".join(d["text"] for d in docs)) if docs else "No matching policy found."
    return {"answer": ans, "sources": [d["src"] for d in docs]}

PHRASES = {"HELLO": "Hello", "STOP": "Please stop", "I_NEED": "I need", "WATER": "water", "FOOD": "food",
           "WASHROOM": "to use the washroom", "THANK_YOU": "Thank you", "YES": "Yes", "NO": "No", "PAIN": "I am in pain",
           "HELP": "I need help, please come quickly"}
LANGS = {"en": "English", "hi": "Hindi", "bho": "Bhojpuri", "mai": "Maithili", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
         "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu", "or": "Odia"}

def compose(tokens):
    docs = retrieve(" ".join(tokens), 2)
    ans = ask_llm("Convert sign-language gesture tokens into ONE short, natural first-person English sentence. Use the dictionary. "
                  "Output only the sentence.", "Dictionary:\n" + "\n".join(d["text"] for d in docs) + "\nTokens: " + " ".join(tokens), 80)
    if ans: return ans.strip().strip('"')
    s = " ".join(PHRASES.get(t, t.lower()) for t in tokens); return s[0].upper() + s[1:] + "."

def translate(text, lang):
    if lang == "en": return text
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="en", target=lang).translate(text)
    except Exception:
        return text + "  (translation unavailable offline)"

def report(kind, data):
    s = ask_llm(f"You write concise {kind} reports for supervisors/teachers: 5 bullet points max plus one recommendation. Never diagnose.",
                json.dumps(data)[:3500], 350)
    return s or "LLM offline. Raw summary: " + json.dumps(data)[:600]
