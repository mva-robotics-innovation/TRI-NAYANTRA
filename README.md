# TRI-NAYANTRA — Vision · Voice · Vitals

Frontend and backend are **separate apps**.

```
TRI-NAYANTRA/
├─ backend/   FastAPI + Python (MediaPipe, YOLOv8, Whisper, RAG+LLM, MRI CNN, XTTS)
│  ├─ app/{main,pipeline,vision,detect,bodyscan,mri,speech,llm,config}.py
│  ├─ knowledge/*.md      <- RAG knowledge base (edit with YOUR safety/health policy)
│  ├─ download_models.py  train_mri.py  run.py  tests/
└─ frontend/  static HTML/JS (no build step): index, health, workplace, students, gesture, advanced, voice
```

## Run
```bash
# 1. backend (Python 3.10-3.11 recommended)
cd backend
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # add ANTHROPIC_API_KEY, or set OLLAMA_URL for a local LLM
python download_models.py --ppe                     # add --xtts for voice cloning (needs: pip install TTS)
python run.py                                       # http://localhost:8000/docs

# 2. frontend (second terminal) - camera needs localhost or https
cd frontend && python -m http.server 5500           # open http://localhost:5500
```
Change the backend URL in `frontend/js/config.js`. Add the frontend origin to `CORS_ORIGINS` in `.env`.

## Models
| Purpose | Model | How you get it |
|---|---|---|
| Face / body / hands (dotted tracking) | MediaPipe Holistic | bundled in `pip install mediapipe==0.10.14` |
| Workstation objects, phones, persons | YOLOv8n (COCO) | auto via `download_models.py` |
| PPE (helmet, gloves, goggles, mask, shoes) | YOLOv8 PPE weights (Hugging Face `keremberke/yolov8m-protective-equipment-detection`) | `download_models.py --ppe` -> `models/ppe_best.pt`. **Verify the repo still exists**; otherwise point `PPE_MODEL` at any YOLOv8 PPE `.pt` |
| Speech recognition | faster-whisper (`small`) | auto |
| RAG embeddings | all-MiniLM-L6-v2 | auto |
| Voice clone | Coqui XTTS-v2 (non-commercial licence) | `--xtts` |
| Brain-MRI classifier | ResNet-18 you train | `python train_mri.py --data data/mri` (Kaggle Brain Tumor MRI dataset) |
| Sign language | rule-based starter vocabulary | replace `vision.classify_hand` with a landmark classifier trained on ISL/ASL |

## What is real, what is approximate
* **Real:** dotted face/body/hand tracking, expression & fatigue heuristics (EAR/MAR/PERCLOS), head-pose attention, phone/PPE detection, fall check, rPPG heart rate (approximate), Whisper ASR, personal voice-pattern matching, RAG answers, MRI contour overlay on uploaded scans.
* **Estimated / simulated:** the "organ view" is an *anatomical atlas aligned to your pose* over a *pseudo-thermal* image. A webcam cannot see organs or temperature. For real heat feed a FLIR Lepton / MLX90640 frame into `bodyscan.thermal_view(heat=...)`.
* MediaPipe Holistic tracks **one** primary person; YOLO counts everyone. For several workers at once, swap in YOLOv8-pose + per-person crops.
* Thresholds (eyes, gaze, motion) are at the top of `vision.py` — tune for your camera.
* Not tested here: the live MediaPipe/YOLO/Whisper/XTTS paths (the build sandbox had no internet or GPU). Unit tests cover rPPG, PPE logic, phone-in-hand, finger states, RAG fallback; the MRI module was run on a synthetic image. Expect to tune on your hardware.

## Responsible use
Get informed consent for workplace/student monitoring (and follow local labour/privacy law), for voice cloning, and for health scanning. Outputs are wellness/safety aids, **not diagnoses**. Events are logged locally in `backend/data/events.jsonl` (`/api/events.csv`).
