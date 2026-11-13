<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow&logoColor=white" alt="TensorFlow 2.15" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/MediaPipe-Holistic-red?logo=google&logoColor=white" alt="MediaPipe" />
  <img src="https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/Groq_LLM-llama--3.1--8b-purple" alt="Groq LLM" />
</p>

# 🤟 Vocalize — Real-Time Sign Language Translation

**Vocalize** is an end-to-end, AI-powered system that translates American Sign Language (ASL) into natural language **in real time**. A user signs in front of their webcam, and within seconds the system detects individual signs, reconstructs them into grammatically correct English sentences, translates those sentences into 5+ languages, and — with the Live Chat Room — streams the translations to a remote partner over WebSocket.

The entire pipeline — from a raw webcam frame to a translated sentence appearing on a partner's screen — runs through 4 stages of AI:

1. **MediaPipe Holistic** extracts 1,662 body/face/hand landmarks per frame
2. **Bi-LSTM + Attention** classifies 30-frame sequences into sign labels
3. **Groq LLM (Llama 3.1 8B)** corrects ASL gloss grammar and translates
4. **WebSocket Signaling** relays translations peer-to-peer in real time

> Built from scratch across 9+ development phases, from raw dataset parsing to a polished dark-mode web interface with peer-to-peer chat.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Architecture & Data Flow](#architecture--data-flow)
- [Project Structure](#project-structure)
- [Development Phases](#development-phases)
- [Setup & Installation](#setup--installation)
- [Testing with Two Devices (ngrok)](#testing-with-two-devices-ngrok)
- [WebSocket Protocol Reference](#websocket-protocol-reference)
- [Future Roadmap](#future-roadmap)
- [Known Limitations & Challenges Solved](#known-limitations--challenges-solved)

---

## How It Works

```
┌─────────────┐     Base64 JPEG      ┌──────────────────────────────────────────┐
│   Browser   │ ──── WebSocket ────▶ │  FastAPI Backend                         │
│   Webcam    │       15 FPS         │                                          │
│   (app.js)  │                      │  1. Decode frame                         │
│             │ ◀── predictions ──── │  2. MediaPipe Holistic → 1662 keypoints  │
│             │ ◀── nlp results ──── │  3. Rolling 30-frame buffer              │
│             │                      │  4. Bi-LSTM + Attention → class + conf   │
│             │                      │  5. Deduplicated DB logging              │
│             │                      │  6. NLP debounce → Groq LLM correction   │
└──────┬──────┘                      └──────────────────────────────────────────┘
       │
       │  nlp_relay (via /ws/signaling)
       ▼
┌─────────────┐
│  Partner's  │
│  Browser    │
└─────────────┘
```

**Frame → Translation in 6 steps:**

| Step | Component | What Happens |
|------|-----------|-------------|
| 1 | **Frontend** | Captures webcam at 15 FPS, downscales to 640px, encodes as Base64 JPEG |
| 2 | **MediaPipe** | Holistic model extracts 33 pose + 468 face + 42 hand landmarks (1,662 floats) |
| 3 | **Keypoint Buffer** | Landmarks are nose-normalized for translation invariance, appended to a 30-frame rolling deque |
| 4 | **Bi-LSTM + Attention** | When buffer is full, the model predicts the sign class with confidence score |
| 5 | **DB Logging** | High-confidence predictions (≥60%) are deduplicated and saved to PostgreSQL |
| 6 | **NLP Engine** | After 2s of signing silence, accumulated gloss words are sent to Groq → corrected English + translation + next-word predictions |

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML Pipeline** | MediaPipe Holistic | Real-time body/face/hand landmark extraction |
| | TensorFlow 2.15 / Keras | Bi-LSTM + custom Attention layer for sequence classification |
| | NumPy, OpenCV, scikit-learn | Data processing, augmentation, train/test split |
| **NLP** | Groq Cloud API | Async LLM calls to Llama 3.1 8B for grammar correction, translation & auto-complete |
| **Backend** | FastAPI | WebSocket inference endpoint, REST transcript API, signaling relay |
| | SQLAlchemy + PostgreSQL | Persistent transcript storage with session isolation |
| | Pydantic | Request/response schema validation |
| **Frontend** | Vanilla HTML/CSS/JS | Single-page app with dark glassmorphism UI, no build step |
| | Web Speech API | Optional spoken input via browser microphone |
| **Infrastructure** | uvicorn | ASGI server with hot-reload |
| | python-dotenv | Environment variable management |

---

## Architecture & Data Flow

```mermaid
graph LR
    subgraph Browser["Browser (app.js)"]
        CAM[Webcam 15fps] --> B64[Base64 JPEG]
        B64 -->|WS frame| INF_WS["/ws/inference/{session}"]
    end

    subgraph Backend["FastAPI Backend"]
        INF_WS --> DEC[Decode Frame]
        DEC --> MP[MediaPipe Holistic]
        MP --> KP["Keypoints (1662)"]
        KP --> BUF[30-Frame Buffer]
        BUF --> LSTM["Bi-LSTM + Attention"]
        LSTM --> PRED[Prediction + Confidence]
        PRED -->|≥60%| DB[(PostgreSQL)]
        PRED -->|word tracking| NLP_Q[NLP Debounce 2s]
        NLP_Q --> GROQ[Groq LLM API]
        GROQ --> NLP_R[NLP Result]
    end

    PRED -->|prediction msg| Browser
    NLP_R -->|nlp msg| Browser
    Browser -->|nlp_relay| SIG["/ws/signaling/{room}"]
    SIG -->|broadcast| PARTNER[Partner Browser]
```

### Two Independent WebSocket Channels

1. **`/ws/inference/{session_id}`** — Dedicated per-user ML pipeline. Streams frames in, returns predictions and NLP results. Each connection loads the model singleton, maintains its own keypoint buffer, word tracker, and NLP debounce timer.

2. **`/ws/signaling/{room_id}`** — Stateless message relay. Any JSON sent by one peer is broadcast to all other peers in the same room. Used by the Live Chat Room feature to relay NLP-corrected translations between partners.

---

## Project Structure

```
Vocalize/
├── .env                        # Environment variables (DB URL, Groq API key)
├── .gitignore                  # Ignores venv, datasets, model weights, .env
├── .python-version             # Python version pin (3.11)
├── requirements.txt            # All Python dependencies
├── README.md                   # This file
│
├── ── ML Pipeline ──────────────────────────────────────────────
├── wlasl_parser.py             # Phase 0: Parse WLASL JSON, copy videos to dataset/
├── landmark_extractor.py       # Phase 1: MediaPipe Holistic detection + keypoint extraction
├── dataset_processor.py        # Phase 2: Batch-extract keypoints from all videos → .npy
├── augment_data.py             # Phase 2b: Generate 15 augmented copies per sample
├── train_model.py              # Phase 3: Train Bi-LSTM + Attention, save wlasl_15_gloss.keras
├── attention.py                # Custom Bahdanau attention Keras layer (shared)
├── realtime_inference.py       # Phase 4: Standalone webcam inference with OpenCV HUD
│
├── ── Backend (FastAPI) ────────────────────────────────────────
├── backend/
│   ├── __init__.py             # Package marker
│   ├── main.py                 # App entrypoint: lifespan, CORS, router mounts, static files
│   ├── config.py               # Pydantic settings from .env (DB URL, Groq key, CORS)
│   ├── database.py             # SQLAlchemy engine, session factory, get_db dependency
│   ├── models.py               # ORM model: Transcript table with session isolation
│   ├── schemas.py              # Pydantic schemas: TranscriptCreate/Read, SessionSummary
│   ├── nlp_engine.py           # Groq-powered gloss→English→translation + auto-complete
│   ├── wlasl_15_gloss.keras               # Trained Bi-LSTM+Attention model weights (~23 MB)
│   ├── wlasl_15_gloss_labels.json      # Class index → label name mapping
│   └── routers/
│       ├── __init__.py         # Package marker
│       ├── inference.py        # WS /ws/inference/{id} — frame decode, ML, DB, NLP
│       ├── signaling.py        # WS /ws/signaling/{id} — peer relay + room management
│       └── transcripts.py      # REST /transcripts — CRUD + session summary
│
├── ── Frontend ─────────────────────────────────────────────────
├── frontend/
│   ├── index.html              # Single-page UI: video panel, sidebar cards, all CSS
│   └── app.js                  # Client logic: webcam capture, WS, transcript, chat room
│
├── ── Data Directories (gitignored) ────────────────────────────
├── dataset/                    # Raw MP4 videos organized by label
├── processed_data/             # Extracted .npy keypoint sequences
├── output/                     # Ad-hoc saved keypoints from Phase 1
└── logs/                       # TensorBoard training logs
```

---

## Development Phases

### Phase 1 — Landmark Extraction
`landmark_extractor.py`

Established the foundational computer vision pipeline using **MediaPipe Holistic**. Extracts 1,662 keypoints per frame (33 pose × 4 + 468 face × 3 + 21 left hand × 3 + 21 right hand × 3) with **nose-relative normalization** for translation invariance. Includes standalone webcam preview with landmark overlays.

### Phase 2 — Dataset Processing & Augmentation
`wlasl_parser.py` · `dataset_processor.py` · `augment_data.py`

Built the data pipeline: parse the WLASL v0.3 JSON to identify and copy relevant videos, then batch-extract keypoints into `.npy` arrays. The augmentation module generates **15 augmented copies** per sample using 6 techniques: Gaussian noise (3 levels), coordinate scaling, temporal shifting, temporal scaling, frame dropout, and horizontal mirroring with hand-swap — achieving a **16× dataset expansion**.

### Phase 3 — Model Training
`train_model.py` · `attention.py`

Designed and trained the core classifier: a **Bidirectional LSTM** stack (128 → 64 units) with a custom **Bahdanau Attention** layer that learns per-frame importance. Architecture: `Masking → BiLSTM → BiLSTM → Attention → Dense(64) → Softmax`. Uses EarlyStopping, ModelCheckpoint, and TensorBoard logging. Currently trained on 5 classes: `before`, `book`, `computer`, `drink`, `go`.

### Phase 4 — Real-Time Local Inference
`realtime_inference.py`

Standalone OpenCV application that loads the trained model and runs live webcam inference with a rolling 30-frame buffer. Uses `model(tensor, training=False)` direct calls instead of `model.predict()` to avoid TF graph re-tracing overhead. Includes a rendered HUD with prediction text, confidence bar, and buffer fill indicator.

### Phase 5 — Backend API & Database
`backend/main.py` · `database.py` · `models.py` · `schemas.py` · `config.py` · `routers/transcripts.py`

Built the FastAPI backend with PostgreSQL persistence. The `Transcript` table stores detections with session isolation, deduplication (2s cooldown per label), and composite indexing for ordered replay. REST endpoints provide CRUD operations and session summaries. Pydantic settings load config from `.env`.

### Phase 6 — WebSocket Inference Pipeline
`backend/routers/inference.py`

Migrated the ML pipeline into a WebSocket endpoint that receives Base64 JPEG frames from the browser, runs the full MediaPipe → LSTM pipeline server-side, and streams predictions back. Implements frame queue draining (only processes the latest frame) to prevent backpressure. High-confidence detections are logged to PostgreSQL in real time.

### Phase 7 — NLP Grammar Correction & Translation
`backend/nlp_engine.py` (integrated in `inference.py`)

Added the Groq LLM layer that fires after a **2-second debounce** (no new sign detected). Accumulated gloss words are sent to Llama 3.1 8B with a strict JSON-only prompt. Returns corrected English, translation into the user's selected language (English, Hindi, Kannada, Tamil, Spanish), and 3 next-word predictions. Includes defensive parsing: strips markdown fences, catches `JSONDecodeError`, and never crashes the inference loop.

### Phase 8 — Frontend Web Application
`frontend/index.html` · `frontend/app.js`

Single-page dark-mode UI served as static files. Features:
- **Video panel** with webcam feed, prediction overlay, confidence bar, and frame buffer indicator
- **Sidebar cards** (all collapsible): Session info, AI Translation (with language selector), Next Word Predictions, Transcript Log, Spoken Input (Web Speech API)
- Glassmorphism design with CSS variables, ambient gradient backgrounds, and slide-in animations
- WebSocket client that captures frames at 15 FPS and renders server responses in real time

### Phase 9 — WebSocket Signaling & Peer Relay
`backend/routers/signaling.py`

WebSocket relay server for peer-to-peer communication. The `ConnectionManager` class maintains room-based connection pools and broadcasts any JSON message to all other peers in the same room. Automatically handles peer join notifications (`room_info`), disconnect cleanup (`peer_left`), and provides a REST endpoint to list active rooms.

### Phase 9b — Live Chat Room (Peer-to-Peer NLP Chat)
`frontend/app.js` · `frontend/index.html` (additions only)

Added a second WebSocket connection to the signaling endpoint, enabling two users to see each other's NLP-corrected translations in real time:
- **Room setup**: Generate or enter a 6-character alphanumeric room code
- **Peer status**: Live indicator showing "Waiting for partner..." or "Partner connected"
- **Chat feed**: Right-aligned bubbles (own translations, accent gradient) and left-aligned bubbles (partner translations, card background) with timestamps
- NLP results are automatically relayed via `nlp_relay` messages; the inference WS remains completely untouched
- Error handling: inline warnings (no `alert()`), graceful disconnect recovery, zero impact on inference FPS

---

## Setup & Installation

### Prerequisites

- **Python 3.11** (specified in `.python-version`)
- **PostgreSQL 14+** running locally
- **Groq API key** (free tier at [console.groq.com](https://console.groq.com))

### 1. Clone & Virtual Environment

```bash
git clone https://github.com/your-username/vocalize.git
cd vocalize

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. PostgreSQL Database

```sql
-- Connect to PostgreSQL and create the database:
CREATE DATABASE vocalize_db;
```

Tables are created automatically on first startup via `Base.metadata.create_all()`.

### 4. Environment Variables

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/vocalize_db
GROQ_API_KEY=gsk_your_groq_api_key_here
DEBUG=True
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,http://127.0.0.1:8000
```

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | ✅ | PostgreSQL connection string |
| `GROQ_API_KEY` | ✅ | Groq API key for NLP correction & translation |
| `DEBUG` | ❌ | Enables SQL echo logging (default: `True`) |
| `CORS_ORIGINS` | ❌ | Comma-separated allowed origins |

### 5. Run the Dev Server

```bash
uvicorn backend.main:app --reload --port 8000
```

Open the app at: **http://127.0.0.1:8000/app/**

### 6. (Optional) Train Your Own Model

If you want to retrain with different signs:

```bash
# Step 1: Parse WLASL dataset
python wlasl_parser.py --json_path path/to/WLASL_v0.3.json --videos_dir path/to/videos

# Step 2: Extract keypoints
python dataset_processor.py --input_dir dataset --output_dir processed_data

# Step 3: Augment data (15x expansion)
python augment_data.py --data_dir processed_data

# Step 4: Train
python train_model.py --data_dir processed_data --epochs 200

# The model saves to wlasl_15_gloss.keras and wlasl_15_gloss_labels.json
# Copy both into backend/ to deploy
```

---

## Testing with Two Devices (ngrok)

To test the Live Chat Room feature across two devices (or two machines on different networks):

### Step 1 — Start the Server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Step 2 — Install & Authenticate ngrok

```bash
# Install ngrok (https://ngrok.com/download)
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

### Step 3 — Expose the Server

```bash
ngrok http 8000
```

ngrok will output a public URL like `https://abc123.ngrok-free.app`.

### Step 4 — Device A (Your Machine)

1. Open `https://abc123.ngrok-free.app/app/` in Chrome
2. Click **Start** to begin the webcam stream
3. In the **Live Chat Room** card, click **Generate** to get a room code (e.g., `X7K2M9`)
4. Click **Join Room**
5. Share the room code with Device B

### Step 5 — Device B (Partner's Machine/Phone)

1. Open the same ngrok URL in Chrome: `https://abc123.ngrok-free.app/app/`
2. Click **Start** to begin their webcam stream
3. Enter the room code `X7K2M9` in the Live Chat Room input
4. Click **Join Room**
5. The peer status should change to **"Partner connected"** on both devices

### Step 6 — Test

- Both users sign in front of their cameras
- After 2 seconds of signing silence, NLP corrections appear in the AI Translation card
- Those same translations appear as chat bubbles in both users' chat feeds — right-aligned for your own, left-aligned for your partner's
- Verify the FPS counter shows no degradation with the room active

> **Note:** ngrok's free tier adds a browser interstitial on first visit. Click through it on both devices. For production use, consider a paid ngrok plan or deploy behind a proper reverse proxy.

---

## WebSocket Protocol Reference

### `/ws/inference/{session_id}` — ML Inference Pipeline

| Direction | Type | Fields | Description |
|-----------|------|--------|-------------|
| **Client → Server** | `frame` | `data` (Base64 JPEG string) | Webcam frame for inference |
| **Client → Server** | `config` | `target_language` (string) | Change the NLP translation language |
| **Server → Client** | `status` | `message`, `buffer` | Connection/buffering status updates |
| **Server → Client** | `prediction` | `label`, `confidence`, `buffer` | ML model prediction result |
| **Server → Client** | `nlp` | `corrected_english`, `translated_text`, `auto_complete[]`, `raw_gloss[]`, `target_language` | NLP-corrected sentence + translation + next-word predictions |

### `/ws/signaling/{room_id}` — Peer Relay

| Direction | Type | Fields | Description |
|-----------|------|--------|-------------|
| **Server → Client** | `room_info` | `session_id`, `peer_count` | Sent on connect with current room state |
| **Server → Peers** | `peer_left` | `session_id`, `remaining` | Broadcast when a peer disconnects |
| **Client → Server** | `peer_joined` | `peer_id` | Handshake sent by new peer on connect |
| **Client → Server** | `nlp_relay` | `from`, `corrected_english`, `translated_text`, `raw_gloss[]`, `target_language` | Forward own NLP result to partner |
| _Any_ | _Any JSON_ | _Any_ | Server relays all messages to other peers in the room |

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check (app name, version, status) |
| `GET` | `/health` | Simple health ping |
| `POST` | `/transcripts/` | Log a new sign detection |
| `GET` | `/transcripts/{session_id}` | Get ordered transcript for a session |
| `GET` | `/transcripts/{session_id}/summary` | Get session stats (count, unique labels, time range) |
| `GET` | `/signaling/rooms` | List all active signaling rooms and peer counts |

---

## Future Roadmap

### Phase 10 — 3D Avatar Visualization
Render a 3D human avatar that mirrors the user's detected landmarks in real time. The avatar would replay signs visually, serving as both a learning tool and an accessibility aid. Potential stack: Three.js or Babylon.js for WebGL rendering, driven by the same MediaPipe landmark stream already available on the frontend.

### Phase 11 — Live Learning (KNN Gesture Registration)
Allow users to teach the system new signs on-the-fly using K-Nearest Neighbors. A user would perform a new gesture 5–10 times, the system would store the keypoint sequences as exemplars, and at inference time a KNN classifier would run in parallel with the LSTM. This enables personalized sign vocabularies without retraining the neural network.

### Phase 12 — Domain Optimization
Fine-tune the model and NLP prompts for specific domains (medical, legal, educational) where sign language vocabulary and sentence structures differ significantly from casual conversation. Includes:
- Domain-specific WLASL subset training
- Custom Groq prompt templates per domain
- Contextual auto-complete tuned to domain vocabulary
- Specialized glossary management for technical terms

---

## Known Limitations & Challenges Solved

### Current Limitations

| Limitation | Detail |
|------------|--------|
| **5-class vocabulary** | Currently trained on `before`, `book`, `computer`, `drink`, `go`. Expandable by retraining with more WLASL videos. |
| **Single-user inference** | Each WebSocket connection loads the model singleton; concurrent users share the same MediaPipe Holistic instance. |
| **NLP latency** | Groq API calls add 200–500ms; mitigated by the 2-second debounce (NLP fires during natural signing pauses). |
| **No offline mode** | Requires network access for Groq API; inference itself runs locally but NLP does not. |
| **Browser-only input** | Webcam access requires HTTPS in production (ngrok handles this for dev). |

### Challenges Solved

| Challenge | Solution |
|-----------|----------|
| **TF graph re-tracing** caused frozen predictions in real-time loop | Replaced `model.predict()` with `model(tensor, training=False)` direct call |
| **Groq returning markdown-wrapped JSON** broke NLP parsing | Added regex stripping of ` ```json ``` ` fences before `json.loads()` |
| **NLP errors crashing the video loop** | Wrapped entire NLP block in `try/except` — NLP failures are logged but never interrupt inference |
| **Frame backpressure** causing increasing latency | Implemented queue draining — only the newest frame is processed per iteration |
| **Translation-dependent landmarks** (user position in frame) | Nose-relative coordinate normalization in `extract_keypoints()` |
| **Small dataset overfit** | 15× augmentation pipeline with noise, scaling, temporal warping, mirroring, and combined transforms |
| **Duplicate DB entries** for held signs | 2-second cooldown deduplication keyed on label name |
| **Room disconnect leaving stale UI** | `leaveRoom()` nulls the WS reference before closing to prevent recursive `onclose` → `leaveRoom` loops |

---

<p align="center">
  <strong>Vocalize</strong> — Breaking communication barriers, one sign at a time.
</p>
