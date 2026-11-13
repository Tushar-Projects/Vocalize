<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow 2.15" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/MediaPipe-Holistic-007FFF?logo=google&logoColor=white" alt="MediaPipe Holistic" />
  <img src="https://img.shields.io/badge/PostgreSQL-14+-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/Groq-Llama%203.1%208B-F05A28" alt="Groq Llama 3.1 8B" />
</p>

# Vocalize

## Project Overview

**What Vocalize is:**
Vocalize is an end-to-end, real-time sign language translation and communication platform. Operating directly through standard webcams and web browsers, it tracks body, face, and hand articulators, classifies continuous sign gestures into vocabulary labels, reconstructs conversational grammar using Large Language Models, translates across multiple spoken languages, and synchronizes conversations across remote participants via low-latency WebSockets.

**Why it exists:**
Bridging the communication barrier between deaf or hard-of-hearing signers and non-signing individuals requires more than basic static gesture recognition. Real sign language involves high-frequency dynamic motion, facial expressions, and distinct grammar rules that differ fundamentally from spoken syntax. Traditional systems often suffer from significant frame-processing lag, rigid vocabularies that cannot adapt to individual users, and clumsy user interfaces. Vocalize bridges this gap by unifying MediaPipe Holistic extraction, deep temporal sequence modeling, on-the-fly custom gesture learning via K-Nearest Neighbors (KNN), and cloud-accelerated Large Language Model (LLM) grammar reconstruction into an interactive, zero-backpressure web application.

**The problems it solves:**
- **High-Dimensional Spatial-Temporal Modeling:** Extracts 1,662 geometric keypoints per frame across 33 pose, 468 facial mesh, and 42 bilateral hand coordinates with nose-relative normalization to ensure full translation invariance.
- **Dynamic Sequence Classification:** Employs a stacked Bidirectional Long Short-Term Memory (Bi-LSTM) network coupled with a Bahdanau Attention mechanism to evaluate 30-frame temporal windows and focus on the most expressive frames.
- **Few-Shot Custom Gesture Learning:** Resolves rigid vocabulary limitations by allowing signers to record and register personalized signs directly in the browser, matching 128-dimensional attention embeddings in real time without retraining neural network weights.
- **Gloss to Natural Language Reconstruction:** Converts isolated sign detections into fluent, grammatically correct sentences, provides real-time translation into five target languages, and generates predictive auto-complete suggestions using Groq-hosted Llama 3.1 8B.
- **Zero-Backpressure Streaming Pipeline:** Solves client-server frame congestion through an asynchronous WebSocket queue-draining pattern that discards stale video frames and processes only the latest frame at 15 FPS.
- **Modular Multi-Window Interface:** Replaces static video pages with a draggable, resizable, canvas-based Window Manager featuring grid snapping, localStorage state persistence, dark and light themes, and peer-to-peer live chat rooms.

---

## Key Features

- **Continuous Real-Time Translation:** Ingests live webcam streams at 15 frames per second, processing 30-frame sequences with sub-second response times.
- **Robust Landmark Extraction:** MediaPipe Holistic pipeline extracts 1,662 coordinates per frame (pose, face, left hand, right hand), normalized relative to the nose tip for scale and positional stability.
- **Attention-Enhanced Sequence Classifier:** Two-layer Bidirectional LSTM architecture with custom Bahdanau additive attention trained on the World-Level American Sign Language (WLASL) dataset.
- **15-Gloss Baseline Vocabulary:** Out-of-the-box recognition covering fifteen essential conversational signs: basketball, before, book, candy, computer, drink, family, go, help, no, tall, thin, what, who, and yes.
- **Live Custom Sign Registration (KNN):** Few-shot personalized gesture registration. Users perform any custom sign across five repetitions with automated hand visibility validation; the system averages and normalizes the attention layer embeddings for immediate real-time recognition.
- **Groq-Accelerated NLP Engine:** Automatic grammar reconstruction using Llama 3.1 8B with a 2-second signing silence debounce timer. Reconstructs sign glosses into coherent English sentences.
- **Multi-Language Translation:** Direct translation of reconstructed sentences into English, Hindi, Kannada, Tamil, and Spanish.
- **Context-Aware Next-Word Auto-Complete:** Generates three predictive next-word suggestions based on current conversation context to accelerate interaction.
- **Canvas-Based Floating Window Interface:** Custom Window Manager providing draggable, resizable, and closable panels on a 24px grid canvas with edge snapping, z-ordering, and local persistence.
- **Live Peer-to-Peer Chat Rooms:** Room-based WebSocket signaling relay supporting six-character room codes to exchange translated dialogue between remote conversation partners.
- **Spoken Voice Input:** Integrated Web Speech API enables non-signing partners to respond with browser-based voice transcription.
- **Session-Isolated Persistent Logging:** PostgreSQL database backend stores verified detections with 2-second cooldown deduplication and indexed session replay capabilities.
- **End-to-End Latency Benchmarking:** Dedicated instrumentation suite (`benchmark_latency.py`) to profile cold start, MediaPipe extraction, LSTM inference, database commits, WebSocket round-trips, and NLP response times.

---

## Architecture

Vocalize links client-side video capture, server-side computer vision, deep temporal classification, few-shot metric learning, cloud NLP, and peer communication into a cohesive real-time loop.

### End-to-End System Pipeline

```mermaid
%%{init: {
  'theme': 'base',
  'flowchart': { 'curve': 'linear', 'nodeSpacing': 70, 'rankSpacing': 90, 'padding': 24, 'htmlLabels': true },
  'themeVariables': {
    'background': '#0d1117',
    'primaryColor': '#1e293b',
    'primaryTextColor': '#f8fafc',
    'primaryBorderColor': '#64748b',
    'lineColor': '#94a3b8',
    'textColor': '#f1f5f9',
    'edgeLabelBackground': '#0d1117',
    'fontFamily': 'Helvetica Neue, Arial, sans-serif',
    'fontSize': '17px'
  }
}}%%
flowchart TB
    subgraph Tier1 [" "]
        L1["<b>1</b>&nbsp;&nbsp;&nbsp;CLIENT<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PRESENTATION<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;LAYER<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"]
        S1a[" "]
        UI["Web Workspace &amp;<br/>Media Capture Controller"]
        S1b[" "]
    end

    subgraph Tier2 [" "]
        L2["<b>2</b>&nbsp;&nbsp;&nbsp;API GATEWAY<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&amp; ROUTING<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;LAYER<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"]
        SIG["Real-Time<br/>Signaling Service"]
        INF["Gesture<br/>Inference Gateway"]
        TRN["Transcript &amp;<br/>Catalog Service"]
    end

    subgraph Tier3 [" "]
        L3["<b>3</b>&nbsp;&nbsp;&nbsp;MACHINE<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;LEARNING<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&amp; NLP LAYER<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"]
        PREP["Landmark &amp;<br/>Sequence Pipeline"]
        ATTN["Temporal Attention<br/>Classifier &amp; Embeddings"]
        NLP["NLP Engine<br/>(Groq LLM)"]
    end

    subgraph Tier4 [" "]
        L4["<b>4</b>&nbsp;&nbsp;&nbsp;DATA<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PERSISTENCE<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;LAYER<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"]
        S4a[" "]
        DB[("Relational Database &amp; ORM<br/>Transcripts · Custom Signs")]
        S4b[" "]
    end

    L1 ~~~ L2 ~~~ L3 ~~~ L4
    S1a ~~~ SIG
    S1b ~~~ TRN
    PREP ~~~ S4a
    NLP ~~~ S4b

    %% Tier 1 ↔ Tier 2
    UI <-->|"Chat &amp; Sentence Relay"| SIG
    UI <-->|"JPEG Frames ⇄ Predictions"| INF
    UI <-->|"History &amp; Catalog"| TRN

    %% Tier 2 → Tier 3
    INF -->|"Frame Decode &amp; Keypoints"| PREP
    INF -->|"Keypoint Sequences"| ATTN
    INF <-->|"Gloss ⇄ Sentence"| NLP

    %% Tier 3 / 2 → Tier 4
    ATTN -->|"Custom Sign Embeddings"| DB

    style Tier1 fill:#161f2c,stroke:#334155,stroke-width:1.5px
    style Tier2 fill:#161f2c,stroke:#334155,stroke-width:1.5px
    style Tier3 fill:#161f2c,stroke:#334155,stroke-width:1.5px
    style Tier4 fill:#161f2c,stroke:#334155,stroke-width:1.5px

    classDef academicNode fill:#1e293b,stroke:#60a5fa,stroke-width:1.5px,color:#f8fafc
    classDef storageNode fill:#1e293b,stroke:#94a3b8,stroke-width:1.5px,color:#f8fafc
    classDef layerLabel fill:none,stroke:none,color:#cbd5e1,font-size:20px,text-align:left
    classDef spacer fill:none,stroke:none,color:#161f2c
    class UI,SIG,INF,TRN,PREP,ATTN,NLP academicNode
    class DB storageNode
    class L1,L2,L3,L4 layerLabel
    class S1a,S1b,S4a,S4b spacer

```

### Dual WebSocket Channel Architecture

The platform separates high-throughput computer vision processing from conversational relay through two isolated WebSocket endpoints:

1. **Inference Channel (`/ws/inference/{session_id}?client_id={client_id}`)**:
   Dedicated per-client machine learning pipeline. Ingests raw video frames, runs MediaPipe Holistic landmark extraction, updates rolling sequence buffers, performs Bi-LSTM classification, queries client-specific KNN custom gesture prototypes, deduplicates entries into PostgreSQL, and executes debounced Groq LLM queries.

2. **Signaling Channel (`/ws/signaling/{room_id}`)**:
   Stateless peer-to-peer message relay. Facilitates room coordination using six-character codes, managing join/leave lifecycles and broadcasting translated messages (`nlp_relay`) between participants without placing computational load on the inference loop.

---

## Project Structure

```text
Vocalize/
├── .env                        # Local environment variables (DB URL, Groq API key)
├── .gitignore                  # Git exclusions for models, datasets, logs, and venv
├── .python-version             # Python version pin (3.11)
├── attention.py                # Bahdanau additive attention Keras layer
├── augment_data.py             # Skeleton data augmentation pipeline (16x expansion)
├── benchmark_latency.py        # End-to-end performance and latency diagnostic tool
├── dataset_processor.py        # Batch video landmark extractor producing .npy arrays
├── landmark_extractor.py       # MediaPipe Holistic keypoint extraction and normalizer
├── README.md                   # Project documentation and architectural specification
├── realtime_inference.py       # Standalone OpenCV desktop inference script with HUD
├── train_model.py              # Bi-LSTM with Attention model training script
├── wlasl_parser.py             # WLASL v0.3 dataset parser and video staging utility
│
├── backend/                    # FastAPI backend service
│   ├── __init__.py             # Python package marker
│   ├── config.py               # Application settings and environment validation
│   ├── database.py             # SQLAlchemy database engine and session factory
│   ├── embedding_model.py      # Feature extractor exposing 128-dim attention embeddings
│   ├── main.py                 # Application entry point, lifespan, CORS, static routes
│   ├── models.py               # ORM schemas (transcripts, custom_gestures)
│   ├── nlp_engine.py           # Async Groq LLM client for grammar and translation
│   ├── schemas.py              # Pydantic request and response schemas
│   ├── wlasl_15_gloss.keras    # Trained Bi-LSTM + Attention model weights
│   ├── wlasl_15_gloss_labels.json # Mapping of class indices to 15 gloss names
│   └── routers/
│       ├── __init__.py         # Package marker
│       ├── gestures.py         # REST endpoints for custom sign management
│       ├── inference.py        # WebSocket /ws/inference endpoint and hybrid engine
│       ├── signaling.py        # WebSocket /ws/signaling peer-to-peer room router
│       └── transcripts.py      # REST CRUD and summary endpoints for session logs
│
├── frontend/                   # Vanilla JavaScript modular web client
│   ├── app.js                  # Frontend controllers, webcam streamer, WebSocket client
│   ├── index.html              # HTML structure containing floating window templates
│   ├── style.css               # Design system, glassmorphism UI, light/dark themes
│   └── window-manager.js       # Window manager with grid snapping and persistence
│
├── latency_results/            # Automated benchmark exports (JSON reports, raw CSVs)
├── dataset/                    # Staged video files organized by gloss (gitignored)
├── processed_data/             # Keypoint sequences in .npy format (gitignored)
├── logs/                       # TensorBoard training logs and checkpoints (gitignored)
└── output/                     # Diagnostic keypoint frame dumps (gitignored)
```

---

## Technology Stack

- **Core Runtime:** Python 3.10 / 3.11 environment.
- **Deep Learning Framework:** TensorFlow 2.15 (Keras Functional API) utilizing Masking, Bidirectional LSTMs, and custom attention mechanisms.
- **Computer Vision:** MediaPipe 0.10+ (Holistic solution tracking 1,662 facial, pose, and hand coordinates) and OpenCV for image decoding and color space conversion.
- **Natural Language Processing:** Groq Cloud API running `llama-3.1-8b-instant` with structured JSON constraint prompting.
- **Backend Framework:** FastAPI with Uvicorn ASGI server and Pydantic Settings for runtime configuration.
- **Database Layer:** PostgreSQL with SQLAlchemy 2.0 ORM, storing transcripts and vector arrays (`ARRAY(Float)`).
- **Frontend Architecture:** Vanilla JavaScript (ES6+), HTML5 Canvas, WebSockets, Web Speech API, and CSS Grid/Flexbox with no build step required.
- **Data Engineering:** NumPy, scikit-learn, and tqdm for dataset extraction, sequence padding, and temporal augmentation.

---

## Installation

### Prerequisites

- **Python:** Version 3.10 or 3.11 installed.
- **PostgreSQL:** Version 14 or higher running locally or accessible via network.
- **Groq API Key:** A valid key obtained from [Groq Console](https://console.groq.com).
- **Webcam:** An integrated or USB camera for video capture.
- **Modern Web Browser:** Google Chrome, Brave, or Microsoft Edge (recommended for Web Speech API and WebSockets).

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/vocalize.git
   cd vocalize/Vocalize
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv

   # Windows PowerShell:
   .\venv\Scripts\Activate.ps1

   # Linux or macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install opencv-python mediapipe numpy tensorflow scikit-learn fastapi "uvicorn[standard]" sqlalchemy psycopg2-binary pydantic-settings python-dotenv groq tqdm websockets
   ```

4. **Initialize PostgreSQL database:**
   Connect to PostgreSQL using `psql` or your database manager and create the application database:
   ```sql
   CREATE DATABASE vocalize_db;
   ```
   All tables (`transcripts` and `custom_gestures`) are automatically generated on initial startup by FastAPI.

5. **Configure environment variables:**
   Create a `.env` file in the project root:
   ```env
   DATABASE_URL=postgresql://postgres:your_password@localhost:5432/vocalize_db
   GROQ_API_KEY=gsk_your_groq_api_key_here
   DEBUG=True
   CORS_ORIGINS=http://localhost:3000,http://localhost:5173,http://127.0.0.1:8000
   ```

---

## Configuration

### Environment Configuration Reference

| Parameter | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `DATABASE_URL` | Yes | `postgresql://postgres:postgres@localhost:5432/vocalize_db` | Connection string for PostgreSQL database |
| `GROQ_API_KEY` | Yes | None | Authentication key for Groq LLM API requests |
| `DEBUG` | No | `True` | Enables SQL query echo and verbose diagnostic logging |
| `CORS_ORIGINS` | No | `http://localhost:3000,http://localhost:5173,http://127.0.0.1:8000` | Comma-separated list of permitted browser origins |

### Model Weights and Class Mappings

Vocalize requires trained model weights and corresponding class labels placed in the `backend/` directory:
- `backend/wlasl_15_gloss.keras`: Serialized Keras model containing the Bi-LSTM layers and custom `AttentionLayer`.
- `backend/wlasl_15_gloss_labels.json`: JSON dictionary mapping integer indices (0 through 14) to sign names.

Baseline 15-gloss classes included:
`0: basketball`, `1: before`, `2: book`, `3: candy`, `4: computer`, `5: drink`, `6: family`, `7: go`, `8: help`, `9: no`, `10: tall`, `11: thin`, `12: what`, `13: who`, `14: yes`.

---

## Usage

### 1. Launching the Application

Start the FastAPI application with Uvicorn:
```bash
uvicorn backend.main:app --reload --port 8000
```
Open your browser and navigate to:
```text
http://127.0.0.1:8000/app/
```

### 2. Interface Workflow

```text
+---------------------------------------------------------------------------------+
| Vocalize Toolbar      [+ Add Node]  [Reset Layout]     [Connected]   [Theme Toggle] |
+---------------------------------------------------------------------------------+
|  +-----------------------------+  +-------------------+  +--------------------+ |
|  | CH.01 WEBCAM                |  | CH.02 SESSION     |  | CH.05 CUSTOM SIGNS | |
|  | [Live Video Display]        |  | Session: 4f1a..   |  | [Sign Name Input]  | |
|  | Buffer: [............] 30/30|  | Detections: 18    |  | [Record 5 Reps]    | |
|  | Confidence: [======] 85%    |  | Unique: 6         |  | Active Custom List | |
|  | Prediction: "computer"      |  +-------------------+  +--------------------+ |
|  | [Start] [Stop] [Mic] 15 FPS |  | CH.03 TRANSLATION |  +--------------------+ |
|  +-----------------------------+  | Lang: Spanish     |  | CH.06 LIVE CHAT    | |
|  +-----------------------------+  | "Estoy usando la  |  | Room: [X7K2M9]     | |
|  | CH.07 SPOKEN INPUT          |  |  computadora."    |  | Status: Connected  | |
|  | Non-signer speech to text   |  +-------------------+  | Chat Conversation  | |
|  +-----------------------------+  | CH.04 PREDICTIONS |  | Feed Bubbles       | |
|                                   | [work] [code] ... |  +--------------------+ |
+---------------------------------------------------------------------------------+
```

1. **Start Streaming:** Click **Start** in the Webcam window. Allow browser webcam access. The frame buffer indicator will fill to 30 frames.
2. **Perform Signs:** Perform any of the 15 supported signs clearly within camera view. Detected labels and confidence bars update in real time.
3. **Automatic Grammar Correction:** When you pause signing for two seconds, the system gathers accumulated glosses and queries Groq. Reconstructed English and target translations appear in the AI Translation window.
4. **Select Target Language:** Use the dropdown in CH.03 to switch output between English, Hindi, Kannada, Tamil, or Spanish on the fly.
5. **Contextual Auto-Complete:** Click any of the three suggested next-word prediction pills in CH.04 to review expected follow-up vocabulary.

### 3. Registering Custom Signs (KNN Live Learning)

Users can add custom signs without retraining the core neural network:
1. Open the **CH.05 CUSTOM SIGNS** window.
2. Type a unique name in the sign label field (for example, `my_custom_sign`).
3. Click **Record**. The status will prompt you for five consecutive repetitions.
4. Perform the gesture once per repetition prompt. An automated quality gate validates that hand landmarks are visible in at least 60% of the frames.
5. After five valid repetitions, the system computes the 128-dimensional embedding from the attention layer, L2-normalizes the average prototype, and saves it to PostgreSQL.
6. The sign is instantly active. When performed, the hybrid engine recognizes it, displays a **Custom Sign** badge, and routes the label into the Groq translation flow.

### 4. Live Chat Room (Two-Device Communication via ngrok)

To communicate between two separate computers or mobile devices across the internet:

1. **Start the local server:**
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

2. **Expose via ngrok:**
   ```bash
   ngrok http 8000
   ```
   Copy the generated forwarding URL (for example, `https://abc123.ngrok-free.app`).

3. **Device A Setup:**
   - Open `https://abc123.ngrok-free.app/app/` in Chrome.
   - Click **Start** to initiate video capture.
   - In **CH.06 LIVE CHAT**, click **Generate** to create a 6-character room code (e.g., `K9M2X4`), then click **Join Room**.

4. **Device B Setup:**
   - Open the same ngrok URL on the second device.
   - Click **Start** to activate their webcam.
   - Enter room code `K9M2X4` and click **Join Room**.

5. **Bidirectional Interaction:**
   - When either participant signs, their translated sentence is automatically broadcast over the signaling channel.
   - Sent messages appear as right-aligned gradient bubbles; incoming partner translations appear as left-aligned cards.
   - Non-signing partners can click the **Mic** button to speak their response, rendering spoken text into the shared space.

### 5. Running the Latency Benchmark Suite

To measure real-time performance on a new computer or verify hardware latency:
```bash
# Benchmark using live webcam for 45 seconds:
python benchmark_latency.py --source 0 --duration 45

# Benchmark using a recorded video file:
python benchmark_latency.py --source path/to/sample.mp4 --duration 45
```
The script instruments all backend components, creates an isolated test session, executes live WebSocket transactions, removes temporary test records from the database, and exports detailed reports into `latency_results/` in both JSON and CSV formats.

---

## Model Training and Pipeline Scripts

If you wish to retrain the neural network with additional classes or modify the augmentation parameters:

### Step 1: Parse WLASL Dataset
Download the WLASL dataset and extract desired glosses into organized subdirectories:
```bash
python wlasl_parser.py --json_path path/to/WLASL_v0.3.json --videos_dir path/to/videos --words "basketball,before,book,candy,computer,drink,family,go,help,no,tall,thin,what,who,yes"
```

### Step 2: Extract Keypoints
Batch process video files through MediaPipe Holistic to extract 1,662 keypoints per frame:
```bash
python dataset_processor.py --input_dir dataset --output_dir processed_data
```

### Step 3: Augment Keypoint Sequences
Expand small datasets by generating 15 augmented variations per sample (16x total expansion):
```bash
python augment_data.py --data_dir processed_data --num_augments 15
```
Augmentation operations applied:
- Additive Gaussian noise (three distinct variance levels)
- Coordinate scaling (simulating varied camera distances)
- Temporal shifting (forward and backward sequence rolling)
- Temporal scaling (linear interpolation simulating varying signing speeds)
- Frame dropout (simulating dropped webcam frames)
- Horizontal coordinate mirroring with bilateral hand index swapping

### Step 4: Train Bi-LSTM with Attention
Train the classification model with EarlyStopping, ModelCheckpoint, and TensorBoard monitoring:
```bash
python train_model.py --data_dir processed_data --epochs 200 --batch_size 16 --model_out backend/wlasl_15_gloss.keras
```

---

## Command and Protocol Reference

### Command Line Tools

| Command | File | Description |
| :--- | :--- | :--- |
| `uvicorn backend.main:app --reload` | `backend/main.py` | Starts the FastAPI development server with hot-reload |
| `python realtime_inference.py` | `realtime_inference.py` | Launches standalone OpenCV desktop inference with on-screen HUD |
| `python benchmark_latency.py` | `benchmark_latency.py` | Executes the end-to-end automated latency and throughput test |
| `python dataset_processor.py` | `dataset_processor.py` | Batch extracts MediaPipe landmarks from MP4 files to NumPy `.npy` |
| `python augment_data.py` | `augment_data.py` | Performs 16x time-series data augmentation on landmark sequences |
| `python train_model.py` | `train_model.py` | Builds and trains the Bi-LSTM model with Bahdanau attention |
| `python wlasl_parser.py` | `wlasl_parser.py` | Filters WLASL JSON annotations and stages videos by class |

### WebSocket Protocol Reference

#### `/ws/inference/{session_id}`: Machine Learning Inference Channel

| Direction | Message Type | Key Payload Fields | Description |
| :--- | :--- | :--- | :--- |
| Client to Server | `frame` | `data` (Base64 JPEG string) | Real-time video frame for inference |
| Client to Server | `config` | `target_language` (string) | Changes target translation language |
| Client to Server | `record_gesture_start` | `label` (string) | Initiates 5-repetition custom gesture recording |
| Client to Server | `record_gesture_stop` | None | Cancels active custom gesture recording |
| Server to Client | `status` | `message`, `buffer` | Pipeline initialization and buffer progress |
| Server to Client | `prediction` | `label`, `confidence`, `buffer`, `source` | Model prediction (`source: "model"` or `"custom"`) |
| Server to Client | `nlp` | `corrected_english`, `translated_text`, `auto_complete[]`, `raw_gloss[]`, `target_language` | Grammar-corrected sentence, translation, and auto-complete |
| Server to Client | `gesture_rep_captured` | `rep`, `of` | Confirms successful capture of a repetition |
| Server to Client | `gesture_rep_rejected` | `reason` | Notifies user that repetition lacked hand visibility |
| Server to Client | `gesture_recorded` | `label`, `samples` | Confirms custom sign saved to database and cache |

#### `/ws/signaling/{room_id}`: Peer-to-Peer Signaling Channel

| Direction | Message Type | Key Payload Fields | Description |
| :--- | :--- | :--- | :--- |
| Server to Client | `room_info` | `session_id`, `peer_count` | Dispatched on connection with current peer count |
| Server to Peers | `peer_left` | `session_id`, `remaining` | Broadcast when a peer disconnects from the room |
| Client to Server | `peer_joined` | `peer_id` | Initial handshake sent upon room entry |
| Client to Server | `nlp_relay` | `from`, `corrected_english`, `translated_text`, `raw_gloss[]`, `target_language` | Forwards own translation results to partner feed |
| Any | Any JSON | Arbitrary fields | Server relays message to all other peers in room |

### REST Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check returning service status, title, and version |
| `GET` | `/health` | Lightweight uptime health verification endpoint |
| `POST` | `/transcripts/` | Persists a validated sign detection to PostgreSQL |
| `GET` | `/transcripts/{session_id}` | Retrieves ordered transcript records for a session |
| `GET` | `/transcripts/{session_id}/summary` | Aggregates detection counts, unique signs, and timestamps |
| `GET` | `/signaling/rooms` | Lists active chat rooms and connected peer counts |
| `GET` | `/gestures/{client_id}` | Lists all registered custom signs for a client |
| `DELETE` | `/gestures/{gesture_id}` | Deletes a custom sign and invalidates in-memory KNN cache |

---

## Performance and Latency Profile

Based on empirical runs executed via `benchmark_latency.py` on standard modern desktop hardware:

| Subsystem Component | Metric Measured | Typical Duration | Notes |
| :--- | :--- | :--- | :--- |
| **Server Cold Start** | Boot & Database Connect | ~240 ms to 260 ms | Single execution at server launch |
| **Model Warmup** | Bi-LSTM & Embedding Load | ~1.8 s to 2.1 s | One-time lazy-loading on first connection |
| **MediaPipe Holistic** | Landmark Extraction | ~35 ms to 50 ms | Per processed video frame |
| **Sequence Classifier** | Bi-LSTM + Attention Forward Pass | ~10 ms to 15 ms | Executed when 30-frame buffer is full |
| **KNN Embedding Match** | Attention Vector Extraction & Distance | ~2 ms to 4 ms | Evaluates candidate prototypes in memory |
| **Database Commit** | PostgreSQL Transaction | ~3 ms to 5 ms | Synchronous write during confident detections |
| **Groq NLP Inference** | Llama 3.1 8B API Round-Trip | ~300 ms to 500 ms | Triggered after 2.0s signing silence |
| **Local Peer Relay** | WebSocket Loopback Broadcast | < 1 ms | In-memory connection manager dispatch |

Frame queue draining ensures that processing delays never accumulate; incoming frames that arrive while MediaPipe is computing are dropped, guaranteeing real-time responsiveness at all times.

---

## Engineering Challenges and Solutions

- **TensorFlow Graph Retracing Memory Leak:**
  *Problem:* Calling `model.predict()` inside high-frequency real-time loops caused TensorFlow to retrace execution graphs, steadily increasing memory consumption and introducing severe video stutter.
  *Solution:* Replaced `model.predict()` with direct callable invocation `model(tensor, training=False).numpy()`, executing graph operations without retracing overhead.

- **WebSocket Frame Backpressure:**
  *Problem:* When client camera streams exceeded server processing capacity, queued frames created escalating latency, causing predictions to lag several seconds behind real physical motion.
  *Solution:* Implemented an asynchronous queue-draining pattern in the inference WebSocket loop that clears pending messages and processes only the most recent frame.

- **Positional and Scale Sensitivity in Video Capture:**
  *Problem:* Landmark coordinates varied drastically depending on user distance and position relative to the camera lens.
  *Solution:* Implemented nose-relative coordinate normalization in `landmark_extractor.py`, subtracting the nose coordinate (`keypoints[0]`) from all pose and facial landmarks to ensure translation invariance.

- **Noisy or Degenerate Custom Sign Recording:**
  *Problem:* Users recording custom signs occasionally stepped out of view or lowered their hands, corrupting the embedding prototype with zero-vector noise.
  *Solution:* Added an automated quality gate requiring hand landmarks to be actively tracked in at least 60% of recorded frames. Failed repetitions are rejected with immediate user feedback.

- **Race Conditions in Custom Sign Invalidation:**
  *Problem:* Deleting a custom sign via a REST endpoint while the async inference loop was reading candidate embeddings risked collection mutation errors across threads.
  *Solution:* Handled cache invalidation via atomic key deletion rather than in-place mutation, guaranteeing that inference loops safely read consistent candidate snapshots.

- **Markdown Fences in LLM Responses:**
  *Problem:* Groq occasionally returned JSON payloads wrapped inside triple-backtick markdown blocks, causing `json.loads` deserialization errors.
  *Solution:* Applied defensive regex pre-processing to strip markdown wrappers before parsing, backed by graceful fallbacks that prevent NLP failures from interrupting the video inference loop.

- **Held Sign Database Duplication:**
  *Problem:* Users holding a sign steadily triggered dozens of consecutive identical database entries every second.
  *Solution:* Implemented a 2.0-second cooldown deduplication filter tied to the detected sign label, logging new entries only when the sign changes or when the cooldown window elapses.

---

## Roadmap

- **Phase 10: 3D Avatar Visualization:** Render a 3D avatar using WebGL (Three.js or Babylon.js) driven by MediaPipe landmark streams to visually mirror and replay signs for bidirectional accessibility.
- **Phase 12: Domain Optimization:** Fine-tune specialized vocabulary models and customized Groq prompting for medical, emergency, and legal environments.
- **WebRTC Data Channels:** Transition live peer-to-peer chat feeds from WebSocket server relays to direct browser-to-browser WebRTC DataChannels for reduced network latency.
- **Local On-Device LLM Fallback:** Integrate quantized local models (such as small GGUF models) to enable grammar correction in fully air-gapped or offline environments.

---

## License

No license has been selected for this project yet.
