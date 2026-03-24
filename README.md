# Vocalize

**Real-Time Sign Language Translation for the Modern Web**

Vocalize is a full-stack, browser-based sign language translator that converts live hand and body gestures into natural, grammatically correct text -- in multiple languages -- using nothing more than a standard webcam. There are no gloves, no depth sensors, and no specialized hardware. A user signs in front of their laptop, and the system interprets, corrects, translates, and displays the meaning in real time.

The project was built to address a fundamental communication barrier: the vast majority of the hearing population does not understand sign language. Vocalize aims to bridge that gap by making translation instantaneous, accessible, and free of friction.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Technical Stack](#technical-stack)
- [System Pipeline](#system-pipeline)
- [Key Engineering Decisions](#key-engineering-decisions)
- [How to Run Locally](#how-to-run-locally)
- [Project Structure](#project-structure)
- [Future Roadmap](#future-roadmap)
- [License](#license)

---

## Architecture Overview

Vocalize operates as a five-stage pipeline that transforms raw webcam pixels into translated, human-readable sentences. Each stage is purpose-built and optimized for low-latency, real-time performance.

```
Webcam (15 FPS)
    |
    v
[1] MediaPipe Holistic --- extracts 1,662 spatial coordinates per frame
    |
    v
[2] Bi-LSTM + Attention --- reads 30-frame sequences, predicts sign 'gloss'
    |
    v
[3] FastAPI WebSocket --- event-driven backend, processes frames with near-zero latency
    |
    v
[4] PostgreSQL --- persistent, deduplicated session transcript logging
    |
    v
[5] Groq LLM (Llama 3.1) --- single-pass grammar correction + multi-language translation
    |
    v
Browser UI --- displays translated sentence, English subtitle, next-word predictions
```

---

## Technical Stack

| Layer | Technology | Role |
|-------|-----------|------|
| Computer Vision | MediaPipe Holistic | Extracts pose, face, and hand landmarks from each video frame |
| Machine Learning | TensorFlow / Keras | Custom 2M-parameter Bi-LSTM with learned Attention mechanism |
| Backend Server | FastAPI + Uvicorn | Async WebSocket server handling real-time frame ingestion |
| Database | PostgreSQL + SQLAlchemy | Session-level transcript persistence with indexed queries |
| NLP Engine | Groq API (Llama-3.1-8b-instant) | Grammar correction, auto-complete, and multi-language translation |
| Frontend | Vanilla HTML/CSS/JS | Responsive dark-theme UI with live video, prediction overlay, and transcript sidebar |

---

## System Pipeline

### Stage 1 -- Spatial Coordinate Extraction

Every webcam frame is passed through Google's MediaPipe Holistic model, which simultaneously detects 33 pose landmarks, 468 face mesh points, and 21 landmarks on each hand. This yields 1,662 floating-point values per frame.

To make the system **translation-invariant** (meaning the model produces the same output regardless of where the user is sitting in the frame), all coordinates are made relative to the nose landmark. The nose acts as the origin point, and every other landmark is expressed as an offset from it. This eliminates positional bias and significantly improves accuracy.

### Stage 2 -- Sequence Classification (Bi-LSTM + Attention)

Sign language is inherently temporal. A single frame is meaningless; the gesture only makes sense as a sequence of movements over time. Vocalize captures a rolling window of 30 frames (approximately 2 seconds at 15 FPS) and feeds them into a custom neural network.

The model architecture:

- **Bidirectional LSTM layers** process the sequence in both forward and reverse temporal order, capturing context from both the beginning and end of a gesture.
- A **custom Attention layer** (Bahdanau-style additive attention) learns which specific frames within the 30-frame window carry the most discriminative information. Not all frames are equally important; the model learns to focus on the frames where the hands are in their most distinctive positions.
- **Dropout regularization** (30%) prevents the model from memorizing training data.

The model contains approximately 2 million trainable parameters and outputs a probability distribution over the known sign vocabulary. Only predictions exceeding 60% confidence are accepted.

### Stage 3 -- Event-Driven Backend

The FastAPI server communicates with the browser over a persistent WebSocket connection. This is not a request-response API; it is a continuous, bidirectional data stream.

The backend implements several performance-critical optimizations:

- **Frame draining**: If the client sends frames faster than the server can process them (which happens frequently), the server discards all stale frames in the buffer and only processes the most recent one. This prevents cascading latency.
- **Database deduplication**: A 2-second cooldown prevents the same detected word from being logged to the database repeatedly. This keeps the transcript clean and meaningful.
- **Lazy model loading**: The TensorFlow model and MediaPipe instance are loaded once on first use and reused across all sessions, avoiding repeated initialization overhead.

### Stage 4 -- Persistent Session Logging

Every high-confidence detection is written to a PostgreSQL database with a session identifier, timestamp, confidence score, and sequence index. The frontend fetches this transcript via a REST endpoint and renders it chronologically in the sidebar.

Database columns are explicitly indexed on `session_id` and `created_at` for fast retrieval, even as the transcript grows over a long session.

### Stage 5 -- NLP and Multi-Language Translation

Raw sign language does not follow English grammar. A user might sign "I GO SHOP" to mean "I am going to the shop." Vocalize uses the Groq API (running Meta's Llama-3.1-8b-instant model) to perform three tasks in a single LLM call:

1. **Grammar correction**: Reconstruct the raw gloss sequence into a natural English sentence.
2. **Auto-complete**: Predict the three most likely next words the user might sign, based on conversational context.
3. **Translation**: Translate the corrected sentence into the user's selected language (Hindi, Kannada, Tamil, Spanish, or English).

This is executed as a **single-pass operation** to avoid the latency penalty of chaining multiple API calls. The LLM receives the raw words and the target language in one request and returns all three outputs simultaneously.

A **2-second debounce timer** governs when the NLP engine fires. The system waits until the user pauses signing for 2 seconds, then sends the accumulated word sequence to Groq. This prevents unnecessary API calls during continuous signing and ensures the LLM receives complete phrases rather than individual words.

---

## Key Engineering Decisions

**Why relative coordinates instead of absolute?**
Absolute pixel coordinates encode the user's position in the camera frame, which is irrelevant to the meaning of a sign. By subtracting the nose position from every landmark, the data becomes position-independent. A user signing "hello" in the top-left corner of the frame produces the same feature vector as one signing in the center.

**Why Bidirectional LSTM over a Transformer?**
For 30-frame sequences with 1,662 features, a Transformer's self-attention mechanism would require significantly more memory and compute. Bidirectional LSTMs are well-suited to short, sequential data and train efficiently on consumer hardware. The custom Attention layer provides the key benefit of Transformers (learned frame importance) without the overhead.

**Why WebSockets instead of REST?**
REST would require the client to make 15 HTTP requests per second, each with full headers and connection overhead. A single persistent WebSocket connection eliminates this entirely and allows the server to push results back to the client without being polled.

**Why a debounce timer for NLP?**
Without debouncing, the system would call the Groq API on every single high-confidence detection (potentially multiple times per second). The 2-second silence window batches words into coherent phrases, reduces API costs, and produces better translations because the LLM receives more context.

---

## How to Run Locally

### Prerequisites

- Python 3.10 or later
- PostgreSQL (with a database named `vocalize_db` created)
- A Groq API key (free tier available at [console.groq.com](https://console.groq.com))
- A webcam

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/Tushar-Projects/Vocalize.git
cd Vocalize
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

The core dependencies include: FastAPI, Uvicorn, SQLAlchemy, psycopg2-binary, python-dotenv, groq, tensorflow, mediapipe, and opencv-python.

4. **Configure environment variables**

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/vocalize_db
GROQ_API_KEY=your_groq_api_key_here
```

5. **Start the server**

```bash
uvicorn backend.main:app --reload --port 8000
```

The server will automatically create the required database tables on first boot.

6. **Open the application**

Navigate to [http://localhost:8000/app/](http://localhost:8000/app/) in a web browser. Grant camera access when prompted, press **Start**, and begin signing.

---

## Project Structure

```
Vocalize/
|-- backend/
|   |-- main.py              # FastAPI application entry point
|   |-- config.py             # Pydantic-settings configuration loader
|   |-- database.py           # SQLAlchemy engine and session factory
|   |-- models.py             # Transcript ORM model
|   |-- schemas.py            # Pydantic request/response schemas
|   |-- nlp_engine.py         # Groq-powered grammar correction and translation
|   |-- routers/
|       |-- inference.py      # WebSocket endpoint for real-time inference
|       |-- transcripts.py    # REST endpoints for transcript history
|       |-- signaling.py      # WebRTC signaling scaffold
|
|-- frontend/
|   |-- index.html            # Application UI
|   |-- app.js                # Client-side webcam capture and WebSocket logic
|
|-- landmark_extractor.py     # MediaPipe Holistic keypoint extraction
|-- attention.py              # Custom Keras Attention layer
|-- train_model.py            # Bi-LSTM + Attention training script
|-- realtime_inference.py     # Standalone webcam inference utility
|-- augment_data.py           # Data augmentation pipeline
|-- dataset_processor.py      # Dataset extraction and preprocessing
|-- requirements.txt          # Python dependencies
|-- .env                      # Environment variables (not committed)
```

---

## Future Roadmap

- **Reverse Pipeline (Speech-to-Sign)**: Integrate the Web Speech API to capture spoken English in real time and convert it into sign language instructions, enabling truly bidirectional communication.

- **3D Signing Avatar**: Build an animated avatar using Three.js or Unity WebGL that performs sign language gestures corresponding to spoken input. This would allow a hearing person to "speak" to a deaf user through a virtual signer on screen.

- **Real-Time Custom Sign Training**: Implement a K-Nearest Neighbors (KNN) classifier that allows users to teach the system new signs on the fly. A user would perform a gesture several times, label it, and the system would immediately recognize it going forward -- no retraining of the neural network required.

---

## License

This project was developed as a 6th-semester Computer Science Engineering capstone. For licensing inquiries, please contact the repository owner.
