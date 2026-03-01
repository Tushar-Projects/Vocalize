# Vocalize: Bidirectional Sign Language Translator

Vocalize is a real-time, bidirectional computer vision and machine learning application designed to bridge the communication gap between sign language users and non-signers. 

Built for a 6th-semester Computer Science Engineering capstone, this project utilizes a custom-trained LSTM neural network, WebSockets for ultra-low latency streaming, and an enterprise PostgreSQL database for conversation tracking.

## 🚀 Tech Stack
* **Machine Learning:** TensorFlow / Keras (LSTM), MediaPipe Holistic
* **Backend:** FastAPI, Uvicorn, SQLAlchemy (Sync), WebSockets
* **Database:** PostgreSQL
* **Frontend:** HTML5, Vanilla JavaScript, CSS (Glassmorphism UI)
* **Data Processing:** OpenCV, NumPy

## 🏗️ Architecture Overview
1. **Frontend Capture:** The browser extracts webcam frames, corrects aspect ratios via a hidden canvas, and streams lightweight Base64 JPEGs to the backend at 15 FPS.
2. **Real-Time Inference:** A FastAPI WebSocket endpoint receives the frames, drains the TCP queue to prevent latency, and processes the newest frame through MediaPipe to extract 1,662 facial, pose, and hand landmarks.
3. **Sequence Prediction:** A 30-frame rolling window of landmarks is fed into a custom LSTM neural network.
4. **Data Persistence:** Predictions with >60% confidence are de-duplicated and logged to a PostgreSQL database via SQLAlchemy ORM.

## ⚙️ Local Setup Instructions

### 1. Database Setup
Ensure PostgreSQL is installed and running on port `5432`.
Create a database named `vocalize_db`.

### 2. Environment Variables
Create a `.env` file in the root directory:
\`\`\`env
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/vocalize_db
\`\`\`

### 3. Installation & Boot
\`\`\`bash
# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server (Auto-generates database tables on first boot)
python -m uvicorn backend.main:app --reload --port 8000
\`\`\`

Navigate to `http://localhost:8000/app/` to access the application.

## 🗺️ Master Roadmap
- [x] Phase 1-3: MediaPipe Extraction, Data Augmentation, and LSTM Training
- [x] Phase 4: Local OpenCV Real-Time Inference
- [x] Phase 5: FastAPI WebRTC/WebSocket Streaming & PostgreSQL DB Integration
- [ ] Phase 6: Frontend Conversation History UI
- [ ] Phase 7: Domain-Specific Vocabulary Scaling (Final Dataset)
- [ ] Phase 8: Reverse Pipeline (Speech-to-Text & Sign Video Generation)
- [ ] Phase 9: UI/UX Polish & Documentation
- [ ] Phase 10: Virtual Camera Integration for Zoom / Google Meet
