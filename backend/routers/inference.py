"""
Vocalize — Inference WebSocket Router

Receives Base64-encoded video frames from the browser via WebSocket,
runs MediaPipe Holistic → LSTM inference, and sends predictions back.
High-confidence detections are logged to PostgreSQL with deduplication.
"""

import asyncio
import base64
import json
import logging
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Phase 1 landmark extraction
from landmark_extractor import mediapipe_detect, extract_keypoints
# Custom Attention layer (must be registered for model.load)
from attention import AttentionLayer

from backend.database import SessionLocal
from backend.models import Transcript

logger = logging.getLogger("vocalize.inference")

router = APIRouter(tags=["inference"])

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
SEQUENCE_LENGTH   = 30
N_FEATURES        = 1662
CONFIDENCE_THRESH = 0.60
DEDUP_COOLDOWN_S  = 2.0   # minimum seconds between DB writes of the same label

# ──────────────────────────────────────────────
# Model singleton (loaded once at import time)
# ──────────────────────────────────────────────
MODEL_PATH  = str(Path(__file__).resolve().parent.parent / "action.h5")
LABELS_PATH = str(Path(__file__).resolve().parent.parent / "action_labels.json")

_model = None
_label_map = None
_holistic = None


def get_model():
    """Lazy-load the Bi-LSTM+Attention model with custom_objects and warm it up."""
    global _model, _label_map
    if _model is None:
        import json as _json

        logger.info(f"Loading model from {MODEL_PATH}")
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"AttentionLayer": AttentionLayer},
        )

        logger.info(f"Loading labels from {LABELS_PATH}")
        with open(LABELS_PATH, "r") as f:
            raw = _json.load(f)
        _label_map = {int(k): v for k, v in raw.items()}

        # Warm-up pass
        dummy = tf.constant(np.zeros((1, SEQUENCE_LENGTH, N_FEATURES), dtype=np.float32))
        _model(dummy, training=False)
        logger.info(f"Model loaded and warmed up. Classes: {_label_map}")
    return _model, _label_map


def get_holistic():
    """Lazy-load a shared MediaPipe Holistic instance."""
    global _holistic
    if _holistic is None:
        _holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )
        logger.info("MediaPipe Holistic initialized.")
    return _holistic


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def decode_frame(b64_data: str) -> np.ndarray | None:
    """Decode a Base64 string (with or without data-URI prefix) to a BGR numpy array."""
    try:
        # Strip optional data URI prefix
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        raw = base64.b64decode(b64_data)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.warning(f"Frame decode error: {e}")
        return None


def save_transcript(session_id: str, label: str, confidence: float, seq_idx: int):
    """Save a detection to PostgreSQL (sync, in a dedicated session)."""
    db = SessionLocal()
    try:
        entry = Transcript(
            session_id=session_id,
            sign_label=label,
            confidence=confidence,
            sequence_index=seq_idx,
        )
        db.add(entry)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"DB write failed: {e}")
    finally:
        db.close()


# ──────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────

@router.websocket("/ws/inference/{session_id}")
async def inference_endpoint(websocket: WebSocket, session_id: str):
    """
    Real-time inference via WebSocket.

    Protocol
    --------
    Client → Server:  {"type": "frame", "data": "<base64 JPEG>"}
    Server → Client:  {"type": "prediction", "label": "...", "confidence": 0.85, "buffer": 30}
                      {"type": "status", "message": "..."}
    """
    await websocket.accept()
    logger.info(f"[+] Inference session '{session_id}' connected")

    model, label_map = get_model()
    holistic = get_holistic()

    # Rolling keypoint buffer
    sequence_buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

    # Deduplication state
    last_saved_label = ""
    last_saved_time  = 0.0
    seq_counter      = 0

    await websocket.send_text(json.dumps({
        "type": "status",
        "message": f"Connected. Model loaded ({len(label_map)} classes). Buffering frames...",
    }))

    try:
        while True:
            # ── Drain the queue: read all available messages, keep only
            #    the newest frame to avoid processing stale backlog ──
            latest_frame_data: str | None = None

            # First message: blocking wait (normal read)
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                if msg.get("type") == "frame":
                    latest_frame_data = msg.get("data", "")
            except json.JSONDecodeError:
                pass

            # Non-blocking drain: pull any queued messages and keep
            # overwriting latest_frame_data with the freshest frame
            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(), timeout=0.001
                    )
                    try:
                        msg = json.loads(raw)
                        if msg.get("type") == "frame":
                            latest_frame_data = msg.get("data", "")
                    except json.JSONDecodeError:
                        pass
                except asyncio.TimeoutError:
                    break  # queue is empty, proceed with latest frame

            if latest_frame_data is None:
                continue

            # ── 1. Decode frame ──
            frame = decode_frame(latest_frame_data)
            if frame is None:
                continue

            # ── 2. MediaPipe detection ──
            frame, results = mediapipe_detect(frame, holistic)

            # ── 3. Extract keypoints → buffer ──
            keypoints = extract_keypoints(results)
            sequence_buffer.append(keypoints.copy())

            buf_len = len(sequence_buffer)

            # ── 4. Predict when buffer full ──
            if buf_len == SEQUENCE_LENGTH:
                input_seq = np.array(list(sequence_buffer), dtype=np.float32)
                input_seq = np.expand_dims(input_seq, axis=0)

                input_tensor = tf.constant(input_seq)
                probs = model(input_tensor, training=False).numpy()[0]
                class_idx = int(np.argmax(probs))
                confidence = float(probs[class_idx])
                label = label_map.get(class_idx, "unknown")

                # Send prediction to client
                await websocket.send_text(json.dumps({
                    "type": "prediction",
                    "label": label,
                    "confidence": round(confidence, 4),
                    "buffer": buf_len,
                }))

                # ── 5. Log to DB with deduplication ──
                now = time.time()
                if confidence >= CONFIDENCE_THRESH:
                    is_new_word = (label != last_saved_label)
                    cooldown_elapsed = (now - last_saved_time) >= DEDUP_COOLDOWN_S

                    if is_new_word or cooldown_elapsed:
                        seq_counter += 1
                        save_transcript(session_id, label, confidence, seq_counter)
                        last_saved_label = label
                        last_saved_time = now
                        logger.info(
                            f"[DB] session={session_id}  #{seq_counter}  "
                            f"{label} ({confidence:.0%})"
                        )
            else:
                # Still buffering
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "message": f"Buffering... ({buf_len}/{SEQUENCE_LENGTH})",
                    "buffer": buf_len,
                }))

    except WebSocketDisconnect:
        logger.info(f"[-] Inference session '{session_id}' disconnected "
                    f"({seq_counter} detections saved)")
    except Exception as e:
        logger.error(f"[!] Inference session '{session_id}' error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass

