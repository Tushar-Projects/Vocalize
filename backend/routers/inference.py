"""
Vocalize — Inference WebSocket Router

Receives Base64-encoded video frames from the browser via WebSocket,
runs MediaPipe Holistic → LSTM inference, and sends predictions back.
High-confidence detections are logged to PostgreSQL with deduplication.
NLP grammar correction fires when no new sign is detected for 2 seconds.

Phase 11 adds KNN custom-gesture matching: user-recorded sign embeddings
are compared against live attention-layer embeddings for hybrid inference.
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
# Phase 7 NLP engine
from backend.nlp_engine import correct_and_predict

from backend.database import SessionLocal
from backend.models import Transcript, CustomGesture

# Phase 11 embedding extractor
from backend.embedding_model import (
    build_embedding_model,
    l2_normalize,
    nearest_custom_gesture,
)

logger = logging.getLogger("vocalize.inference")

router = APIRouter(tags=["inference"])

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
SEQUENCE_LENGTH   = 30
N_FEATURES        = 1662
CONFIDENCE_THRESH = 0.60
DEDUP_COOLDOWN_S  = 2.0   # minimum seconds between DB writes of the same label
NLP_DEBOUNCE_S    = 2.0   # seconds of silence before triggering NLP

# Phase 11 — custom gesture recording & matching
TOTAL_REPS        = 5     # number of recording reps per custom gesture
KNN_DIST_THRESHOLD = 0.75  # L2 distance on unit vectors (range 0–2); retune empirically
HAND_QUALITY_FRAC = 0.60  # fraction of frames that must have hand landmarks visible
HAND_KP_START     = 1536  # index where hand keypoints begin (pose=132 + face=1404)

# ──────────────────────────────────────────────
# Model singleton (loaded once at import time)
# ──────────────────────────────────────────────
MODEL_PATH  = str(Path(__file__).resolve().parent.parent / "action.h5")
LABELS_PATH = str(Path(__file__).resolve().parent.parent / "action_labels.json")

_model = None
_label_map = None
_holistic = None

# Phase 11 — embedding model singleton + per-client gesture cache
_embed_model = None
_custom_gesture_cache: dict[str, list[tuple[str, np.ndarray]]] = {}
# client_id → [(label, embedding_vector), ...]   — keyed by persistent client_id


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


def get_embed_model():
    """Lazy-build the embedding model from the loaded classifier."""
    global _embed_model
    if _embed_model is None:
        model, _ = get_model()
        _embed_model = build_embedding_model(model)
    return _embed_model


# ──────────────────────────────────────────────
# Phase 11 — Gesture cache helpers
# ──────────────────────────────────────────────

def _load_gesture_cache(client_id: str) -> list[tuple[str, np.ndarray]]:
    """
    Load custom gesture embeddings from DB for a client_id.
    Called lazily on first access per client_id.
    """
    db = SessionLocal()
    try:
        rows = (
            db.query(CustomGesture)
            .filter(CustomGesture.client_id == client_id)
            .all()
        )
        entries = [
            (row.label, np.array(row.embedding, dtype=np.float32))
            for row in rows
        ]
        logger.info(
            f"[GESTURE-CACHE] Loaded {len(entries)} custom gesture(s) "
            f"for client={client_id}"
        )
        return entries
    except Exception as e:
        logger.error(f"[GESTURE-CACHE] DB load failed: {e}")
        return []
    finally:
        db.close()


def _ensure_gesture_cache(client_id: str) -> list[tuple[str, np.ndarray]]:
    """
    Return the cached gesture list for client_id, loading from DB
    on first access.  Thread-safe because we only ever assign a new
    list to the dict key (atomic under GIL), never mutate in-place.
    """
    if client_id not in _custom_gesture_cache:
        _custom_gesture_cache[client_id] = _load_gesture_cache(client_id)
    return _custom_gesture_cache[client_id]


def invalidate_gesture_cache(client_id: str) -> None:
    """
    Remove a client's gesture cache so it's reloaded from DB on
    next access.

    IMPORTANT: This is called from the REST router's DELETE endpoint,
    which runs in a threadpool thread.  Must use atomic key deletion
    (never in-place list mutation) so a concurrent KNN reader in the
    async event loop sees either the old complete list or no list at
    all — never a partially-modified one.
    """
    _custom_gesture_cache.pop(client_id, None)
    logger.info(f"[GESTURE-CACHE] Invalidated cache for client={client_id}")


def _check_hand_quality(frames: list[np.ndarray]) -> bool:
    """
    Check that hand landmarks are non-zero in at least HAND_QUALITY_FRAC
    of the recorded frames.

    Keypoint layout: [pose(132) | face(1404) | left_hand(63) | right_hand(63)]
    Hand keypoints start at index HAND_KP_START (1536).
    """
    min_frames = int(len(frames) * HAND_QUALITY_FRAC)
    good = sum(1 for kp in frames if np.any(kp[HAND_KP_START:]))
    return good >= min_frames


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
                      {"type": "config", "target_language": "English"}
                      {"type": "record_gesture_start", "label": "my_sign"}
                      {"type": "record_gesture_stop"}
    Server → Client:  {"type": "prediction", "label": "...", "confidence": 0.85, "buffer": 30, "source": "model"|"custom"}
                      {"type": "status", "message": "..."}
                      {"type": "nlp", "corrected_english": "...", "translated_text": "...", "auto_complete": [...], "raw_gloss": [...]}
                      {"type": "gesture_rep_captured", "rep": 1, "of": 5}
                      {"type": "gesture_rep_rejected", "reason": "hands not visible"}
                      {"type": "gesture_recorded", "label": "...", "samples": 5}
    """
    await websocket.accept()
    logger.info(f"[+] Inference session '{session_id}' connected")

    # Read persistent client_id from query string (sent by frontend)
    client_id: str | None = websocket.query_params.get("client_id")
    if client_id:
        logger.info(f"    client_id={client_id}")

    model, label_map = get_model()
    embed_model = get_embed_model()
    holistic = get_holistic()

    # Rolling keypoint buffer
    sequence_buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

    # DB deduplication state
    last_saved_label = ""
    last_saved_time  = 0.0
    seq_counter      = 0

    # ── NLP state (bulletproof) ──
    session_words: list[str] = []         # accumulated gloss words
    last_detected_word: str | None = None # last word appended to session_words
    last_word_time: float = 0.0           # timestamp when last NEW word was added
    target_language: str = "English"       # user-selected output language

    # ── Phase 11: Custom gesture recording state ──
    recording_gesture: bool = False
    recording_label: str = ""
    recording_buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    recording_embeddings: list[np.ndarray] = []
    recording_rep_count: int = 0

    await websocket.send_text(json.dumps({
        "type": "status",
        "message": f"Connected. Model loaded ({len(label_map)} classes). Buffering frames...",
    }))

    # Pre-load custom gesture cache for this client (if client_id present)
    if client_id:
        _ensure_gesture_cache(client_id)

    try:
        while True:
            # ── Drain the queue: keep only the newest frame ──
            latest_frame_data: str | None = None

            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                if msg.get("type") == "frame":
                    latest_frame_data = msg.get("data", "")
                elif msg.get("type") == "config":
                    target_language = msg.get("target_language", "English")
                    print(f"[CONFIG] Target language set to: {target_language}")
                elif msg.get("type") == "record_gesture_start":
                    # Start (or restart) recording a custom gesture.
                    # Re-entrant: if already recording, restart from scratch
                    # (handles double-clicks / flaky connection retries).
                    recording_label = msg.get("label", "").strip()
                    if recording_label and client_id:
                        recording_gesture = True
                        recording_buffer = deque(maxlen=SEQUENCE_LENGTH)
                        recording_embeddings = []
                        recording_rep_count = 0
                        logger.info(
                            f"[GESTURE] Recording started: '{recording_label}' "
                            f"for client={client_id}"
                        )
                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "message": f"Recording '{recording_label}' — perform the sign (rep 1/{TOTAL_REPS})",
                        }))
                elif msg.get("type") == "record_gesture_stop":
                    # Manual stop — discard partial recording
                    if recording_gesture:
                        recording_gesture = False
                        recording_label = ""
                        recording_buffer = deque(maxlen=SEQUENCE_LENGTH)
                        recording_embeddings = []
                        recording_rep_count = 0
                        logger.info(f"[GESTURE] Recording manually stopped")
                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "message": "Recording cancelled.",
                        }))
            except json.JSONDecodeError:
                pass

            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(), timeout=0.001
                    )
                    try:
                        msg = json.loads(raw)
                        if msg.get("type") == "frame":
                            latest_frame_data = msg.get("data", "")
                        elif msg.get("type") == "config":
                            target_language = msg.get("target_language", "English")
                            print(f"[CONFIG] Target language set to: {target_language}")
                        elif msg.get("type") == "record_gesture_start":
                            recording_label = msg.get("label", "").strip()
                            if recording_label and client_id:
                                recording_gesture = True
                                recording_buffer = deque(maxlen=SEQUENCE_LENGTH)
                                recording_embeddings = []
                                recording_rep_count = 0
                        elif msg.get("type") == "record_gesture_stop":
                            if recording_gesture:
                                recording_gesture = False
                                recording_label = ""
                                recording_buffer = deque(maxlen=SEQUENCE_LENGTH)
                                recording_embeddings = []
                                recording_rep_count = 0
                    except json.JSONDecodeError:
                        pass
                except asyncio.TimeoutError:
                    break

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

            # ── Phase 11: Feed recording buffer if active ──
            if recording_gesture:
                recording_buffer.append(keypoints.copy())

                rec_len = len(recording_buffer)
                if rec_len == SEQUENCE_LENGTH:
                    # A full rep has been captured — check quality then embed
                    rec_frames = list(recording_buffer)

                    if _check_hand_quality(rec_frames):
                        # Good rep — compute embedding
                        rec_seq = np.array(rec_frames, dtype=np.float32)
                        rec_seq = np.expand_dims(rec_seq, axis=0)
                        rec_tensor = tf.constant(rec_seq)
                        rec_embedding = embed_model(rec_tensor, training=False).numpy()[0]
                        rec_embedding = l2_normalize(rec_embedding)

                        recording_embeddings.append(rec_embedding)
                        recording_rep_count += 1

                        await websocket.send_text(json.dumps({
                            "type": "gesture_rep_captured",
                            "rep": recording_rep_count,
                            "of": TOTAL_REPS,
                        }))
                        logger.info(
                            f"[GESTURE] Rep {recording_rep_count}/{TOTAL_REPS} "
                            f"captured for '{recording_label}'"
                        )
                    else:
                        # Bad rep — reject but don't count
                        await websocket.send_text(json.dumps({
                            "type": "gesture_rep_rejected",
                            "reason": "hands not visible",
                        }))
                        logger.info(
                            f"[GESTURE] Rep rejected for '{recording_label}' "
                            f"— insufficient hand visibility"
                        )

                    # Always clear deque after a rep attempt (accepted or rejected)
                    # so the next rep starts from an empty buffer
                    recording_buffer = deque(maxlen=SEQUENCE_LENGTH)

                    # Check if all reps captured — auto-stop
                    if recording_rep_count >= TOTAL_REPS:
                        # Average embeddings → re-normalize → store
                        avg_embedding = np.mean(recording_embeddings, axis=0)
                        avg_embedding = l2_normalize(avg_embedding)

                        # Save to DB
                        db = SessionLocal()
                        try:
                            gesture = CustomGesture(
                                client_id=client_id,
                                session_id=session_id,
                                label=recording_label,
                                embedding=avg_embedding.tolist(),
                                rep_count=TOTAL_REPS,
                            )
                            db.add(gesture)
                            db.commit()
                            logger.info(
                                f"[GESTURE] Saved '{recording_label}' "
                                f"for client={client_id} ({TOTAL_REPS} reps)"
                            )
                        except Exception as e:
                            db.rollback()
                            logger.error(f"[GESTURE] DB write failed: {e}")
                        finally:
                            db.close()

                        # Update in-memory cache — atomic swap with new list
                        old_entries = _custom_gesture_cache.get(client_id, [])
                        new_entries = old_entries + [(recording_label, avg_embedding)]
                        _custom_gesture_cache[client_id] = new_entries

                        await websocket.send_text(json.dumps({
                            "type": "gesture_recorded",
                            "label": recording_label,
                            "samples": TOTAL_REPS,
                        }))

                        # Exit recording state
                        recording_gesture = False
                        recording_label = ""
                        recording_embeddings = []
                        recording_rep_count = 0

                # While recording, skip prediction/NLP — user is deliberately
                # performing a new sign, not chatting
                continue

            # ── 4. Predict when buffer full ──
            if buf_len == SEQUENCE_LENGTH:
                input_seq = np.array(list(sequence_buffer), dtype=np.float32)
                input_seq = np.expand_dims(input_seq, axis=0)

                input_tensor = tf.constant(input_seq)
                probs = model(input_tensor, training=False).numpy()[0]
                class_idx = int(np.argmax(probs))
                confidence = float(probs[class_idx])
                label = label_map.get(class_idx, "unknown")

                # ── Phase 11: Hybrid decision — KNN custom gesture matching ──
                source = "model"
                final_label = label
                final_confidence = confidence

                if client_id:
                    candidates = _ensure_gesture_cache(client_id)
                    if candidates:
                        embedding = embed_model(input_tensor, training=False).numpy()[0]
                        knn_label, knn_dist = nearest_custom_gesture(embedding, candidates)

                        use_custom = knn_label is not None and knn_dist < KNN_DIST_THRESHOLD
                        if use_custom:
                            final_label = knn_label
                            final_confidence = max(0.0, 1.0 - knn_dist)
                            source = "custom"

                # Send prediction to client
                await websocket.send_text(json.dumps({
                    "type": "prediction",
                    "label": final_label,
                    "confidence": round(final_confidence, 4),
                    "buffer": buf_len,
                    "source": source,
                }))

                # ── 5. Log to DB with deduplication ──
                now = time.time()
                if final_confidence >= CONFIDENCE_THRESH:
                    is_new_word = (final_label != last_saved_label)
                    cooldown_elapsed = (now - last_saved_time) >= DEDUP_COOLDOWN_S

                    if is_new_word or cooldown_elapsed:
                        seq_counter += 1
                        save_transcript(session_id, final_label, final_confidence, seq_counter)
                        last_saved_label = final_label
                        last_saved_time = now
                        logger.info(
                            f"[DB] session={session_id}  #{seq_counter}  "
                            f"{final_label} ({final_confidence:.0%}) [{source}]"
                        )

                    # ── 5b. NLP word tracking (separate from DB dedup) ──
                    # Uses final_label so custom signs flow into the Groq
                    # correction pipeline alongside model-detected words.
                    if final_label != last_detected_word:
                        session_words.append(final_label)
                        last_detected_word = final_label
                        last_word_time = now
                        print(f"[NLP-TRACK] New word: '{final_label}' | session_words = {session_words}")

            else:
                # Still buffering
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "message": f"Buffering... ({buf_len}/{SEQUENCE_LENGTH})",
                    "buffer": buf_len,
                }))

            # ══════════════════════════════════════════════════════════
            # ── 6. NLP DEBOUNCE — runs on EVERY loop iteration ──
            # ══════════════════════════════════════════════════════════
            now = time.time()
            if (
                len(session_words) > 0
                and last_word_time > 0
                and (now - last_word_time) > NLP_DEBOUNCE_S
            ):
                print(f"🚨 Timer hit! Calling Groq... session_words={session_words}")

                try:
                    nlp_result = await correct_and_predict(list(session_words), target_language)
                    print(f"✅ Groq response: {nlp_result}")

                    await websocket.send_text(json.dumps({
                        "type": "nlp",
                        "corrected_english": nlp_result.get("corrected_english", ""),
                        "translated_text": nlp_result.get("translated_text", ""),
                        "auto_complete": nlp_result.get("auto_complete", []),
                        "raw_gloss": list(session_words),
                        "target_language": target_language,
                    }))

                    logger.info(
                        f"[NLP] session={session_id}  "
                        f"gloss={session_words} → \"{nlp_result.get('corrected_sentence', '')}\""
                    )
                except Exception as e:
                    print(f"NLP Error: {e}")
                    logger.error(f"[NLP] Error: {e}")
                    # Don't let NLP errors kill the video loop — just continue

                # Reset — clear words so we don't re-trigger
                session_words.clear()
                last_detected_word = None
                last_word_time = 0.0

    except WebSocketDisconnect:
        logger.info(f"[-] Inference session '{session_id}' disconnected "
                    f"({seq_counter} detections saved)")
    except Exception as e:
        logger.error(f"[!] Inference session '{session_id}' error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass
