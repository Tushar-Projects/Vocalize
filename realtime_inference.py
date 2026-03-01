"""
Vocalize — Phase 4: Real-Time Webcam Inference

Loads the trained LSTM model (action.h5) and runs live sign-language
prediction using a rolling 30-frame window.  Predictions are overlaid
on the video feed only when confidence exceeds a configurable threshold.

Usage
─────
    python realtime_inference.py
    python realtime_inference.py --model action.h5 --threshold 0.65
    python realtime_inference.py --debug          # enable diagnostic logging

Controls
────────
    q  — quit
"""

import os
import json
import time
import argparse
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

# ── Phase 1 reuse ──
from landmark_extractor import mediapipe_detect, extract_keypoints, draw_landmarks

import mediapipe as mp

mp_holistic = mp.solutions.holistic

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
SEQUENCE_LENGTH   = 30          # must match training
N_FEATURES        = 1662
DEFAULT_THRESHOLD = 0.70

# ──────────────────────────────────────────────
# Model & label loading
# ──────────────────────────────────────────────

def load_model_and_labels(
    model_path: str,
    labels_path: str,
) -> tuple[tf.keras.Model, dict[int, str]]:
    """
    Load the Keras model and the label index map.

    Args:
        model_path:  Path to action.h5.
        labels_path: Path to action_labels.json.

    Returns:
        (model, label_map) where label_map is {int: str}.
    """
    from attention import AttentionLayer

    print(f"[INFO] Loading model from: {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"AttentionLayer": AttentionLayer},
    )

    print(f"[INFO] Loading labels from: {labels_path}")
    with open(labels_path, "r") as f:
        raw = json.load(f)
    label_map = {int(k): v for k, v in raw.items()}

    print(f"[INFO] Classes: {label_map}")
    return model, label_map


# ──────────────────────────────────────────────
# Prediction helper  (FIXED: use direct model call)
# ──────────────────────────────────────────────

def predict_sign(
    model: tf.keras.Model,
    sequence: np.ndarray,
    label_map: dict[int, str],
) -> tuple[str, float, np.ndarray]:
    """
    Run inference on a single sequence using a direct model call.

    IMPORTANT: Uses model(tensor, training=False) instead of
    model.predict() to avoid TF graph re-tracing overhead that
    caused the frozen-prediction bug.

    Args:
        model:     Trained Keras LSTM model.
        sequence:  np.ndarray of shape (1, SEQUENCE_LENGTH, N_FEATURES).
        label_map: Dict mapping class index → label name.

    Returns:
        (predicted_label, confidence, raw_probs) tuple.
    """
    # Direct call — fast for single-sample inference in a loop
    input_tensor = tf.constant(sequence)
    probs = model(input_tensor, training=False).numpy()[0]  # shape (n_classes,)

    class_idx = int(np.argmax(probs))
    confidence = float(probs[class_idx])
    label = label_map.get(class_idx, "unknown")
    return label, confidence, probs


# ──────────────────────────────────────────────
# HUD rendering
# ──────────────────────────────────────────────

def render_hud(
    frame: np.ndarray,
    prediction: str,
    confidence: float,
    buffer_fill: int,
    threshold: float,
) -> np.ndarray:
    """
    Overlay prediction text and status info on the frame.

    Args:
        frame:       BGR image to draw on (modified in-place).
        prediction:  Current predicted label (or "..." if below threshold).
        confidence:  Model confidence for the prediction.
        buffer_fill: Number of frames currently in the rolling buffer.
        threshold:   Confidence threshold for showing a prediction.

    Returns:
        The annotated frame.
    """
    h, w = frame.shape[:2]

    # ── semi-transparent banner at top ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # ── prediction text ──
    if buffer_fill < SEQUENCE_LENGTH:
        display_text = f"Buffering... ({buffer_fill}/{SEQUENCE_LENGTH})"
        color = (100, 100, 100)
    elif confidence >= threshold:
        display_text = f"{prediction.upper()}  ({confidence:.0%})"
        color = (0, 255, 0)
    else:
        display_text = f"...  ({confidence:.0%})"
        color = (0, 165, 255)

    cv2.putText(
        frame, display_text,
        (20, 55), cv2.FONT_HERSHEY_SIMPLEX,
        1.4, color, 3, cv2.LINE_AA,
    )

    # ── status bar at bottom ──
    status = f"Threshold: {threshold:.0%}  |  Buffer: {buffer_fill}/{SEQUENCE_LENGTH}  |  'q' quit"
    cv2.putText(
        frame, status,
        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (200, 200, 200), 1, cv2.LINE_AA,
    )

    return frame


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────

def run_inference(
    model: tf.keras.Model,
    label_map: dict[int, str],
    threshold: float = DEFAULT_THRESHOLD,
    debug: bool = False,
) -> None:
    """
    Open the webcam and run real-time sign-language inference.

    This function is kept separate from main() for easy reuse
    in Phase 5 (WebRTC) — just replace the video source.

    Args:
        model:     Loaded Keras model.
        label_map: Label index map.
        threshold: Minimum confidence to display a prediction.
        debug:     If True, print diagnostic info every 30 frames.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    # Rolling buffer of keypoint vectors
    sequence_buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

    current_prediction = "..."
    current_confidence = 0.0
    frame_count = 0
    prev_kp_sum = None

    # Warm up the model with a dummy call so the first real frame isn't slow
    print("[INFO] Warming up model...")
    dummy = tf.constant(np.zeros((1, SEQUENCE_LENGTH, N_FEATURES), dtype=np.float32))
    _ = model(dummy, training=False)
    print("[INFO] Model warm-up complete.")

    print("[INFO] Starting inference loop. Press 'q' to quit.\n")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as holistic:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame_count += 1

            # ── 1. Detect landmarks ──
            frame, results = mediapipe_detect(frame, holistic)
            draw_landmarks(frame, results)

            # ── 2. Extract keypoints → append to rolling buffer ──
            keypoints = extract_keypoints(results)
            kp_sum = float(np.sum(keypoints))

            # ── DEBUG: log keypoint health every 30 frames ──
            if debug and frame_count % 30 == 0:
                kp_nonzero = int(np.count_nonzero(keypoints))
                delta = abs(kp_sum - prev_kp_sum) if prev_kp_sum is not None else 0.0
                has_pose = results.pose_landmarks is not None
                has_face = results.face_landmarks is not None
                has_lh   = results.left_hand_landmarks is not None
                has_rh   = results.right_hand_landmarks is not None
                print(
                    f"[DEBUG F{frame_count:>5}] "
                    f"kp_sum={kp_sum:>10.2f}  "
                    f"nonzero={kp_nonzero:>5}/{N_FEATURES}  "
                    f"delta={delta:>8.2f}  "
                    f"pose={has_pose}  face={has_face}  LH={has_lh}  RH={has_rh}"
                )

            prev_kp_sum = kp_sum

            # Make an explicit copy for the buffer to avoid reference issues
            sequence_buffer.append(keypoints.copy())

            # ── 3. Predict when buffer is full ──
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                input_seq = np.array(list(sequence_buffer), dtype=np.float32)
                input_seq = np.expand_dims(input_seq, axis=0)  # (1, 30, 1662)

                label, conf, raw_probs = predict_sign(model, input_seq, label_map)

                # Always update with the latest prediction
                current_prediction = label
                current_confidence = conf

                # ── DEBUG: log prediction details ──
                if debug and frame_count % 30 == 0:
                    buf_variance = float(np.var(input_seq))
                    probs_str = "  ".join(
                        f"{label_map[i]}={p:.3f}" for i, p in enumerate(raw_probs)
                    )
                    print(
                        f"[DEBUG F{frame_count:>5}] "
                        f"buf_var={buf_variance:.6f}  "
                        f"pred={label}({conf:.3f})  "
                        f"probs=[{probs_str}]"
                    )

            # ── 4. Render HUD ──
            render_hud(
                frame,
                current_prediction,
                current_confidence,
                len(sequence_buffer),
                threshold,
            )

            cv2.imshow("Vocalize — Real-Time Inference", frame)

            if (cv2.waitKey(5) & 0xFF) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Inference stopped after {frame_count} frames. Goodbye!")


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vocalize Phase 4 — Real-time sign language inference",
    )
    parser.add_argument(
        "--model", type=str, default="action.h5",
        help="Path to trained Keras model (default: action.h5)",
    )
    parser.add_argument(
        "--labels", type=str, default=None,
        help="Path to label map JSON (default: <model>_labels.json)",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Confidence threshold for predictions (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable diagnostic logging every 30 frames",
    )
    args = parser.parse_args()

    # Derive labels path from model path if not specified
    if args.labels is None:
        args.labels = os.path.splitext(args.model)[0] + "_labels.json"

    print("=" * 60)
    print("  Vocalize — Phase 4  Real-Time Inference")
    print("=" * 60)
    print(f"  Model     : {args.model}")
    print(f"  Labels    : {args.labels}")
    print(f"  Threshold : {args.threshold:.0%}")
    print(f"  Debug     : {args.debug}")
    print("=" * 60)

    model, label_map = load_model_and_labels(args.model, args.labels)
    run_inference(model, label_map, args.threshold, args.debug)


if __name__ == "__main__":
    main()
