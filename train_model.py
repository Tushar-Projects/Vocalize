"""
Vocalize — Bi-LSTM + Attention Neural Network Training

Loads padded keypoint sequences from processed_data/, trains a
Bidirectional LSTM with a custom Attention layer, and saves the
model as wlasl_15_gloss.keras.

Usage
─────
    python train_model.py --data_dir processed_data
    python train_model.py --data_dir processed_data --epochs 200 --seq_len 30

Architecture (Functional API)
─────────────────────────────
    Input(30, 1662)
    Masking(0.0)
    Bidirectional(LSTM(128, return_seq))  →  Dropout(0.3)
    Bidirectional(LSTM(64,  return_seq))  →  Dropout(0.3)
    AttentionLayer()                      →  learns per-frame importance
    Dense(64, relu)                       →  Dropout(0.3)
    Dense(n_classes, softmax)
"""

import os
import sys
import json
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Masking,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    TensorBoard,
    ModelCheckpoint,
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from attention import AttentionLayer

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
SEQUENCE_LENGTH = 30        # frames per sample
N_FEATURES     = 1662       # keypoints per frame
MASK_VALUE     = 0.0        # padding value to mask

# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_dataset(data_dir: str) -> tuple[list[np.ndarray], list[int], dict[int, str]]:
    """
    Walk processed_data/<label>/ directories and load .npy files.

    Args:
        data_dir: Root of the processed data (e.g. 'processed_data').

    Returns:
        sequences:  List of np.ndarray, each (num_frames, 1662).
        labels:     List of integer class indices, one per sequence.
        label_map:  Dict mapping class index → label name.
    """
    if not os.path.isdir(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        sys.exit(1)

    label_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if not label_names:
        print(f"[ERROR] No label sub-folders found in: {data_dir}")
        sys.exit(1)

    label_map: dict[int, str] = {i: name for i, name in enumerate(label_names)}
    name_to_idx: dict[str, int] = {name: i for i, name in enumerate(label_names)}

    sequences: list[np.ndarray] = []
    labels: list[int] = []

    for label_name in label_names:
        label_dir = os.path.join(data_dir, label_name)
        npy_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".npy")])

        for npy_file in npy_files:
            path = os.path.join(label_dir, npy_file)
            seq = np.load(path)  # shape (F, 1662)
            sequences.append(seq)
            labels.append(name_to_idx[label_name])

    print(f"[INFO] Loaded {len(sequences)} sequence(s) across {len(label_names)} class(es)")
    for idx, name in label_map.items():
        count = labels.count(idx)
        print(f"       {idx}: {name:15s}  ({count} samples)")

    return sequences, labels, label_map


# ──────────────────────────────────────────────
# Sequence padding / truncating
# ──────────────────────────────────────────────

def pad_sequences_3d(
    sequences: list[np.ndarray],
    max_len: int,
    n_features: int = N_FEATURES,
) -> np.ndarray:
    """
    Pad or truncate each (F, 1662) array to (max_len, 1662).

    - Short sequences → post-padded with 0.0.
    - Long  sequences → truncated from the end (keeps the start).

    Args:
        sequences:  List of arrays, each (F, n_features).
        max_len:    Target sequence length.
        n_features: Feature dimension (default 1662).

    Returns:
        np.ndarray of shape (num_samples, max_len, n_features).
    """
    padded = np.zeros((len(sequences), max_len, n_features), dtype=np.float32)

    stats = {"padded": 0, "truncated": 0, "exact": 0}

    for i, seq in enumerate(sequences):
        length = min(seq.shape[0], max_len)
        padded[i, :length, :] = seq[:length, :]

        if seq.shape[0] < max_len:
            stats["padded"] += 1
        elif seq.shape[0] > max_len:
            stats["truncated"] += 1
        else:
            stats["exact"] += 1

    print(f"[INFO] Sequences → ({len(sequences)}, {max_len}, {n_features})")
    print(f"       Padded: {stats['padded']}  |  Truncated: {stats['truncated']}  |  Exact: {stats['exact']}")

    return padded


# ──────────────────────────────────────────────
# Model architecture — Bi-LSTM + Attention
# ──────────────────────────────────────────────

def build_model(seq_len: int, n_features: int, n_classes: int) -> Model:
    """
    Build a Keras Functional-API model:
        Masking → BiLSTM → BiLSTM → Attention → Dense → Softmax

    Args:
        seq_len:    Number of time-steps (frames).
        n_features: Number of input features per frame.
        n_classes:  Number of output classes.

    Returns:
        Compiled Keras model.
    """
    inp = Input(shape=(seq_len, n_features), name="input_sequence")

    # ── Masking: ignore zero-padded frames ──
    x = Masking(mask_value=MASK_VALUE)(inp)

    # ── Bidirectional LSTM stack ──
    x = Bidirectional(
        LSTM(128, return_sequences=True, activation="tanh"),
        name="bilstm_1",
    )(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(
        LSTM(64, return_sequences=True, activation="tanh"),
        name="bilstm_2",
    )(x)
    x = Dropout(0.3)(x)

    # ── Attention: learn which frames matter most ──
    x = AttentionLayer(name="attention")(x)

    # ── Dense head ──
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(n_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="Vocalize_BiLSTM_Attention")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    return model


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vocalize — Train Bi-LSTM + Attention sign-language classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", type=str, default="processed_data",
        help="Root of processed .npy data (default: processed_data)",
    )
    parser.add_argument(
        "--seq_len", type=int, default=SEQUENCE_LENGTH,
        help=f"Uniform sequence length after pad/truncate (default: {SEQUENCE_LENGTH})",
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Maximum training epochs (default: 200)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Training batch size (default: 16)",
    )
    parser.add_argument(
        "--model_out", type=str, default="wlasl_15_gloss.keras",
        help="Output path for the saved model (default: wlasl_15_gloss.keras)",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Fraction of data for testing (default: 0.2)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Vocalize — Bi-LSTM + Attention Trainer")
    print("=" * 60)
    print(f"  Data dir  : {os.path.abspath(args.data_dir)}")
    print(f"  Seq length: {args.seq_len}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Test split: {args.test_size}")
    print(f"  Model out : {args.model_out}")
    print(f"  TensorFlow: {tf.__version__}")
    print("=" * 60)

    # ── 1. Load data ──
    print("\n▸ Step 1/5 — Loading dataset")
    sequences, labels, label_map = load_dataset(args.data_dir)

    # ── 2. Pad / truncate ──
    print("\n▸ Step 2/5 — Padding sequences")
    X = pad_sequences_3d(sequences, args.seq_len)
    y = to_categorical(labels, num_classes=len(label_map))
    print(f"[INFO] X shape: {X.shape}  |  y shape: {y.shape}")

    # ── 3. Train / test split ──
    print("\n▸ Step 3/5 — Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y,
    )
    print(f"[INFO] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # ── 4. Build & train ──
    print("\n▸ Step 4/5 — Building model")
    model = build_model(args.seq_len, N_FEATURES, len(label_map))
    model.summary()

    # Save the label map alongside the model for inference
    label_map_path = os.path.splitext(args.model_out)[0] + "_labels.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[INFO] Label map saved → {label_map_path}")

    # Callbacks
    log_dir = os.path.join("logs", "vocalize_bilstm_attn")
    callbacks = [
        EarlyStopping(
            monitor="val_categorical_accuracy",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            args.model_out,
            monitor="val_categorical_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        TensorBoard(log_dir=log_dir, histogram_freq=1),
    ]

    print(f"\n▸ Step 4/5 — Training (TensorBoard logs → {log_dir})")
    print("-" * 60)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 5. Final evaluation ──
    print("\n" + "-" * 60)
    print("▸ Step 5/5 — Evaluating on test set")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test loss     : {loss:.4f}")
    print(f"  Test accuracy : {accuracy:.4f}")

    # ── Summary ──
    best_epoch = np.argmax(history.history["val_categorical_accuracy"]) + 1
    best_val   = max(history.history["val_categorical_accuracy"])

    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"  Best val accuracy : {best_val:.4f}  (epoch {best_epoch})")
    print(f"  Model saved       : {os.path.abspath(args.model_out)}")
    print(f"  Label map saved   : {os.path.abspath(label_map_path)}")
    print(f"  TensorBoard       : tensorboard --logdir {log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
