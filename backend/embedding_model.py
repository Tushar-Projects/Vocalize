"""
Vocalize — Phase 11: Embedding Extractor for KNN Custom Gestures

Wraps the trained Bi-LSTM+Attention classifier and exposes its
`attention` layer output as a 128-dim embedding, for few-shot
gesture matching without retraining the underlying model.

Functions
─────────
build_embedding_model(classifier)
    Create a Keras Model that shares weights with the classifier
    but outputs the `attention` layer activations.

l2_normalize(vec)
    Unit-normalize a vector with an epsilon guard against zero-norm.

nearest_custom_gesture(embedding, candidates)
    Find the closest custom gesture prototype by Euclidean distance
    on L2-normalized embeddings.
"""

import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger("vocalize.embedding")

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
EMBEDDING_DIM = 128  # bilstm_2 is Bidirectional(LSTM(64)) → 64*2


# ──────────────────────────────────────────────
# Embedding model builder
# ──────────────────────────────────────────────

def build_embedding_model(classifier: tf.keras.Model) -> tf.keras.Model:
    """
    Return a new Keras Model sharing weights with `classifier` but
    outputting the `attention` layer's activations instead of the
    softmax head.  No new weights, no training — a view into the
    already-trained graph.

    Args:
        classifier: The loaded Bi-LSTM+Attention model (wlasl_15_gloss.keras).

    Returns:
        A Keras Model with the same input, producing a (batch, D)
        embedding where D is the attention layer's output dim.
    """
    attention_layer = classifier.get_layer("attention")
    embed_model = tf.keras.Model(
        inputs=classifier.input,
        outputs=attention_layer.output,
        name="Vocalize_EmbeddingExtractor",
    )

    # Verify output shape with a warm-up call
    dummy = tf.constant(
        np.zeros((1, classifier.input_shape[1], classifier.input_shape[2]),
                 dtype=np.float32)
    )
    out = embed_model(dummy, training=False).numpy()
    actual_dim = out.shape[-1]

    if actual_dim != EMBEDDING_DIM:
        logger.warning(
            f"Embedding dim mismatch: expected {EMBEDDING_DIM}, "
            f"got {actual_dim}. Using actual dim."
        )

    logger.info(
        f"Embedding model built and warmed up. "
        f"Output shape: {out.shape} (dim={actual_dim})"
    )
    return embed_model


# ──────────────────────────────────────────────
# Distance helpers
# ──────────────────────────────────────────────

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """
    L2-normalize a 1-D vector.

    Uses an epsilon of 1e-8 to prevent division-by-zero on
    degenerate embeddings (e.g. all-zero keypoints from a
    badly-tracked frame).

    Args:
        vec: 1-D numpy array.

    Returns:
        Unit-length numpy array of the same shape.
    """
    return vec / (np.linalg.norm(vec) + 1e-8)


def nearest_custom_gesture(
    embedding: np.ndarray,
    candidates: list[tuple[str, np.ndarray]],
) -> tuple[str | None, float]:
    """
    Find the closest custom gesture prototype by Euclidean distance
    on L2-normalized embeddings.

    The query embedding is L2-normalized here; stored candidate
    embeddings are expected to already be L2-normalized at storage
    time (normalize-per-rep → average → re-normalize pipeline).

    Args:
        embedding:   Raw (unnormalized) 1-D embedding from the
                     attention layer.
        candidates:  List of (label, normalized_embedding) tuples
                     from the in-memory cache.

    Returns:
        (label, distance) of the nearest candidate, or
        (None, float('inf')) if candidates is empty.
    """
    if not candidates:
        return None, float("inf")

    query = l2_normalize(embedding)

    best_label: str | None = None
    best_dist = float("inf")

    for label, proto in candidates:
        dist = float(np.linalg.norm(query - proto))
        if dist < best_dist:
            best_dist = dist
            best_label = label

    return best_label, best_dist
