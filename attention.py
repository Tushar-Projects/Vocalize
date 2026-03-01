"""
Vocalize — Custom Attention Layer

Shared Keras layer used by both train_model.py and the inference router.
Import from here to ensure custom_objects compatibility.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class AttentionLayer(Layer):
    """
    Bahdanau-style additive attention over a sequence of LSTM outputs.

    Given a (batch, timesteps, features) tensor, this layer learns
    which timesteps are most important and returns a single
    (batch, features) context vector as a weighted sum.

    Weights:
        W  — (features, features)  projects hidden states
        b  — (features,)           bias
        u  — (features, 1)         context vector for scoring
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])

        self.W = self.add_weight(
            name="attention_W",
            shape=(feature_dim, feature_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_b",
            shape=(feature_dim,),
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            name="attention_u",
            shape=(feature_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, timesteps, features)
        # Score each timestep
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        # Reduce to scalar per timestep
        attention_weights = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=1), axis=1
        )  # (batch, timesteps, 1)
        # Weighted sum → context vector
        context = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context  # (batch, features)

    def get_config(self):
        return super().get_config()
