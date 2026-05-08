#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Siamese CIFAR-10 with VICReg (TensorFlow backend)

This tutorial mirrors the Siamese CIFAR-10 examples but uses the
VICReg multi-view model (deep_lvpm.multi_model.VICReg). We create two
augmented views per image, feed them through a shared-weight encoder,
and train with VICReg losses across both views. After training, we
linearly probe the learned embeddings.
"""

import os
import random

# Force TensorFlow backend before importing keras.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

# Prefer deterministic, memory-efficient execution.
os.environ.update(
    {
        "TF_XLA_FLAGS": "--tf_xla_auto_jit=0",
        "XLA_FLAGS": "--xla_gpu_strict_conv_algorithm_picker=false",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        "TF_DETERMINISTIC_OPS": "1",
        "TF_CUDNN_DETERMINISTIC": "1",
        "TF_CUDNN_AUTOTUNE_DEFAULT": "0",
        "TF_CUDNN_USE_FRONTEND": "0",
        "NVIDIA_TF32_OVERRIDE": "0",
    }
)

import numpy as np
import tensorflow as tf
import keras
from keras import layers, mixed_precision, Sequential
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from deep_lvpm.multi_model import VICReg


# Keep computations in float32 for stability with siamese objective.
mixed_precision.set_global_policy("float32")

# Enable on-demand GPU allocation.
for device in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(device, True)
    except Exception:
        pass

# Use graph execution for performance.
tf.config.run_functions_eagerly(False)


# Dataset metadata
NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)

# Load CIFAR-10
(x_train, y_train_cat), (x_test, y_test_cat) = keras.datasets.cifar10.load_data()
y_train_cat = y_train_cat.squeeze()
y_test_cat = y_test_cat.squeeze()

# Normalise images to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot labels (not used by VICReg, but useful downstream)
y_train = keras.utils.to_categorical(y_train_cat, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test_cat, NUM_CLASSES)


# Reproducibility
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

# Train/validation split
VAL_FRACTION = 0.1
num_train = x_train.shape[0]
indices = np.arange(num_train)
rng = np.random.default_rng(SEED)
rng.shuffle(indices)
cutoff = int(num_train * (1 - VAL_FRACTION))
x_tr, x_val = x_train[indices[:cutoff]], x_train[indices[cutoff:]]


# Siamese augmentations
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 2048

augment = Sequential(
    [
        layers.RandomCrop(24, 24),
        layers.Resizing(32, 32),
        layers.RandomFlip("horizontal"),
        layers.Lambda(
            lambda x: tf.where(
                tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) < 0.1,
                tf.tile(tf.image.rgb_to_grayscale(x), [1, 1, 1, 3]),
                x,
            )
        ),
    ],
    name="augment",
)


def make_siamese_views_dataset(x, batch_size=256, shuffle=True, training=True):
    """Return a dataset that yields pairs of augmented views: ([v1, v2],)."""
    ds = tf.data.Dataset.from_tensor_slices(x)
    if shuffle:
        ds = ds.shuffle(len(x), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(int(batch_size), drop_remainder=training)

    def map_batch(batch):
        view_one = augment(batch, training=training)
        view_two = augment(batch, training=training)
        return ([view_one, view_two],)

    return ds.map(map_batch, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


# Datasets
train_ds = make_siamese_views_dataset(x_tr, batch_size=BATCH_SIZE, shuffle=False, training=True)
val_ds = make_siamese_views_dataset(x_val, batch_size=BATCH_SIZE, shuffle=False, training=True)


# Shared encoder backbone (VICReg adds its own Dense projection head)
WEIGHT_DECAY = 1e-4
NDIMS = 512  # VICReg projection size


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        x = inputs
        attn_output = self.att(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEmbedding(layers.Layer):
    def __init__(self, seq_len: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(1, self.seq_len, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        return x + self.pos_emb


def build_hybrid_backbone():
    inputs = keras.Input(shape=INPUT_SHAPE)
    # Conv stem: 32->16->8
    x = layers.Conv2D(64, 3, padding="same", activation="relu",
                      kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY))(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu",
                      kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY))(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu",
                      kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY))(x)

    # Tokens: (B, 8, 8, 256) -> (B, 64, 256)
    seq_len = 64
    embed_dim = 256
    x = layers.Reshape((seq_len, embed_dim))(x)
    x = PositionalEmbedding(seq_len, embed_dim, name="positional_embedding")(x)

    # Simplified transformer block (num_heads=4)
    x = TransformerBlock(embed_dim, num_heads=4, ff_dim=512, rate=0.1, name="tx_block1")(x)

    # Pool tokens and produce backbone features (VICReg adds the NDIMS projection)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.BatchNormalization()(x)

    return keras.Model(inputs, outputs, name="cifar_image_backbone")


CIFAR_image_backbone = build_hybrid_backbone()


# Build VICReg siamese model with shared weights across the two views
model_list = [CIFAR_image_backbone, CIFAR_image_backbone]
proj_regs = [keras.regularizers.l2(WEIGHT_DECAY), keras.regularizers.l2(WEIGHT_DECAY)]

vic_model = VICReg(
    model_list=model_list,
    regularizer_list=proj_regs,
    ndims=NDIMS,
    is_siamese=True,
    cov_weight = 50.0
)

# Compile with per-branch optimizers (identical settings)
opt_list = [
    keras.optimizers.Adam(learning_rate=1e-4),
    keras.optimizers.Adam(learning_rate=1e-4),
]
vic_model.compile(opt_list)


# Train
EPOCHS = 10
history = vic_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=True)


# Evaluate VICReg metrics on held-out augmented pairs
test_ds = make_siamese_views_dataset(x_test, batch_size=BATCH_SIZE, shuffle=False, training=True)
eval_metrics = vic_model.evaluate(test_ds, verbose=True)
print("Evaluation metrics:", eval_metrics)


# Downstream linear evaluation using the learned VICReg projection
def l2_normalize(arr: np.ndarray, axis: int = 1, eps: float = 1e-7) -> np.ndarray:
    denom = np.linalg.norm(arr, axis=axis, keepdims=True)
    return arr / (denom + eps)


image_head = vic_model.model_list[0]

train_emb = image_head.predict(x_train, batch_size=64, verbose=1)
test_emb = image_head.predict(x_test, batch_size=64, verbose=1)

train_feats = l2_normalize(train_emb)
test_feats = l2_normalize(test_emb)

svm_clf = Pipeline(
    [
        ("scaler", StandardScaler(with_mean=True)),
        ("svm", LinearSVC(C=1.0, max_iter=10000, random_state=42)),
    ]
)
svm_clf.fit(train_feats, y_train_cat)
pred = svm_clf.predict(test_feats)
acc = accuracy_score(y_test_cat, pred)

print(f"\nSVM accuracy on CIFAR-10 test set (VICReg embeddings): {acc:.4f}\n")
print("Classification report:")
print(classification_report(y_test_cat, pred, digits=4))
print("Confusion matrix:")
print(confusion_matrix(y_test_cat, pred))
