"""
TCGA lung cancer tutorial using Deep LVPM with the TensorFlow backend.

Run from the command line:

    python -m deep_lvpm.tutorial.tutorial_tcga_tf

This script integrates five modalities (histology features, RNA-seq, DNA
methylation, miRNA, and somatic mutations) using small residual encoders and
reports the StructuralModel evaluation metrics.
"""

from __future__ import annotations

import os
from importlib import resources

# Use the TensorFlow backend for Keras 3.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf

import keras
from keras import layers, regularizers
from keras.optimizers import Adam, schedules

from deep_lvpm.models.StructuralModel import StructuralModel


def _evaluate_structural_model(model: StructuralModel, data) -> dict[str, float]:
    """Return eval metrics as plain floats regardless of backend response type."""

    results = model.evaluate(data, verbose=False)
    if isinstance(results, dict):
        return {key: float(value) for key, value in results.items()}
    return {f"metric_{idx}": float(val) for idx, val in enumerate(results)}


def _load_tcga_sample() -> list[np.ndarray]:
    """Load the packaged TCGA sample arrays shipped with the toolbox."""

    with resources.as_file(resources.files("deep_lvpm.data") / "Lung_multiomics_sample_train.npz") as train_file:
        arrays = np.load(train_file)
        rnaseq = arrays["rnaseq"]
        snv = arrays["snv"]
        methylation = arrays["methylation"]
        mirna = arrays["mirna"]
        histo20 = arrays["histo20"]
    return [histo20, rnaseq, methylation, mirna, snv]


def _residual_block(input_dim: int, name: str) -> keras.Model:
    """Build a shallow residual encoder for a tabular modality."""

    inputs = keras.Input(shape=(input_dim,), name=f"{name}_in")
    x = layers.Dense(
        input_dim,
        activation="linear",
        kernel_initializer=keras.initializers.Identity(),
        kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
        name=f"{name}_dense1",
    )(inputs)
    x = layers.BatchNormalization(momentum=0.9, name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    x = layers.Dense(
        input_dim,
        activation="linear",
        kernel_initializer=keras.initializers.Identity(),
        kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
        name=f"{name}_dense2",
    )(x)
    x = layers.Add(name=f"{name}_add")([inputs, x])
    x = layers.Dropout(0.5, name=f"{name}_drop")(x)
    return keras.Model(inputs=inputs, outputs=x, name=f"{name}_encoder")


if __name__ == "__main__":
    print("=== Deep LVPM TCGA tutorial (TensorFlow backend) ===")
    print(f"Keras backend: {keras.backend.backend()}")
    print(f"Physical GPUs: {tf.config.list_physical_devices('GPU')}")

    # ------------------------------------------------------------------
    # Step 1. Load the packaged multi-omics training views.
    # ------------------------------------------------------------------
    # ``_load_tcga_sample`` returns the five modalities shipped with the repo.
    # Keeping them in a fixed order is important so the encoders, optimiser
    # assignments, and adjacency matrix stay aligned.
    views = _load_tcga_sample()
    view_names = ["histo20", "rnaseq", "methylation", "mirna", "snv"]

    # ------------------------------------------------------------------
    # Step 2. Build the measurement encoders (one per modality).
    # ------------------------------------------------------------------
    # Each encoder is a shallow residual MLP that preserves the dimensionality
    # of its input.  Residual connections help stabilise FactorLayer training
    # on tabular data without significant feature engineering.
    encoders = [_residual_block(view.shape[1], name) for view, name in zip(views, view_names)]

    # ------------------------------------------------------------------
    # Step 3. Define the structural adjacency and training schedule.
    # ------------------------------------------------------------------
    # The adjacency matrix expresses a hub-and-spoke structure where the second
    # factor (F2) connects bidirectionally with the remaining factors.  A gentle
    # exponential learning-rate decay keeps optimisation stable over 300 epochs.
    adjacency = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 1, 1],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype="float32",
    )

    batch_size = 256
    epochs = 300
    total_steps = max(1, (views[0].shape[0] // batch_size) * epochs)
    lr_schedule = schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=total_steps,
        decay_rate=1e-5 / 1e-4,
        staircase=False,
    )

    regulariser_list = [regularizers.L1L2(l1=1e-2, l2=1e-2) for _ in encoders]

    # ``tot_num`` is inferred from any view (they all share the same sample set).
    model = StructuralModel(
        Path=adjacency,
        model_list=encoders,
        regularizer_list=regulariser_list,
        tot_num=views[0].shape[0],
        ndims=5,
        orthogonalization="Moore-Penrose",
        momentum=0.95,
        epsilon=1e-3,
        train_DLV=True,
    )

    # One Adam optimiser per modality, all driven by the same decay schedule.
    optimisers = [Adam(learning_rate=lr_schedule) for _ in encoders]
    model.compile(optimizer=optimisers)

    # ------------------------------------------------------------------
    # Step 4. Fit the model and monitor optimisation history.
    # ------------------------------------------------------------------
    history = model.fit(
        views,
        batch_size=batch_size,
        epochs=epochs,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Step 5. Inspect metrics and latent representations.
    # ------------------------------------------------------------------
    metrics = _evaluate_structural_model(model, views)
    print("Training metrics:", metrics)
    print("Training history keys:", list(history.history))

    latent = model.predict(views, verbose=False)
    print("Latent tensor shape:", latent.shape)
