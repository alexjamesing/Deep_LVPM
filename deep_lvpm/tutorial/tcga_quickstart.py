"""
Abbreviated TCGA tutorial used by the pytest suite.

The function ``run_tcga_quickstart`` trains the structural model for 50 epochs
on the packaged lung cancer sample and returns the evaluation metrics recorded
on the held-out test split.  The helper keeps the footprint small enough to run
inside unit tests while still enforcing minimum quality thresholds.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

import keras
from keras import layers, regularizers

from deep_lvpm.models.StructuralModel import StructuralModel

DATA_KEYS: Sequence[str] = ("histo20", "rnaseq", "methylation", "mirna", "snv")
TRAIN_FILE = "Lung_multiomics_sample_train.npz"
TEST_FILE = "Lung_multiomics_sample_test.npz"


def _evaluate_structural_model(model: StructuralModel, data) -> dict[str, float]:
    """Return eval metrics as plain floats regardless of backend return type."""

    results = model.evaluate(data, verbose=False)
    if isinstance(results, dict):
        return {key: float(value) for key, value in results.items()}
    if isinstance(results, (list, tuple)):
        return {f"metric_{idx}": float(value) for idx, value in enumerate(results)}
    return {"metric_0": float(results)}


def _load_npz_arrays(filename: str) -> dict[str, np.ndarray]:
    """Load NPZ contents into memory so temp files can be discarded safely."""

    candidate = Path(__file__).resolve().parents[1] / "data" / filename
    if candidate.exists():
        with np.load(candidate) as arrays:
            return {key: arrays[key].astype("float32") for key in arrays.files}

    try:
        from importlib import resources
    except ImportError as exc:  # pragma: no cover - python <3.9 fallback
        raise FileNotFoundError(f"Unable to locate {filename}") from exc

    with resources.as_file(resources.files("deep_lvpm.data") / filename) as handle:
        with np.load(handle) as arrays:
            return {key: arrays[key].astype("float32") for key in arrays.files}


def _build_measurement_models(view_shapes: Sequence[np.ndarray]) -> list[keras.Model]:
    """Create lightweight residual-free encoders for each modality."""

    encoders: list[keras.Model] = []
    for view, name in zip(view_shapes, DATA_KEYS):
        input_dim = view.shape[1]
        hidden = min(512, max(64, input_dim // 8))
        projection = max(32, hidden // 2)
        encoder = keras.Sequential(
            [
                layers.InputLayer(shape=(input_dim,), name=f"{name}_in"),
                layers.Dense(
                    hidden,
                    activation="gelu",
                    kernel_regularizer=regularizers.l2(5e-4),
                    name=f"{name}_dense1",
                ),
                layers.BatchNormalization(name=f"{name}_bn1"),
                layers.Dropout(0.2, name=f"{name}_drop1"),
                layers.Dense(
                    projection,
                    activation="gelu",
                    kernel_regularizer=regularizers.l2(5e-4),
                    name=f"{name}_dense2",
                ),
                layers.BatchNormalization(name=f"{name}_bn2"),
                layers.Dropout(0.2, name=f"{name}_drop2"),
            ],
            name=f"{name}_encoder",
        )
        encoders.append(encoder)
    return encoders


def run_tcga_quickstart(
    epochs: int = 50,
    batch_size: int = 64,
    verbose: bool = False,
) -> dict[str, float | dict[str, float]]:
    """
    Train the abbreviated TCGA tutorial and return cross-validation metrics.

    The model always trains for ``epochs`` iterations (default 50) so CI jobs can
    detect regressions that affect convergence speed or output quality.  Results
    are evaluated on the packaged test split and returned as plain floats.
    """

    train_arrays = _load_npz_arrays(TRAIN_FILE)
    test_arrays = _load_npz_arrays(TEST_FILE)
    train_views = [train_arrays[key] for key in DATA_KEYS]
    test_views = [test_arrays[key] for key in DATA_KEYS]

    encoders = _build_measurement_models(train_views)
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
    regularizer_list = [regularizers.L1L2(l1=3e-4, l2=3e-4) for _ in encoders]

    structural_model = StructuralModel(
        Path=adjacency,
        model_list=encoders,
        regularizer_list=regularizer_list,
        tot_num=train_views[0].shape[0],
        ndims=3,
        orthogonalization="Moore-Penrose",
        momentum=0.9,
        epsilon=1e-3,
        train_DLV=True,
    )

    optimizers = [keras.optimizers.Adam(learning_rate=5e-4) for _ in encoders]
    structural_model.compile(optimizer=optimizers)
    structural_model.fit(
        train_views,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1 if verbose else 0,
    )

    metrics = _evaluate_structural_model(structural_model, test_views)
    cross_val = metrics.get("cross_metric", metrics.get("metric_1"))
    redundancy = metrics.get("redundancy", metrics.get("metric_3"))

    return {
        "cross_val": cross_val,
        "redundancy": redundancy,
        "metrics": metrics,
    }


__all__ = [
    "_evaluate_structural_model",
    "run_tcga_quickstart",
]
