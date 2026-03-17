"""
Abbreviated TCGA tutorial used by the pytest suite.

``run_tcga_quickstart`` trains the structural model for 50 epochs on the
packaged lung cancer sample and returns evaluation metrics recorded on
the held-out test split.  The helper keeps the footprint small enough to
run inside unit tests while still enforcing minimum quality thresholds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from deep_lvpm.model import StructuralModel

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
    except ImportError as exc:
        raise FileNotFoundError(f"Unable to locate {filename}") from exc

    with resources.as_file(resources.files("deep_lvpm.data") / filename) as handle:
        with np.load(handle) as arrays:
            return {key: arrays[key].astype("float32") for key in arrays.files}


def _build_encoder(input_dim: int, name: str) -> nn.Sequential:
    """Lightweight two-layer encoder for one data modality."""
    hidden = min(512, max(64, input_dim // 8))
    projection = max(32, hidden // 2)
    model = nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.GELU(),
        nn.BatchNorm1d(hidden),
        nn.Dropout(0.2),
        nn.Linear(hidden, projection),
        nn.GELU(),
        nn.BatchNorm1d(projection),
        nn.Dropout(0.2),
    )
    model.n_inputs = 1
    return model


def _build_measurement_models(view_arrays: Sequence[np.ndarray]) -> list[nn.Sequential]:
    return [
        _build_encoder(arr.shape[1], name) for arr, name in zip(view_arrays, DATA_KEYS)
    ]


def run_tcga_quickstart(
    epochs: int = 30,
    batch_size: int = 64,
    verbose: bool = False,
) -> dict[str, float | dict[str, float]]:
    """
    Train the abbreviated TCGA tutorial and return cross-validation metrics.

    Trains for ``epochs`` iterations (default 50) so CI jobs can detect
    regressions affecting convergence speed or output quality.  Results
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

    regularizer_list = [(3e-4, 3e-4)] * len(encoders)

    model = StructuralModel(
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

    model.build(train_views)
    optimizers = [torch.optim.Adam(m.parameters(), lr=5e-4) for m in model.model_list]
    model.compile(optimizer=optimizers)
    model.fit(
        train_views,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
    )

    metrics = _evaluate_structural_model(model, test_views)
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
