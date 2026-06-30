"""Small PyTorch TCGA quickstart used by tests and examples."""

from __future__ import annotations

from importlib import resources
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from deep_lvpm.model import StructuralModel
from deep_lvpm import regularizers


DATA_KEYS: Sequence[str] = ("histo20", "rnaseq", "methylation", "mirna", "snv")


def _evaluate_structural_model(model: StructuralModel, data) -> dict[str, float]:
    results = model.evaluate(data, verbose=False)
    return {key: float(value) for key, value in results.items()}


def _load_npz_arrays(filename: str) -> dict[str, np.ndarray]:
    with resources.as_file(resources.files("deep_lvpm.data") / filename) as handle:
        with np.load(handle) as arrays:
            return {key: arrays[key].astype("float32") for key in arrays.files}


def _build_encoder(input_dim: int) -> nn.Sequential:
    hidden = min(256, max(32, input_dim // 8))
    model = nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.GELU(),
        nn.BatchNorm1d(hidden),
        nn.Dropout(0.1),
        nn.Linear(hidden, max(16, hidden // 2)),
        nn.GELU(),
    )
    model.n_inputs = 1
    return model


def run_tcga_quickstart(epochs: int = 8, batch_size: int = 64, verbose: bool = False):
    torch.manual_seed(123)
    train_arrays = _load_npz_arrays("Lung_multiomics_sample_train.npz")
    test_arrays = _load_npz_arrays("Lung_multiomics_sample_test.npz")

    X_train = [train_arrays[key] for key in DATA_KEYS]
    X_test = [test_arrays[key] for key in DATA_KEYS]
    encoders = [_build_encoder(view.shape[1]) for view in X_train]

    path = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 1, 1],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype="float32",
    )

    model = StructuralModel(
        Path=path,
        model_list=encoders,
        regularizer_list=[regularizers.l1_l2(3e-4, 3e-4) for _ in encoders],
        tot_num=X_train[0].shape[0],
        ndims=3,
        orthogonalization="Moore-Penrose",
        momentum=0.9,
        epsilon=1e-3,
        train_DLV=True,
        device="cpu",
    )
    model.compile([torch.optim.Adam(view.parameters(), lr=5e-4) for view in model.model_list])
    model.fit(X_train, batch_size=batch_size, epochs=epochs, verbose=verbose)
    metrics = _evaluate_structural_model(model, X_test)
    return {
        "cross_val": metrics.get("cross_metric"),
        "redundancy": metrics.get("redundancy"),
        "metrics": metrics,
    }


if __name__ == "__main__":
    print(run_tcga_quickstart(verbose=True))
