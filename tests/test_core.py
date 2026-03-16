"""
Tests for the pure-PyTorch deep_lvpm package.

Mirrors the structure of dlvpm_test_core.py (which tests the Keras
package) but uses nn.Module APIs throughout.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from deep_lvpm.layers.FactorLayer import FactorLayer
from deep_lvpm.model import StructuralModel
from tests.test_tcga import (
    _evaluate_structural_model,
    run_tcga_quickstart,
)

# ---------------------------------------------------------------------------
# 1. FactorLayer basics
# ---------------------------------------------------------------------------


def test_factor_layer_training_updates_state():
    """FactorLayer should emit ndims factors and update moving statistics during training."""
    rng = np.random.default_rng(42)
    factor_layer = FactorLayer(tot_num=20, ndims=3, momentum=0.5)

    batch = torch.tensor(rng.normal(size=(5, 4)).astype("float32"))
    factor_layer.train()
    with torch.no_grad():
        result = factor_layer(batch)

    assert result.shape == (5, 3), f"Unexpected shape {result.shape}"
    assert float(factor_layer.run.item()) == pytest.approx(1.0, rel=1e-6)
    # All outputs should be non-zero
    assert (result.abs() > 0).all(), "Expected non-zero DLV outputs"


# ---------------------------------------------------------------------------
# 2. StructuralModel.organize_inputs_by_model
# ---------------------------------------------------------------------------


class _MultiInputModel(nn.Module):
    """Simple model that concatenates two inputs."""

    def __init__(self):
        super().__init__()
        self.n_inputs = 2

    def forward(self, inputs, training: bool = False):
        x1, x2 = inputs
        return torch.cat([x1, x2], dim=1)


def test_structural_model_organize_inputs_supports_multi_input_models():
    """organize_inputs_by_model should regroup flat data lists per measurement model."""
    seq_model = nn.Sequential(nn.Linear(3, 2))
    seq_model.n_inputs = 1

    multi_model = _MultiInputModel()

    path = np.array([[0, 1], [1, 0]], dtype="float32")
    structural = StructuralModel(
        Path=path,
        model_list=[seq_model, multi_model],
        regularizer_list=[None, None],
        tot_num=10,
        ndims=2,
    )

    data = [
        np.zeros((4, 3), dtype="float32"),
        np.ones((4, 2), dtype="float32"),
        np.full((4, 2), 2.0, dtype="float32"),
    ]
    grouped = structural.organize_inputs_by_model(data)

    assert np.array_equal(grouped[0], data[0])
    assert isinstance(grouped[1], list) and len(grouped[1]) == 2
    np.testing.assert_array_equal(grouped[1][0], data[1])
    np.testing.assert_array_equal(grouped[1][1], data[2])


# ---------------------------------------------------------------------------
# 3. _evaluate_structural_model coerces to float
# ---------------------------------------------------------------------------


def test_evaluate_structural_model_coerces_float_outputs():
    """Tutorial helper should always return plain floats regardless of return type."""

    class DummyModel:
        def evaluate(self, data, verbose=False):
            return {"total_loss": np.float32(1.2), "cross_metric": np.array(0.9)}

    metrics = _evaluate_structural_model(DummyModel(), data=None)

    assert set(metrics) == {"total_loss", "cross_metric"}
    assert all(isinstance(v, float) for v in metrics.values())


# ---------------------------------------------------------------------------
# 4. Full quickstart smoke test
# ---------------------------------------------------------------------------


def test_tcga_quickstart_metrics_exceed_thresholds():
    """Fast TCGA run should keep correlation high and redundancy low."""
    results = run_tcga_quickstart()

    assert results["cross_val"] is not None and results["cross_val"] > 0.5
    assert results["redundancy"] is not None and results["redundancy"] < 0.1
