from pathlib import Path

import numpy as np
import pytest
import keras
from keras import layers

from deep_lvpm.layers.FactorLayer import FactorLayer
from deep_lvpm.model import StructuralModel
from deep_lvpm.tutorial.tcga_quickstart import (
    _evaluate_structural_model,
    run_tcga_quickstart,
)


def test_factor_layer_training_updates_state():
    """FactorLayer should emit ndims factors and update moving statistics during training."""
    rng = np.random.default_rng(42)
    inputs = keras.Input(shape=(4,))
    factor_layer = FactorLayer(tot_num=20, ndims=3, momentum=0.5)
    outputs = factor_layer(inputs, training=True)
    model = keras.Model(inputs, outputs)

    batch = rng.normal(size=(5, 4)).astype("float32")
    result = model(batch, training=True)

    assert result.shape == (5, 3)
    assert float(factor_layer.run.numpy()) == pytest.approx(1.0, rel=1e-6)
    np.testing.assert_array_less(np.zeros_like(result), np.abs(result))


def test_structural_model_organize_inputs_supports_multi_input_models():
    """StructuralModel.organize_inputs_by_model should regroup flat data lists per measurement model."""
    seq_model = keras.Sequential([keras.Input(shape=(3,)), layers.Dense(2)])
    left = keras.Input(shape=(2,))
    right = keras.Input(shape=(2,))
    concat = layers.Concatenate()([left, right])
    multi_model = keras.Model([left, right], concat)

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

    assert np.shares_memory(grouped[0], data[0])
    assert isinstance(grouped[1], list) and len(grouped[1]) == 2
    np.testing.assert_array_equal(grouped[1][0], data[1])
    np.testing.assert_array_equal(grouped[1][1], data[2])


def test_evaluate_structural_model_coerces_float_outputs():
    """Tutorial helper should always return plain floats regardless of backend return type."""
    class DummyModel:
        def evaluate(self, data, verbose=False):
            return {"total_loss": np.float32(1.2), "cross_metric": np.array(0.9)}

    metrics = _evaluate_structural_model(DummyModel(), data=None)

    assert set(metrics) == {"total_loss", "cross_metric"}
    assert all(isinstance(value, float) for value in metrics.values())


def test_tcga_quickstart_metrics_exceed_thresholds():
    """Fast TCGA run should keep correlation high and redundancy low."""

    results = run_tcga_quickstart()

    assert results["cross_val"] is not None and results["cross_val"] > 0.5
    assert results["redundancy"] is not None and results["redundancy"] < 0.1
