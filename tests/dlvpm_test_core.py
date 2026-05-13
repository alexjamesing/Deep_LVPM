from pathlib import Path

import numpy as np
import pytest
import keras
from keras import layers

from deep_lvpm.layers.ConfoundLayer import ConfoundLayer
from deep_lvpm.layers.FactorLayer import FactorLayer
from deep_lvpm.model import StructuralModel
from deep_lvpm.multi_model import DGCCA, LeJEPA, VICReg
from deep_lvpm.tutorial.tcga_quickstart import (
    _evaluate_structural_model,
    run_tcga_quickstart,
)


def test_confound_layer_training_updates_state():
    """ConfoundLayer should preserve feature width and update its moving statistics."""
    rng = np.random.default_rng(123)
    x_input = keras.Input(shape=(4,))
    confound_input = keras.Input(shape=(2,))
    confound_layer = ConfoundLayer(tot_num=20, momentum=0.5)
    outputs = confound_layer([x_input, confound_input], training=True)
    model = keras.Model([x_input, confound_input], outputs)

    x_batch = rng.normal(size=(5, 4)).astype("float32")
    confound_batch = rng.normal(size=(5, 2)).astype("float32")
    result = model([x_batch, confound_batch], training=True)

    assert result.shape == (5, 4)
    assert float(confound_layer.run.numpy()) == pytest.approx(1.0, rel=1e-6)
    assert confound_layer.moving_conv2.shape == (3, 3)
    assert confound_layer.moving_convX.shape == (3, 4)
    assert np.any(np.abs(confound_layer.moving_conv2.numpy()) > 0)
    assert np.any(np.abs(confound_layer.moving_convX.numpy()) > 0)


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


def _make_encoder(input_width, hidden_width=8):
    """Create a small encoder for multimodal baseline tests."""
    return keras.Sequential(
        [
            keras.Input(shape=(input_width,)),
            layers.Dense(hidden_width, activation="relu"),
        ]
    )


def _to_metric_floats(metrics):
    """Convert backend tensors returned by Keras metrics into Python floats."""
    return {
        name: float(keras.ops.convert_to_numpy(value))
        for name, value in metrics.items()
    }


def _make_identity_encoder(width):
    """Create an identity encoder so internal loss formulas can be tested exactly."""
    inputs = keras.Input(shape=(width,))
    return keras.Model(inputs, inputs)


def _manual_global_center_prediction_loss(views, num_global_views):
    """Reference LeJEPA prediction loss using the mean of the global views."""
    center = np.mean(np.stack(views[:num_global_views], axis=0), axis=0)
    per_view_losses = [np.mean((center - view) ** 2) for view in views]
    return float(np.mean(per_view_losses))


def _manual_leave_one_out_prediction_loss(views):
    """Reference for the old leave-one-out implementation kept for comparison tests."""
    per_view_losses = []
    for view_index, view in enumerate(views):
        other_views = [other for other_index, other in enumerate(views) if other_index != view_index]
        center = np.mean(np.stack(other_views, axis=0), axis=0)
        per_view_losses.append(np.mean((center - view) ** 2))
    return float(np.mean(per_view_losses))


def test_vicreg_trains_across_all_views_without_path_matrix():
    """VICReg should train using all view pairs without requiring a path matrix."""
    keras.utils.set_random_seed(123)
    rng = np.random.default_rng(123)
    model = VICReg(
        model_list=[_make_encoder(4), _make_encoder(4), _make_encoder(4)],
        regularizer_list=[None, None, None],
        ndims=3,
    )
    model.compile(
        [keras.optimizers.Adam(learning_rate=1e-3) for _ in model.model_list]
    )

    batch = tuple(rng.normal(size=(6, 4)).astype("float32") for _ in range(3))
    metrics = _to_metric_floats(model.train_step((batch,)))

    assert set(metrics) == {"total_loss", "cross_metric", "mse_loss", "redundancy"}
    assert all(np.isfinite(value) for value in metrics.values())
    assert "Path" not in model.get_config()


def test_dgcca_uses_optional_moving_covariance_statistics():
    """DGCCA should store both covariance and projection statistics for clean test-time projection."""
    keras.utils.set_random_seed(321)
    rng = np.random.default_rng(321)
    model = DGCCA(
        model_list=[_make_encoder(4), _make_encoder(4), _make_encoder(4)],
        regularizer_list=[None, None, None],
        ndims=3,
        gcca_reg=1e-2,
        momentum=0.5,
    )
    model.compile(
        [keras.optimizers.Adam(learning_rate=1e-3) for _ in model.model_list]
    )

    batch = tuple(rng.normal(size=(8, 4)).astype("float32") for _ in range(3))
    metrics = _to_metric_floats(model.train_step((batch,)))

    assert set(metrics) == {
        "total_loss",
        "cross_metric",
        "gcca_loss",
        "redundancy",
    }
    assert all(np.isfinite(value) for value in metrics.values())

    moving_covariances_after_train = [
        np.asarray(keras.ops.convert_to_numpy(weight))
        for weight in model._moving_covariances
    ]
    stored_u_after_train = [
        np.asarray(keras.ops.convert_to_numpy(weight))
        for weight in model._stored_u_matrices
    ]
    stored_means_after_train = [
        np.asarray(keras.ops.convert_to_numpy(weight))
        for weight in model._stored_view_means
    ]
    ready_flags = [
        float(keras.ops.convert_to_numpy(flag))
        for flag in model._moving_covariances_ready
    ]
    stored_ready = float(keras.ops.convert_to_numpy(model._stored_projection_ready))
    stored_steps = float(keras.ops.convert_to_numpy(model._stored_projection_steps))

    assert all(flag == pytest.approx(1.0, abs=1e-6) for flag in ready_flags)
    assert any(np.any(np.abs(covariance) > 0.0) for covariance in moving_covariances_after_train)
    assert stored_ready == pytest.approx(1.0, abs=1e-6)
    assert stored_steps == pytest.approx(1.0, abs=1e-6)
    assert any(np.any(np.abs(u_matrix) > 0.0) for u_matrix in stored_u_after_train)
    assert any(np.any(np.abs(view_mean) > 0.0) for view_mean in stored_means_after_train)

    shared_outputs = model.predict(batch, verbose=0)
    assert shared_outputs.shape == (8, 3, 3)
    assert np.all(np.isfinite(shared_outputs))

    shared_outputs_alias = model.predict_shared(batch, verbose=0)
    np.testing.assert_allclose(shared_outputs, shared_outputs_alias, atol=1e-6, rtol=1e-6)

    test_metrics = _to_metric_floats(model.test_step((batch,)))
    moving_covariances_after_test = [
        np.asarray(keras.ops.convert_to_numpy(weight))
        for weight in model._moving_covariances
    ]

    assert set(test_metrics) == {
        "total_loss",
        "cross_metric",
        "gcca_loss",
        "redundancy",
    }
    for before, after in zip(moving_covariances_after_train, moving_covariances_after_test):
        np.testing.assert_allclose(before, after, atol=1e-8, rtol=1e-8)

    config = model.get_config()
    assert config["momentum"] == pytest.approx(0.5, rel=1e-6)


def test_lejepa_prediction_uses_global_view_center():
    """LeJEPA should pull all views toward the mean embedding of the first V_g global views."""
    views = [
        np.array([[2.0, -1.0], [1.5, 0.5]], dtype="float32"),
        np.array([[0.0, 3.0], [-2.0, 4.0]], dtype="float32"),
        np.array([[5.0, 1.0], [0.5, -3.0]], dtype="float32"),
    ]
    model = LeJEPA(
        model_list=[_make_identity_encoder(2) for _ in views],
        regularizer_list=[None for _ in views],
        ndims=2,
        num_global_views=1,
        num_slices=8,
        integration_points=5,
        run_from_config=True,
    )

    stacked = model(tuple(views), training=False)
    pred_loss = float(keras.ops.convert_to_numpy(model._prediction_loss_ops(stacked)))
    expected_global_center = _manual_global_center_prediction_loss(views, num_global_views=1)
    expected_leave_one_out = _manual_leave_one_out_prediction_loss(views)

    assert pred_loss == pytest.approx(expected_global_center, rel=1e-6, abs=1e-6)
    assert pred_loss != pytest.approx(expected_leave_one_out, rel=1e-3, abs=1e-3)


def test_lejepa_trains_using_global_views_without_path_matrix():
    """LeJEPA should default to treating all views as global when V_g is not specified."""
    keras.utils.set_random_seed(456)
    rng = np.random.default_rng(456)
    model = LeJEPA(
        model_list=[_make_encoder(4), _make_encoder(4), _make_encoder(4)],
        regularizer_list=[None, None, None],
        ndims=3,
        num_slices=8,
        integration_points=5,
    )
    model.compile(
        [keras.optimizers.Adam(learning_rate=1e-3) for _ in model.model_list]
    )

    batch = tuple(rng.normal(size=(6, 4)).astype("float32") for _ in range(3))
    metrics = _to_metric_floats(model.train_step((batch,)))
    config = model.get_config()

    assert set(metrics) == {
        "total_loss",
        "cross_metric",
        "pred_loss",
        "sigreg_loss",
        "redundancy",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert "Path" not in config
    assert config["num_global_views"] == len(model.model_list)
    assert "use_path_centers" not in config


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
