import numpy as np
import pytest
import keras
from keras import layers

from deep_lvpm.model import StructuralModel

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - TensorFlow is the expected test backend here
    tf = None


INVALID_MISSING_ERRORS = (ValueError,)
if tf is not None:
    INVALID_MISSING_ERRORS = INVALID_MISSING_ERRORS + (tf.errors.InvalidArgumentError,)


def _make_view_model(input_shape):
    """Create a small encoder used by the missing-view tests."""
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Dense(8, activation="relu"),
        ]
    )


def _make_structural_model(model_list, ndims=3):
    """Build a two-plus-view StructuralModel with symmetric connectivity."""
    keras.utils.set_random_seed(123)
    path = np.ones((len(model_list), len(model_list)), dtype="float32") - np.eye(
        len(model_list), dtype="float32"
    )
    structural = StructuralModel(
        Path=path,
        model_list=model_list,
        regularizer_list=[None for _ in model_list],
        tot_num=32,
        ndims=ndims,
        orthogonalization="zca",
    )
    structural.compile(
        [keras.optimizers.Adam(learning_rate=1e-3) for _ in structural.model_list]
    )
    return structural


def _metric_floats(metrics):
    """Convert backend tensors in a metric dict into plain Python floats."""
    return {
        name: float(keras.ops.convert_to_numpy(value))
        for name, value in metrics.items()
    }


def test_structural_model_scatter_keeps_missing_view_rows_zero():
    """Missing rows should be zero-filled after scattering latent outputs back."""
    rng = np.random.default_rng(123)
    structural = _make_structural_model(
        [_make_view_model((4,)), _make_view_model((4,))]
    )

    view_a = rng.normal(size=(5, 4)).astype("float32")
    view_b = rng.normal(size=(5, 4)).astype("float32")
    view_b[2, :] = np.nan

    latents = structural((view_a, view_b), training=False)
    latents_np = np.asarray(keras.ops.convert_to_numpy(latents))

    assert latents_np.shape == (5, 3, 2)
    assert np.isfinite(latents_np).all()
    np.testing.assert_allclose(latents_np[2, :, 1], 0.0, atol=1e-7)
    assert not np.allclose(latents_np[0, :, 1], 0.0)


def test_structural_model_train_step_ignores_missing_view_rows():
    """Training should stay finite when only some rows are observed for a view."""
    rng = np.random.default_rng(456)
    structural = _make_structural_model(
        [_make_view_model((4,)), _make_view_model((4,))]
    )

    view_a = rng.normal(size=(6, 4)).astype("float32")
    view_b = rng.normal(size=(6, 4)).astype("float32")
    view_b[1, :] = np.nan
    view_b[4, :] = np.nan

    metrics = _metric_floats(structural.train_step(((view_a, view_b),)))

    assert set(metrics) == {"total_loss", "cross_metric", "mse_loss", "redundancy"}
    assert all(np.isfinite(value) for value in metrics.values())


def test_structural_model_test_step_handles_entire_missing_view():
    """Evaluation should skip a view cleanly when the whole batch is missing."""
    rng = np.random.default_rng(789)
    structural = _make_structural_model(
        [_make_view_model((4,)), _make_view_model((4,))]
    )

    view_a = rng.normal(size=(4, 4)).astype("float32")
    view_b = np.full((4, 4), np.nan, dtype="float32")

    metrics = _metric_floats(structural.test_step(((view_a, view_b),)))

    assert metrics["total_loss"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["mse_loss"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["cross_metric"] == pytest.approx(0.0, abs=1e-6)
    assert np.isfinite(metrics["redundancy"])


def test_structural_model_rejects_partial_nan_rows():
    """Partially missing rows should raise instead of being silently masked."""
    rng = np.random.default_rng(321)
    structural = _make_structural_model(
        [_make_view_model((4,)), _make_view_model((4,))]
    )

    view_a = rng.normal(size=(4, 4)).astype("float32")
    view_b = rng.normal(size=(4, 4)).astype("float32")
    view_b[1, 0] = np.nan

    with pytest.raises(INVALID_MISSING_ERRORS, match="partially missing rows"):
        structural((view_a, view_b), training=False)


def test_structural_model_supports_consistent_missing_multi_input_view():
    """Multi-input views should work when all inputs mark the same rows missing."""
    rng = np.random.default_rng(654)
    left = keras.Input(shape=(2,))
    right = keras.Input(shape=(2,))
    merged = layers.Concatenate()([left, right])
    merged = layers.Dense(6, activation="relu")(merged)
    multi_input_model = keras.Model([left, right], merged)

    structural = _make_structural_model(
        [_make_view_model((4,)), multi_input_model]
    )

    view_a = rng.normal(size=(5, 4)).astype("float32")
    view_b_left = rng.normal(size=(5, 2)).astype("float32")
    view_b_right = rng.normal(size=(5, 2)).astype("float32")
    view_b_left[3, :] = np.nan
    view_b_right[3, :] = np.nan

    metrics = _metric_floats(
        structural.train_step(((view_a, view_b_left, view_b_right),))
    )

    assert all(np.isfinite(value) for value in metrics.values())


def test_structural_model_rejects_inconsistent_missing_multi_input_view():
    """Multi-input views should reject mismatched missing-row patterns."""
    rng = np.random.default_rng(987)
    left = keras.Input(shape=(2,))
    right = keras.Input(shape=(2,))
    merged = layers.Concatenate()([left, right])
    merged = layers.Dense(6, activation="relu")(merged)
    multi_input_model = keras.Model([left, right], merged)

    structural = _make_structural_model(
        [_make_view_model((4,)), multi_input_model]
    )

    view_a = rng.normal(size=(5, 4)).astype("float32")
    view_b_left = rng.normal(size=(5, 2)).astype("float32")
    view_b_right = rng.normal(size=(5, 2)).astype("float32")
    view_b_left[2, :] = np.nan

    with pytest.raises(INVALID_MISSING_ERRORS, match="inconsistent missing-row masks"):
        structural((view_a, view_b_left, view_b_right), training=False)
