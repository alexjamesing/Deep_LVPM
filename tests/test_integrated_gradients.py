import numpy as np
import pytest
import torch
import torch.nn as nn

from deep_lvpm.integrated_gradients import calculate_integrated_gradients
from deep_lvpm.model import StructuralModel


def _make_encoder(input_width, hidden_width=6):
    model = nn.Sequential(nn.Linear(input_width, hidden_width), nn.ReLU())
    model.n_inputs = 1
    return model


def _make_model(data, orthogonalization="zca"):
    path = np.array([[0, 1], [1, 0]], dtype="float32")
    model = StructuralModel(
        Path=path,
        model_list=[_make_encoder(data[0].shape[1]), _make_encoder(data[1].shape[1])],
        regularizer_list=[None, None],
        tot_num=data[0].shape[0],
        ndims=2,
        orthogonalization=orthogonalization,
        device="cpu",
    )
    model.build([torch.as_tensor(value, dtype=torch.float32) for value in data])
    return model


def _make_data(seed=123, n_samples=10):
    rng = np.random.default_rng(seed)
    return [
        rng.normal(size=(n_samples, 3)).astype("float32"),
        rng.normal(size=(n_samples, 4)).astype("float32"),
    ]


def _state_dict_clone(model):
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def _assert_state_dict_equal(left, right):
    assert set(left) == set(right)
    for key in left:
        if left[key].is_floating_point():
            assert torch.allclose(left[key], right[key], atol=1e-6), key
        else:
            assert torch.equal(left[key], right[key]), key


def test_integrated_gradients_returns_one_array_per_input_with_matching_shapes():
    data = _make_data()
    model = _make_model(data)

    attributions = calculate_integrated_gradients(
        model,
        data,
        baseline=0.0,
        dlv_index=0,
        steps=3,
    )

    assert isinstance(attributions, list)
    assert len(attributions) == len(data)
    for attribution, source in zip(attributions, data):
        assert attribution.shape == source.shape
        assert np.all(np.isfinite(attribution))


def test_integrated_gradients_accepts_list_baseline_and_tensor_return():
    data = _make_data(seed=456)
    model = _make_model(data)
    baseline = [np.zeros_like(value) for value in data]

    attributions = calculate_integrated_gradients(
        model,
        data,
        baseline=baseline,
        dlv_index=1,
        steps=2,
        return_numpy=False,
    )

    assert isinstance(attributions, list)
    for attribution, source in zip(attributions, data):
        assert torch.is_tensor(attribution)
        assert tuple(attribution.shape) == source.shape
        assert torch.all(torch.isfinite(attribution))


def test_integrated_gradients_accepts_per_input_vector_baselines():
    data = _make_data(seed=567)
    model = _make_model(data)
    baseline = [
        data[0].mean(axis=0).astype("float32"),
        np.zeros(data[1].shape[1], dtype="float32"),
    ]

    attributions = calculate_integrated_gradients(
        model,
        data,
        baseline=baseline,
        dlv_index=0,
        steps=2,
    )

    assert isinstance(attributions, list)
    for attribution, source in zip(attributions, data):
        assert attribution.shape == source.shape
        assert np.all(np.isfinite(attribution))


def test_integrated_gradients_restores_model_state_dict():
    data = _make_data(seed=789)
    model = _make_model(data)
    before = _state_dict_clone(model)

    calculate_integrated_gradients(
        model,
        data,
        baseline=0.0,
        dlv_index=0,
        steps=3,
    )

    after = _state_dict_clone(model)
    _assert_state_dict_equal(before, after)


def test_integrated_gradients_rejects_invalid_arguments():
    data = _make_data(seed=987)
    model = _make_model(data)

    with pytest.raises(ValueError, match="dlv_index"):
        calculate_integrated_gradients(model, data, dlv_index=99)

    with pytest.raises(ValueError, match="steps"):
        calculate_integrated_gradients(model, data, steps=0)

    integer_data = [data[0].astype("int64"), data[1]]
    with pytest.raises(TypeError, match="floating-point"):
        calculate_integrated_gradients(model, integer_data)

    bad_baseline = [np.zeros((2, 2), dtype="float32"), np.zeros_like(data[1])]
    with pytest.raises(ValueError, match="baseline\\[0\\] shape"):
        calculate_integrated_gradients(model, data, baseline=bad_baseline)


def test_integrated_gradients_handles_all_nan_missing_rows():
    data = _make_data(seed=654, n_samples=12)
    data[1][0:3, :] = np.nan
    model = _make_model(data)

    attributions = calculate_integrated_gradients(
        model,
        data,
        baseline=0.0,
        dlv_index=0,
        steps=3,
    )

    assert np.all(np.isfinite(attributions[0]))
    assert np.all(np.isfinite(attributions[1]))
    np.testing.assert_allclose(attributions[1][0:3, :], 0.0, atol=1e-6)
