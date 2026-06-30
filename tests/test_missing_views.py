import numpy as np
import pytest
import torch
import torch.nn as nn

from deep_lvpm.model import StructuralModel


def _make_encoder(width):
    model = nn.Sequential(nn.Linear(width, 6), nn.ReLU())
    model.n_inputs = 1
    return model


def _make_model(**kwargs):
    path = np.array([[0, 1], [1, 0]], dtype="float32")
    model = StructuralModel(
        Path=path,
        model_list=[_make_encoder(2), _make_encoder(2)],
        regularizer_list=[None, None],
        tot_num=8,
        ndims=2,
        device="cpu",
        **kwargs,
    )
    model.compile([torch.optim.Adam(view.parameters(), lr=1e-3) for view in model.model_list])
    return model


def _make_missing_view_data():
    rng = np.random.default_rng(123)
    data = [
        rng.normal(size=(8, 2)).astype("float32"),
        rng.normal(size=(8, 2)).astype("float32"),
    ]
    data[1][0, :] = np.nan
    data[0][1, :] = np.nan
    data[1][4, :] = np.nan
    data[0][5, :] = np.nan
    return data


def test_all_nan_rows_are_treated_as_missing_views():
    data = [
        np.random.randn(8, 2).astype("float32"),
        np.random.randn(8, 2).astype("float32"),
    ]
    data[1][0:2, :] = np.nan
    model = _make_model()
    latents = model.predict(data, batch_size=4)
    np.testing.assert_allclose(latents[0:2, :, 1], 0.0, atol=1e-6)


def test_project_strategy_preserves_zero_scatter_training_masks():
    data = _make_missing_view_data()
    model = _make_model(missing_strategy="project")

    tensors = model._prepare_tensors(data, device=model.device)
    inputs_nested = model.organize_inputs_by_model(tensors)
    with torch.no_grad():
        y, input_present, target_present = model._training_latents_and_masks(
            inputs_nested,
            encoder_training=False,
        )

    assert torch.equal(input_present, target_present)
    np.testing.assert_allclose(y[0, :, 1].detach().numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(y[1, :, 0].detach().numpy(), 0.0, atol=1e-6)


def test_default_impute_does_not_change_predict_missing_view_output():
    data = _make_missing_view_data()
    model = _make_model()

    latents = model.predict(data, batch_size=4)

    np.testing.assert_allclose(latents[0, :, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(latents[1, :, 0], 0.0, atol=1e-6)


def test_low_rank_completion_preserves_observed_values_and_fills_missing_values():
    model = _make_model(latent_imputation_rank=2, latent_imputation_iterations=3)
    y = torch.tensor(
        [
            [[1.0, 2.0, 0.0], [0.5, 1.0, 0.0]],
            [[2.0, 0.0, 6.0], [1.0, 0.0, 3.0]],
            [[3.0, 4.0, 6.0], [1.5, 2.0, 3.0]],
            [[0.0, 5.0, 8.0], [0.0, 2.5, 4.0]],
        ],
        requires_grad=True,
    )
    input_present = torch.tensor(
        [
            [True, True, False],
            [True, False, True],
            [True, True, True],
            [False, True, True],
        ]
    )

    completed, target_present = model._low_rank_impute_latents(y, input_present)

    assert bool(torch.all(target_present).item())
    observed_mask = input_present.unsqueeze(1).expand_as(completed)
    torch.testing.assert_close(completed[observed_mask], y.detach()[observed_mask])

    missing_mask = (~input_present).unsqueeze(1).expand_as(completed)
    assert torch.all(torch.isfinite(completed[missing_mask]))
    assert bool(torch.any(torch.abs(completed[missing_mask]) > 1e-6).item())


def test_latent_imputation_does_not_complete_view_with_no_observed_rows():
    model = _make_model()
    y = torch.randn(4, 2, 3)
    y[:, :, 2] = 0.0
    input_present = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
            [False, True, False],
            [True, True, False],
        ]
    )

    completed, target_present = model._low_rank_impute_latents(y, input_present)

    assert bool(torch.any(target_present[:, 0]).item())
    assert bool(torch.any(target_present[:, 1]).item())
    assert not bool(torch.any(target_present[:, 2]).item())
    np.testing.assert_allclose(completed[:, :, 2].detach().numpy(), 0.0, atol=1e-6)


def test_latent_imputation_is_detached():
    model = _make_model()
    y = torch.randn(4, 2, 2, requires_grad=True)
    input_present = torch.tensor(
        [
            [True, False],
            [True, True],
            [False, True],
            [True, True],
        ]
    )

    completed, _ = model._low_rank_impute_latents(y, input_present)

    assert completed.requires_grad is False
    assert completed.grad_fn is None


def test_rows_with_no_observed_views_raise_clear_error():
    data = [
        np.random.randn(8, 2).astype("float32"),
        np.random.randn(8, 2).astype("float32"),
    ]
    data[0][0, :] = np.nan
    data[1][0, :] = np.nan
    model = _make_model()

    with pytest.raises(ValueError, match="at least one observed view"):
        model.predict(data, batch_size=4)


def test_partial_nan_rows_raise_clear_error():
    data = [
        np.random.randn(8, 2).astype("float32"),
        np.random.randn(8, 2).astype("float32"),
    ]
    data[1][0, 0] = np.nan
    model = _make_model()
    with pytest.raises(ValueError, match="partially missing"):
        model.predict(data, batch_size=4)


class _TwoInputEncoder(nn.Module):
    n_inputs = 2

    def forward(self, inputs):
        left, right = inputs
        return torch.cat([left, right], dim=1)


def test_multi_input_missing_masks_must_match():
    path = np.array([[0, 1], [1, 0]], dtype="float32")
    model = StructuralModel(
        Path=path,
        model_list=[_make_encoder(2), _TwoInputEncoder()],
        regularizer_list=[None, None],
        tot_num=8,
        ndims=2,
        device="cpu",
    )
    data = [
        np.random.randn(8, 2).astype("float32"),
        np.random.randn(8, 2).astype("float32"),
        np.random.randn(8, 2).astype("float32"),
    ]
    data[1][0, :] = np.nan
    with pytest.raises(ValueError, match="inconsistent"):
        model.predict(data, batch_size=4)


def test_training_with_missing_views_produces_finite_metrics_for_both_strategies():
    for missing_strategy in ["project", "impute"]:
        data = _make_missing_view_data()
        model = _make_model(missing_strategy=missing_strategy)

        history = model.fit(data, batch_size=4, epochs=1, verbose=False, shuffle=False)

        for key in ["total_loss", "cross_metric", "mse_loss", "redundancy"]:
            assert np.isfinite(history[key][-1])
