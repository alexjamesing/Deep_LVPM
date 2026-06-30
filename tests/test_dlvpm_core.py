import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from deep_lvpm.layers.ConfoundLayer import ConfoundLayer
from deep_lvpm.layers.FactorLayer import FactorLayer
from deep_lvpm.layers.ZCALayer import ZCALayer
from deep_lvpm.model import StructuralModel
from deep_lvpm.multi_model import CLIP, DGCCA, LeJEPA, VICReg
from deep_lvpm.tuner import HyperParameters, sample_structural_hparams
from deep_lvpm.tutorial.tcga_quickstart import _evaluate_structural_model


def _make_encoder(input_width, hidden_width=8):
    model = nn.Sequential(nn.Linear(input_width, hidden_width), nn.ReLU())
    model.n_inputs = 1
    return model


def test_confound_layer_training_updates_state():
    rng = np.random.default_rng(123)
    layer = ConfoundLayer(tot_num=20, momentum=0.5)
    layer.train()
    x_batch = torch.tensor(rng.normal(size=(5, 4)).astype("float32"))
    confound_batch = torch.tensor(rng.normal(size=(5, 2)).astype("float32"))
    result = layer([x_batch, confound_batch])

    assert result.shape == (5, 4)
    assert float(layer.run) == pytest.approx(1.0, rel=1e-6)
    assert layer.moving_conv2.shape == (3, 3)
    assert layer.moving_convX.shape == (3, 4)
    assert torch.any(torch.abs(layer.moving_conv2) > 0)
    assert torch.any(torch.abs(layer.moving_convX) > 0)


def test_factor_layer_training_updates_state():
    rng = np.random.default_rng(42)
    layer = FactorLayer(tot_num=20, ndims=3, momentum=0.5)
    layer.train()
    batch = torch.tensor(rng.normal(size=(5, 4)).astype("float32"))
    result = layer(batch)

    assert result.shape == (5, 3)
    assert float(layer.run) == pytest.approx(1.0, rel=1e-6)
    assert torch.all(torch.abs(result) > 0)


def test_zca_newton_schulz_matches_eigendecomposition_inverse_sqrt():
    torch.manual_seed(222)
    matrix = torch.randn(16, 16)
    covariance = matrix.T @ matrix + 0.1 * torch.eye(16)
    layer = ZCALayer(ndims=16, epsilon=1e-6, newton_schulz_iters=20)

    approximate = layer.inv_sqrt_newton_schulz(covariance)

    covariance_sym = 0.5 * (covariance + covariance.T) + layer.epsilon * torch.eye(16)
    eigvals, eigvecs = torch.linalg.eigh(covariance_sym)
    exact = (eigvecs * (1.0 / torch.sqrt(torch.clamp(eigvals, min=layer.epsilon))).unsqueeze(0)) @ eigvecs.T

    relative_error = torch.linalg.norm(approximate - exact) / torch.linalg.norm(exact)
    assert float(relative_error) < 1e-3


def test_structural_model_fit_evaluate_predict():
    torch.manual_seed(123)
    rng = np.random.default_rng(123)
    path = np.array([[0, 1], [1, 0]], dtype="float32")
    model = StructuralModel(
        path,
        [_make_encoder(4), _make_encoder(5)],
        [None, None],
        tot_num=24,
        ndims=3,
        orthogonalization="zca",
        device="cpu",
    )
    model.compile([torch.optim.Adam(view.parameters(), lr=1e-3) for view in model.model_list])
    data = [
        rng.normal(size=(24, 4)).astype("float32"),
        rng.normal(size=(24, 5)).astype("float32"),
    ]
    history = model.fit(data, batch_size=8, epochs=1, verbose=False)
    metrics = model.evaluate(data, batch_size=8, verbose=False)
    predictions = model.predict(data, batch_size=8)

    assert set(history) >= {"total_loss", "cross_metric", "mse_loss", "redundancy"}
    assert set(metrics) >= {"total_loss", "cross_metric", "mse_loss", "redundancy"}
    assert predictions.shape == (24, 3, 2)
    assert np.all(np.isfinite(predictions))


def test_structural_model_train_step_and_test_step_match_metric_names():
    torch.manual_seed(321)
    rng = np.random.default_rng(321)
    path = np.array([[0, 1], [1, 0]], dtype="float32")
    model = StructuralModel(
        path,
        [_make_encoder(4), _make_encoder(4)],
        [None, None],
        tot_num=10,
        ndims=2,
        orthogonalization="zca",
        device="cpu",
    )
    model.compile([torch.optim.Adam(view.parameters(), lr=1e-3) for view in model.model_list])
    batch = [
        rng.normal(size=(10, 4)).astype("float32"),
        rng.normal(size=(10, 4)).astype("float32"),
    ]

    train_metrics = model.train_step((batch,))
    test_metrics = model.test_step((batch,))

    assert set(train_metrics) >= {"total_loss", "cross_metric", "mse_loss", "redundancy"}
    assert set(test_metrics) >= {"total_loss", "cross_metric", "mse_loss", "redundancy"}
    assert all(torch.isfinite(value) for value in train_metrics.values())
    assert all(torch.isfinite(value) for value in test_metrics.values())


def test_structural_model_fit_accepts_pytorch_dataloader():
    torch.manual_seed(111)
    rng = np.random.default_rng(111)
    path = np.array([[0, 1], [1, 0]], dtype="float32")
    model = StructuralModel(
        path,
        [_make_encoder(4), _make_encoder(4)],
        [None, None],
        tot_num=16,
        ndims=2,
        orthogonalization="zca",
        device="cpu",
    )
    model.compile([torch.optim.Adam(view.parameters(), lr=1e-3) for view in model.model_list])
    tensors = [
        torch.tensor(rng.normal(size=(16, 4)).astype("float32")),
        torch.tensor(rng.normal(size=(16, 4)).astype("float32")),
    ]
    loader = DataLoader(TensorDataset(*tensors), batch_size=8, shuffle=False)

    history = model.fit(loader, epochs=1, verbose=False)
    metrics = model.evaluate(loader, verbose=False)

    assert set(history) >= {"total_loss", "cross_metric", "mse_loss", "redundancy"}
    assert set(metrics) >= {"total_loss", "cross_metric", "mse_loss", "redundancy"}


class _MultiInputModel(nn.Module):
    n_inputs = 2

    def forward(self, inputs):
        left, right = inputs
        return torch.cat([left, right], dim=1)


def test_structural_model_organize_inputs_supports_multi_input_models():
    path = np.array([[0, 1], [1, 0]], dtype="float32")
    structural = StructuralModel(
        Path=path,
        model_list=[_make_encoder(3), _MultiInputModel()],
        regularizer_list=[None, None],
        tot_num=10,
        ndims=2,
        device="cpu",
    )
    data = [
        torch.zeros((4, 3)),
        torch.ones((4, 2)),
        torch.full((4, 2), 2.0),
    ]
    grouped = structural.organize_inputs_by_model(data)

    assert torch.equal(grouped[0], data[0])
    assert isinstance(grouped[1], list)
    assert torch.equal(grouped[1][0], data[1])
    assert torch.equal(grouped[1][1], data[2])


def test_missing_view_rows_are_zeroed_in_predictions():
    rng = np.random.default_rng(456)
    path = np.array([[0, 1], [1, 0]], dtype="float32")
    model = StructuralModel(
        path,
        [_make_encoder(4), _make_encoder(4)],
        [None, None],
        tot_num=12,
        ndims=2,
        device="cpu",
    )
    data = [
        rng.normal(size=(12, 4)).astype("float32"),
        rng.normal(size=(12, 4)).astype("float32"),
    ]
    data[1][0:3, :] = np.nan
    preds = model.predict(data, batch_size=6)
    np.testing.assert_allclose(preds[0:3, :, 1], 0.0, atol=1e-6)


def test_multimodal_models_smoke_train_and_predict():
    rng = np.random.default_rng(789)
    data = [
        rng.normal(size=(20, 4)).astype("float32"),
        rng.normal(size=(20, 5)).astype("float32"),
    ]
    for cls in (VICReg, CLIP, DGCCA, LeJEPA):
        kwargs = {"num_slices": 8} if cls is LeJEPA else {}
        model = cls([_make_encoder(4), _make_encoder(5)], [None, None], ndims=3, device="cpu", **kwargs)
        model.compile([torch.optim.Adam(view.parameters(), lr=1e-3) for view in model.model_list])
        history = model.fit(data, batch_size=8, epochs=1, verbose=False)
        preds = model.predict(data, batch_size=8)
        assert history
        assert preds.shape == (20, 3, 2)
        assert np.all(np.isfinite(preds))


def test_multimodal_train_step_and_test_step_match_metric_names():
    rng = np.random.default_rng(654)
    data = [
        rng.normal(size=(12, 4)).astype("float32"),
        rng.normal(size=(12, 5)).astype("float32"),
    ]
    expected_metrics = {
        VICReg: {"total_loss", "cross_metric", "mse_loss", "redundancy"},
        CLIP: {"clip_loss"},
        DGCCA: {"total_loss", "cross_metric", "gcca_loss", "redundancy"},
        LeJEPA: {"total_loss", "cross_metric", "pred_loss", "sigreg_loss", "redundancy"},
    }
    for cls, names in expected_metrics.items():
        kwargs = {"num_slices": 8, "integration_points": 5} if cls is LeJEPA else {}
        model = cls([_make_encoder(4), _make_encoder(5)], [None, None], ndims=3, device="cpu", **kwargs)
        model.compile([torch.optim.Adam(view.parameters(), lr=1e-3) for view in model.model_list])

        train_metrics = model.train_step((data,))
        test_metrics = model.test_step((data,))

        assert set(train_metrics) == names
        assert set(test_metrics) == names
        assert all(torch.isfinite(value) for value in train_metrics.values())
        assert all(torch.isfinite(value) for value in test_metrics.values())


def test_clip_fit_accepts_pytorch_dataloader():
    rng = np.random.default_rng(987)
    tensors = [
        torch.tensor(rng.normal(size=(12, 4)).astype("float32")),
        torch.tensor(rng.normal(size=(12, 4)).astype("float32")),
    ]
    loader = DataLoader(TensorDataset(*tensors), batch_size=6, shuffle=False)
    model = CLIP([_make_encoder(4), _make_encoder(4)], [None, None], ndims=3, device="cpu")
    model.compile([torch.optim.Adam(view.parameters(), lr=1e-3) for view in model.model_list])

    history = model.fit(loader, epochs=1, verbose=False)
    metrics = model.evaluate(loader, verbose=False)

    assert set(history) == {"clip_loss"}
    assert set(metrics) == {"clip_loss"}


def test_tuner_hyperparameter_sampling():
    hp = HyperParameters(seed=1)
    result = sample_structural_hparams(
        hp,
        n_views=2,
        target_view=1,
        current_sparse=[0.0, 0.0],
        current_regularizers=[None, None],
        sparse_config={"values": [0.0, 1e-4]},
        regularizer_config={"choices": ["none", "l2"]},
    )
    assert len(result["sparse_l1_list"]) == 2
    assert len(result["regularizer_list"]) == 2


def test_evaluate_structural_model_coerces_float_outputs():
    class DummyModel:
        def evaluate(self, data, verbose=False):
            return {"total_loss": np.float32(1.2), "cross_metric": np.array(0.9)}

    metrics = _evaluate_structural_model(DummyModel(), data=None)
    assert set(metrics) == {"total_loss", "cross_metric"}
    assert all(isinstance(value, float) for value in metrics.values())
