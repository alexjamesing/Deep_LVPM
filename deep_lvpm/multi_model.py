#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Alternative multiview objectives implemented in native PyTorch."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from deep_lvpm import regularizers
from deep_lvpm.model import _as_tensor, _call_module, _default_device


@contextmanager
def _temporary_mode(module: nn.Module, training: bool):
    old_mode = module.training
    module.train(training)
    try:
        yield
    finally:
        module.train(old_mode)


class ProjectionView(nn.Module):
    """A user encoder followed by a lazy linear projection head."""

    def __init__(self, encoder: nn.Module, ndims: int, regularizer=None, name="projection"):
        super().__init__()
        self.encoder = encoder
        self.projection = nn.LazyLinear(int(ndims))
        self.regularizer = regularizer
        self.name = name
        self.n_inputs = int(getattr(encoder, "n_inputs", 1))

    def forward(self, inputs):
        x = _call_module(self.encoder, inputs)
        if x.ndim > 2:
            x = torch.flatten(x, start_dim=1)
        return self.projection(x)

    def regularization_loss(self, reference=None):
        loss = regularizers.penalty(self.projection.parameters(), self.regularizer, reference=reference)
        if hasattr(self.encoder, "regularization_loss"):
            loss = loss + self.encoder.regularization_loss().to(dtype=loss.dtype, device=loss.device)
        return loss


class _MultiViewModel(nn.Module):
    metric_names = ("total_loss", "cross_metric", "mse_loss", "redundancy")

    def __init__(self, model_list, regularizer_list, ndims, run_from_config=False, is_siamese=False, device=None):
        super().__init__()
        self.ndims = int(ndims)
        self.regularizer_list = list(regularizer_list)
        self.is_siamese = bool(is_siamese)
        self.device = torch.device(device) if device is not None else _default_device()
        self.optimizers = None

        if run_from_config:
            wrapped_models = list(model_list)
        elif self.is_siamese:
            wrapped = self._add_projection(model_list[0], regularizer_list[0])
            wrapped_models = [wrapped for _ in model_list]
        else:
            wrapped_models = [
                self._add_projection(model, regularizer)
                for model, regularizer in zip(model_list, regularizer_list)
            ]
        self.model_list = nn.ModuleList(wrapped_models)

    def _add_projection(self, model, regularizer):
        if not isinstance(model, nn.Module):
            raise ValueError("Each measurement model must be a torch.nn.Module.")
        return ProjectionView(model, self.ndims, regularizer=regularizer)

    def compile(self, optimizer):
        if isinstance(optimizer, (list, tuple)):
            if len(optimizer) != len(self.model_list):
                raise ValueError("When optimizer is a list, it must have one optimizer per model.")
            self.optimizers = list(optimizer)
        elif isinstance(optimizer, torch.optim.Optimizer):
            self.optimizers = [optimizer]
        else:
            raise ValueError("optimizer must be a torch optimizer or a list of torch optimizers.")

    def _optimizer_param_ids(self, optimizer):
        ids = set()
        for group in optimizer.param_groups:
            for param in group["params"]:
                ids.add(id(param))
        return ids

    def _ensure_optimizer_has_parameters(self, optimizer, parameters):
        existing = self._optimizer_param_ids(optimizer)
        missing = [param for param in parameters if param.requires_grad and id(param) not in existing]
        if missing:
            optimizer.add_param_group({"params": missing})

    def _prepare_tensors(self, data, device=None):
        if isinstance(data, TensorDataset):
            return [_as_tensor(tensor, device=device) for tensor in data.tensors]
        if isinstance(data, tuple) and len(data) == 1:
            data = data[0]
        if not isinstance(data, (list, tuple)):
            raise TypeError("Data must be a list or tuple containing one tensor/array per model input.")
        return [_as_tensor(value, device=device) for value in data]

    def _batch_to_inputs(self, batch, device=None):
        if isinstance(batch, (list, tuple)) and len(batch) == 1 and isinstance(batch[0], (list, tuple)):
            batch = batch[0]
        if not isinstance(batch, (list, tuple)):
            batch = [batch]
        return [_as_tensor(value, device=device) for value in batch]

    def _make_loader(self, data, batch_size=32, shuffle=False, drop_last=False):
        if isinstance(data, DataLoader):
            return data
        tensors = self._prepare_tensors(data)
        dataset = TensorDataset(*tensors)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=bool(shuffle),
            drop_last=bool(drop_last),
        )

    def organize_inputs_by_model(self, data_inputs):
        organized_inputs = []
        data_index = 0
        for model in self.model_list:
            num_inputs = int(getattr(model, "n_inputs", 1))
            if num_inputs == 1:
                organized_inputs.append(data_inputs[data_index])
                data_index += 1
            else:
                organized_inputs.append(list(data_inputs[data_index:data_index + num_inputs]))
                data_index += num_inputs
        return organized_inputs

    def _encode_all(self, inputs, training):
        inputs_nested = self.organize_inputs_by_model(inputs)
        outputs = []
        for view_index, model in enumerate(self.model_list):
            use_training = bool(training) and inputs_nested[view_index][0].shape[0] > 1 if isinstance(inputs_nested[view_index], list) else bool(training) and inputs_nested[view_index].shape[0] > 1
            with _temporary_mode(model, use_training):
                outputs.append(model(inputs_nested[view_index]))
        return torch.stack(outputs, dim=-1)

    def forward(self, inputs, training=False):
        return self._encode_all(inputs, training=training)

    def _regularization_loss(self, reference):
        total = torch.zeros((), dtype=reference.dtype, device=reference.device)
        for model in self.model_list:
            total = total + model.regularization_loss(reference=reference)
        return total

    def _corr_pair(self, a, b):
        eps = torch.as_tensor(getattr(self, "eps", 1e-6), dtype=a.dtype, device=a.device)
        a_c = a - torch.mean(a, dim=0)
        b_c = b - torch.mean(b, dim=0)
        a_n = a_c / (torch.sqrt(torch.sum(torch.square(a_c), dim=0)) + eps)
        b_n = b_c / (torch.sqrt(torch.sum(torch.square(b_c), dim=0)) + eps)
        return torch.mean(torch.sum(a_n * b_n, dim=0))

    def calculate_redundancy(self, Y, epsilon=1e-8):
        Y = Y.float()
        col_mean = torch.mean(Y, dim=0, keepdim=True)
        Yc = Y - col_mean
        denom_n = max(Y.shape[0] - 1, 1)
        cov = Yc.T @ Yc / float(denom_n)
        var = torch.sum(Yc * Yc, dim=0) / float(denom_n)
        std = torch.sqrt(torch.clamp(var, min=epsilon))
        corr = cov / torch.clamp(std.reshape(-1, 1) * std.reshape(1, -1), min=epsilon)
        mask = torch.ones_like(corr) - torch.eye(corr.shape[0], dtype=corr.dtype, device=corr.device)
        num_pairs = max(corr.shape[0] * (corr.shape[0] - 1), 1)
        return torch.sum(torch.abs(corr) * mask) / float(num_pairs)

    def calculate_corrmat(self, DLVs):
        if not torch.is_tensor(DLVs):
            DLVs = torch.as_tensor(DLVs, dtype=torch.float32, device=self.device)
        correlation_matrices = []
        n_samples = torch.as_tensor(DLVs.shape[0], dtype=DLVs.dtype, device=DLVs.device)
        eps = torch.as_tensor(1e-7, dtype=DLVs.dtype, device=DLVs.device)
        for dim_index in range(DLVs.shape[1]):
            dim_DLVs = DLVs[:, dim_index, :]
            mean_centered = dim_DLVs - torch.mean(dim_DLVs, dim=0)
            std_dev = torch.std(dim_DLVs, dim=0, unbiased=False) + eps
            normalized = mean_centered / std_dev
            correlation_matrices.append(normalized.T @ normalized / n_samples)
        return correlation_matrices

    def _pairwise_cross_metric(self, Z):
        total = torch.zeros((), dtype=Z.dtype, device=Z.device)
        count = 0
        for i in range(len(self.model_list)):
            for j in range(i + 1, len(self.model_list)):
                total = total + self._corr_pair(Z[:, :, i], Z[:, :, j])
                count += 1
        return total / max(count, 1)

    def _redundancy_ops(self, Z):
        total = torch.zeros((), dtype=Z.dtype, device=Z.device)
        for view_index in range(len(self.model_list)):
            total = total + self.calculate_redundancy(Z[:, :, view_index])
        return total / float(len(self.model_list))

    def _compute_losses(self, inputs, training):
        raise NotImplementedError

    def _metrics_to_float(self, metrics):
        return {key: float(value.detach().cpu()) for key, value in metrics.items()}

    def _build_from_batch(self, batch_tensors):
        self.to(self.device)
        inputs = [tensor.to(self.device) for tensor in batch_tensors]
        with torch.no_grad():
            self._compute_losses(inputs, training=False)
        if self.optimizers is not None:
            all_params = list(self.parameters())
            if len(self.optimizers) == 1:
                self._ensure_optimizer_has_parameters(self.optimizers[0], all_params)
            else:
                for optimizer, model in zip(self.optimizers, self.model_list):
                    self._ensure_optimizer_has_parameters(optimizer, model.parameters())
                self._ensure_optimizer_has_parameters(self.optimizers[0], [p for n, p in self.named_parameters() if n == "logit_scale"])

    def train_step(self, data):
        if self.optimizers is None:
            raise RuntimeError("Call compile(optimizer) before train_step().")

        inputs = self._batch_to_inputs(data, device=self.device)
        self._build_from_batch(inputs)
        self.train()
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
        metrics = self._compute_losses(inputs, training=True)
        loss_key = "total_loss" if "total_loss" in metrics else self.metric_names[0]
        metrics[loss_key].backward()
        for optimizer in self.optimizers:
            optimizer.step()
        return metrics

    def test_step(self, data):
        inputs = self._batch_to_inputs(data, device=self.device)
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            return self._compute_losses(inputs, training=False)

    def fit(self, X_train, batch_size=32, epochs=10, verbose=True, validation_data=None, shuffle=True, **kwargs):
        del kwargs
        if self.optimizers is None:
            raise RuntimeError("Call compile(optimizer) before fit().")

        if isinstance(X_train, DataLoader):
            loader = X_train
        else:
            tensors = self._prepare_tensors(X_train)
            loader = self._make_loader(
                tensors,
                batch_size=batch_size,
                shuffle=bool(shuffle),
                drop_last=tensors[0].shape[0] > batch_size,
            )
        if len(loader) == 0:
            raise ValueError("Training data produced no batches.")

        self._build_from_batch(self._batch_to_inputs(next(iter(loader)), device=self.device))
        history = {name: [] for name in self.metric_names}

        for epoch in range(int(epochs)):
            self.train()
            sums = {name: 0.0 for name in self.metric_names}
            batch_count = 0

            for batch_tensors in loader:
                inputs = self._batch_to_inputs(batch_tensors, device=self.device)
                for optimizer in self.optimizers:
                    optimizer.zero_grad(set_to_none=True)
                metrics = self._compute_losses(inputs, training=True)
                loss_key = "total_loss" if "total_loss" in metrics else self.metric_names[0]
                metrics[loss_key].backward()
                for optimizer in self.optimizers:
                    optimizer.step()
                for name in self.metric_names:
                    sums[name] += float(metrics[name].detach().cpu())
                batch_count += 1

            for name in self.metric_names:
                history[name].append(sums[name] / max(batch_count, 1))

            if validation_data is not None:
                val_metrics = self.evaluate(validation_data, batch_size=batch_size, verbose=False)
                for name, value in val_metrics.items():
                    history.setdefault(f"val_{name}", []).append(value)

            if verbose:
                pieces = [f"{name}: {history[name][-1]:.5f}" for name in self.metric_names]
                print(f"Epoch {epoch + 1}/{epochs} - " + " - ".join(pieces))

        return history

    def evaluate(self, data, batch_size=256, verbose=True, **kwargs):
        del kwargs
        loader = self._make_loader(data, batch_size=batch_size, shuffle=False, drop_last=False)
        self.to(self.device)
        self.eval()

        sums = {name: 0.0 for name in self.metric_names}
        batch_count = 0
        with torch.no_grad():
            for batch_tensors in loader:
                inputs = self._batch_to_inputs(batch_tensors, device=self.device)
                metrics = self._compute_losses(inputs, training=False)
                for name in self.metric_names:
                    sums[name] += float(metrics[name].detach().cpu())
                batch_count += 1

        metrics = {name: sums[name] / max(batch_count, 1) for name in self.metric_names}
        if verbose:
            pieces = [f"{name}: {metrics[name]:.5f}" for name in self.metric_names]
            print("Eval - " + " - ".join(pieces))
        return metrics

    def predict(self, data, batch_size=256, verbose=0, **kwargs):
        del verbose, kwargs
        loader = self._make_loader(data, batch_size=batch_size, shuffle=False, drop_last=False)
        self.to(self.device)
        self.eval()

        chunks = []
        with torch.no_grad():
            for batch_tensors in loader:
                inputs = self._batch_to_inputs(batch_tensors, device=self.device)
                chunks.append(self.forward(inputs, training=False).detach().cpu())
        return torch.cat(chunks, dim=0).numpy()

    def get_config(self):
        return {
            "ndims": self.ndims,
            "is_siamese": self.is_siamese,
            "regularizer_list": self.regularizer_list,
        }

    def save(self, path):
        torch.save({"config": self.get_config(), "state_dict": self.state_dict()}, path)


class CLIP(_MultiViewModel):
    metric_names = ("clip_loss",)

    def __init__(self, model_list, regularizer_list, ndims, run_from_config=False, is_siamese=False, device=None, **kwargs):
        super().__init__(model_list, regularizer_list, ndims, run_from_config, is_siamese, device=device)
        self.logit_scale = nn.Parameter(torch.as_tensor(float(np.log(1.0 / 0.07)), dtype=torch.float32))

    def forward(self, inputs, training=False):
        Z = self._encode_all(inputs, training=training)
        denom = torch.sqrt(torch.sum(torch.square(Z), dim=1, keepdim=True) + 1e-7)
        return Z / denom

    def _clip_pair_loss(self, z_m, z_n, scale):
        logits_mn = scale * (z_m @ z_n.T)
        loss_mn = torch.logsumexp(logits_mn, dim=1) - torch.diagonal(logits_mn)
        logits_nm = scale * (z_n @ z_m.T)
        loss_nm = torch.logsumexp(logits_nm, dim=1) - torch.diagonal(logits_nm)
        return 0.5 * (loss_mn.mean() + loss_nm.mean())

    def _compute_losses(self, inputs, training):
        Z = self.forward(inputs, training=training)
        scale = torch.exp(self.logit_scale.to(dtype=Z.dtype, device=Z.device))
        total = torch.zeros((), dtype=Z.dtype, device=Z.device)
        count = 0
        for m in range(len(self.model_list)):
            for n in range(len(self.model_list)):
                if m == n:
                    continue
                total = total + self._clip_pair_loss(Z[:, :, m], Z[:, :, n], scale)
                count += 1
        loss = total / max(count, 1) + self._regularization_loss(total)
        return {"clip_loss": loss}


class VICReg(_MultiViewModel):
    def __init__(
        self,
        model_list,
        regularizer_list,
        ndims,
        var_weight: float = 25.0,
        inv_weight: float = 25.0,
        cov_weight: float = 1.0,
        gamma: float = 1.0,
        run_from_config: bool = False,
        is_siamese: bool = False,
        eps: float = 1e-4,
        device=None,
        **kwargs,
    ):
        super().__init__(model_list, regularizer_list, ndims, run_from_config, is_siamese, device=device)
        self.var_weight = float(var_weight)
        self.inv_weight = float(inv_weight)
        self.cov_weight = float(cov_weight)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def _variance_loss(self, z):
        zc = z - torch.mean(z, dim=0)
        std = torch.sqrt(torch.mean(torch.square(zc), dim=0) + self.eps)
        return torch.mean(torch.relu(self.gamma - std))

    def _covariance_loss(self, z):
        zc = z - torch.mean(z, dim=0)
        denom = max(z.shape[0] - 1, 1)
        cov = zc.T @ zc / float(denom)
        eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        off = cov * (1.0 - eye)
        num = max(cov.shape[0] * (cov.shape[0] - 1), 1)
        return torch.sum(torch.square(off)) / float(num)

    def _compute_losses(self, inputs, training):
        Z = self.forward(inputs, training=training)
        inv_total = torch.zeros((), dtype=Z.dtype, device=Z.device)
        corr_total = torch.zeros((), dtype=Z.dtype, device=Z.device)
        pair_count = 0
        for i in range(len(self.model_list)):
            for j in range(i + 1, len(self.model_list)):
                inv_total = inv_total + torch.mean(torch.square(Z[:, :, i] - Z[:, :, j]))
                corr_total = corr_total + self._corr_pair(Z[:, :, i], Z[:, :, j])
                pair_count += 1
        inv_loss = inv_total / max(pair_count, 1)
        cross_metric = corr_total / max(pair_count, 1)

        var_loss = torch.stack([self._variance_loss(Z[:, :, v]) for v in range(len(self.model_list))]).mean()
        cov_loss = torch.stack([self._covariance_loss(Z[:, :, v]) for v in range(len(self.model_list))]).mean()
        redundancy = self._redundancy_ops(Z)
        total_loss = (
            self.inv_weight * inv_loss
            + self.var_weight * var_loss
            + self.cov_weight * cov_loss
            + self._regularization_loss(inv_loss)
        )
        return {
            "total_loss": total_loss,
            "cross_metric": cross_metric,
            "mse_loss": inv_loss,
            "redundancy": redundancy,
        }


class DGCCA(_MultiViewModel):
    metric_names = ("total_loss", "cross_metric", "gcca_loss", "redundancy")

    def __init__(
        self,
        model_list,
        regularizer_list,
        ndims,
        gcca_reg: float = 1e-3,
        momentum: float = 0.0,
        eps: float = 1e-6,
        center_outputs: bool = True,
        run_from_config: bool = False,
        is_siamese: bool = False,
        device=None,
        **kwargs,
    ):
        super().__init__(model_list, regularizer_list, ndims, run_from_config, is_siamese, device=device)
        self.gcca_reg = float(gcca_reg)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.center_outputs = bool(center_outputs)

    def _center_view(self, z):
        if not self.center_outputs:
            return z
        return z - torch.mean(z, dim=0, keepdim=True)

    def _compute_losses(self, inputs, training):
        Z = self.forward(inputs, training=training)
        centered = [self._center_view(Z[:, :, v]) for v in range(len(self.model_list))]
        projection_sum = None

        for z in centered:
            y = z.T
            cov = y @ y.T
            cov = 0.5 * (cov + cov.T)
            cov = cov + self.gcca_reg * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
            cov_inv = torch.linalg.pinv(cov)
            proj = z @ cov_inv @ z.T
            proj = 0.5 * (proj + proj.T)
            projection_sum = proj if projection_sum is None else projection_sum + proj

        eigenvalues = torch.linalg.eigvalsh(projection_sum)
        top = torch.flip(eigenvalues, dims=(0,))[: self.ndims]
        rank_used = torch.as_tensor(top.shape[0], dtype=Z.dtype, device=Z.device)
        gcca_loss = float(len(self.model_list)) * rank_used - torch.sum(top)
        gcca_loss = gcca_loss.to(dtype=Z.dtype)

        cross_metric = self._pairwise_cross_metric(Z)
        redundancy = self._redundancy_ops(Z)
        total_loss = gcca_loss + self._regularization_loss(gcca_loss)
        return {
            "total_loss": total_loss,
            "cross_metric": cross_metric,
            "gcca_loss": gcca_loss,
            "redundancy": redundancy,
        }

    def predict_shared(self, inputs, batch_size=None, verbose=0):
        return self.predict(inputs, batch_size=batch_size or 256, verbose=verbose)


class LeJEPA(_MultiViewModel):
    metric_names = ("total_loss", "cross_metric", "pred_loss", "sigreg_loss", "redundancy")

    def __init__(
        self,
        model_list,
        regularizer_list,
        ndims,
        lambda_weight: float = 0.05,
        num_global_views=None,
        num_slices: int = 256,
        integration_min: float = -5.0,
        integration_max: float = 5.0,
        integration_points: int = 17,
        run_from_config: bool = False,
        is_siamese: bool = False,
        eps: float = 1e-6,
        device=None,
        **kwargs,
    ):
        super().__init__(model_list, regularizer_list, ndims, run_from_config, is_siamese, device=device)
        self.lambda_weight = float(lambda_weight)
        self.num_global_views = len(self.model_list) if num_global_views is None else int(num_global_views)
        self.num_slices = int(num_slices)
        self.integration_min = float(integration_min)
        self.integration_max = float(integration_max)
        self.integration_points = int(integration_points)
        self.eps = float(eps)
        if self.num_global_views < 1 or self.num_global_views > len(self.model_list):
            raise ValueError("num_global_views must lie between 1 and the number of views.")

    def _global_center(self, Z):
        return torch.mean(Z[:, :, : self.num_global_views], dim=2)

    def _prediction_loss_ops(self, Z):
        center = self._global_center(Z)
        total = torch.zeros((), dtype=Z.dtype, device=Z.device)
        for view_index in range(len(self.model_list)):
            total = total + torch.mean(torch.square(center - Z[:, :, view_index]))
        return total / float(len(self.model_list))

    def _sample_slices(self, ref_tensor):
        A = torch.randn(self.ndims, self.num_slices, dtype=ref_tensor.dtype, device=ref_tensor.device)
        norm = torch.sqrt(torch.sum(A * A, dim=0, keepdim=True)) + self.eps
        return A / norm

    def _sigreg_view(self, z, A, t):
        proj = z @ A
        proj_t = proj.unsqueeze(-1) * t.reshape(1, 1, self.integration_points)
        ecf_real = torch.mean(torch.cos(proj_t), dim=0)
        ecf_imag = torch.mean(torch.sin(proj_t), dim=0)
        target_cf = torch.exp(-0.5 * torch.square(t))
        diff_sq = torch.square(ecf_real - target_cf.reshape(1, self.integration_points)) + torch.square(ecf_imag)
        weighted = diff_sq * target_cf.reshape(1, self.integration_points)
        dx = t[1:] - t[:-1]
        y_mid = 0.5 * (weighted[:, :-1] + weighted[:, 1:])
        integral = torch.sum(y_mid * dx, dim=-1)
        return torch.mean(integral * float(z.shape[0]))

    def _sigreg_loss_ops(self, Z):
        A = self._sample_slices(Z)
        t = torch.linspace(
            self.integration_min,
            self.integration_max,
            self.integration_points,
            dtype=Z.dtype,
            device=Z.device,
        )
        total = torch.zeros((), dtype=Z.dtype, device=Z.device)
        for view_index in range(len(self.model_list)):
            total = total + self._sigreg_view(Z[:, :, view_index], A, t)
        return total / float(len(self.model_list))

    def _cross_metric_ops(self, Z):
        center = self._global_center(Z)
        total = torch.zeros((), dtype=Z.dtype, device=Z.device)
        for view_index in range(len(self.model_list)):
            total = total + self._corr_pair(center, Z[:, :, view_index])
        return total / float(len(self.model_list))

    def _compute_losses(self, inputs, training):
        Z = self.forward(inputs, training=training)
        pred_loss = self._prediction_loss_ops(Z)
        sigreg_loss = self._sigreg_loss_ops(Z)
        total_loss = (
            (1.0 - self.lambda_weight) * pred_loss
            + self.lambda_weight * sigreg_loss
            + self._regularization_loss(pred_loss)
        )
        return {
            "total_loss": total_loss,
            "cross_metric": self._cross_metric_ops(Z),
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "redundancy": self._redundancy_ops(Z),
        }
