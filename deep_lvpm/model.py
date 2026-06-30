#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native PyTorch implementation of the Deep LVPM structural model."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pydot
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from deep_lvpm.layers.FactorLayer import FactorLayer
from deep_lvpm.layers.ZCALayer import ZCALayer


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _as_tensor(value, device: torch.device | None = None) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    elif tensor.is_floating_point() and tensor.dtype != torch.float32:
        tensor = tensor.float()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


@contextmanager
def _temporary_mode(module: nn.Module, training: bool):
    old_mode = module.training
    module.train(training)
    try:
        yield
    finally:
        module.train(old_mode)


def _call_module(module: nn.Module, inputs):
    if isinstance(inputs, (list, tuple)):
        try:
            return module(inputs)
        except TypeError:
            return module(*inputs)
    return module(inputs)


class DLVPMViewModel(nn.Module):
    """One user encoder plus the final DLVPM projection layer."""

    def __init__(self, encoder: nn.Module, dlv_layer: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.dlv_layer = dlv_layer
        self.n_inputs = int(getattr(encoder, "n_inputs", 1))

    def forward(self, inputs):
        x = _call_module(self.encoder, inputs)
        if x.ndim > 2:
            x = torch.flatten(x, start_dim=1)
        return self.dlv_layer(x)

    def regularization_loss(self, reference: torch.Tensor | None = None) -> torch.Tensor:
        total = self.dlv_layer.regularization_loss(reference=reference)
        if hasattr(self.encoder, "regularization_loss"):
            extra = self.encoder.regularization_loss()
            total = total + extra.to(dtype=total.dtype, device=total.device)
        return total

    def weight_normalizer(self, inputs):
        return self.dlv_layer.weight_normalizer(inputs)

    def apply_constraints(self):
        if hasattr(self.dlv_layer, "apply_constraints"):
            self.dlv_layer.apply_constraints()

    def last_layer(self):
        return self.dlv_layer


class StructuralModel(nn.Module):
    """
    Coordinate multiple PyTorch measurement models using the DLVPM objective.

    The public API intentionally mirrors the previous high-level version. ``compile``,
    ``fit``, ``evaluate`` and ``predict`` are Deep LVPM convenience methods
    implemented with standard PyTorch training commands.
    """

    def __init__(
        self,
        Path,
        model_list,
        regularizer_list,
        tot_num,
        ndims,
        orthogonalization="Moore-Penrose",
        momentum=0.95,
        epsilon=1e-4,
        train_DLV=True,
        run_from_config=False,
        is_siamese=False,
        diag_offset=1e-3,
        sparse_l1_list=0.0,
        order=False,
        order_association_cutoff=None,
        missing_strategy="impute",
        latent_imputation_rank=2,
        latent_imputation_iterations=3,
        device=None,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        self._path_array = np.asarray(Path, dtype=np.float32)
        self.register_buffer("Path", torch.as_tensor(self._path_array, dtype=torch.float32))

        self.tot_num = int(tot_num)
        self.ndims = int(ndims)
        self.initial_ndims = int(ndims)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.orthogonalization = orthogonalization
        self.regularizer_list = list(regularizer_list)
        self.train_DLV = bool(train_DLV)
        self.is_siamese = bool(is_siamese)
        self.diag_offset = float(diag_offset)
        self.sparse_l1_list = self._normalize_sparse_l1_list(sparse_l1_list, len(model_list))
        self.order = bool(order)
        self.order_association_cutoff = (
            None if order_association_cutoff is None else float(order_association_cutoff)
        )
        self.missing_strategy = str(missing_strategy)
        if self.missing_strategy not in {"impute", "project"}:
            raise ValueError('missing_strategy must be "impute" or "project".')
        self.latent_imputation_rank = int(latent_imputation_rank)
        self.latent_imputation_iterations = int(latent_imputation_iterations)
        if self.latent_imputation_rank < 1:
            raise ValueError("latent_imputation_rank must be at least 1.")
        if self.latent_imputation_iterations < 0:
            raise ValueError("latent_imputation_iterations must be non-negative.")
        self.retained_order_dims = int(ndims)
        self.device = torch.device(device) if device is not None else _default_device()
        self.optimizers: list[torch.optim.Optimizer] | None = None

        if self.order and self.orthogonalization != "zca":
            raise ValueError("'order' is only available when orthogonalization='zca'.")
        if self.order_association_cutoff is not None:
            if not (0.0 < self.order_association_cutoff <= 1.0):
                raise ValueError("order_association_cutoff must lie in the interval (0, 1].")
            if not self.order:
                raise ValueError("order_association_cutoff requires order=True.")

        if run_from_config:
            wrapped_models = list(model_list)
        elif self.is_siamese:
            wrapped = self.add_DLVPM_layer(model_list[0], regularizer_list[0], self.sparse_l1_list[0])
            wrapped_models = [wrapped for _ in model_list]
        else:
            wrapped_models = [
                self.add_DLVPM_layer(model, regularizer, sparse_l1)
                for model, regularizer, sparse_l1 in zip(model_list, regularizer_list, self.sparse_l1_list)
            ]
        self.model_list = nn.ModuleList(wrapped_models)

        self.register_buffer("order_moving_omega", torch.zeros(self.initial_ndims, self.initial_ndims))

    def add_DLVPM_layer(self, model, regularizer, sparse_l1=0.0):
        if not isinstance(model, nn.Module):
            raise ValueError("The input model must be a torch.nn.Module instance.")

        if self.orthogonalization == "Moore-Penrose":
            dlv_layer = FactorLayer(
                kernel_regularizer=regularizer,
                tot_num=self.tot_num,
                ndims=self.ndims,
                momentum=self.momentum,
                epsilon=self.epsilon,
                sparse_l1=sparse_l1,
            )
        elif self.orthogonalization == "zca":
            dlv_layer = ZCALayer(
                kernel_regularizer=regularizer,
                tot_num=self.tot_num,
                ndims=self.ndims,
                momentum=self.momentum,
                epsilon=self.epsilon,
                diag_offset=self.diag_offset,
                sparse_l1=sparse_l1,
            )
        else:
            raise ValueError('orthogonalization must be "Moore-Penrose" or "zca"')

        return DLVPMViewModel(model, dlv_layer)

    def _normalize_sparse_l1_list(self, sparse_l1_list, n_views):
        if sparse_l1_list is None:
            values = [0.0] * n_views
        elif isinstance(sparse_l1_list, (list, tuple, np.ndarray)):
            values = [float(x) for x in list(sparse_l1_list)]
            if len(values) != n_views:
                raise ValueError(f"sparse_l1_list must have length {n_views}, got {len(values)}")
        else:
            values = [float(sparse_l1_list)] * n_views

        if self.is_siamese and any(abs(x - values[0]) > 1e-12 for x in values):
            raise ValueError("In siamese mode, all entries of sparse_l1_list must be identical.")
        return values

    def get_config(self):
        return {
            "Path": self._path_array.copy(),
            "tot_num": self.tot_num,
            "ndims": self.ndims,
            "orthogonalization": self.orthogonalization,
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "train_DLV": self.train_DLV,
            "is_siamese": self.is_siamese,
            "diag_offset": self.diag_offset,
            "sparse_l1_list": list(self.sparse_l1_list),
            "order": self.order,
            "order_association_cutoff": self.order_association_cutoff,
            "missing_strategy": self.missing_strategy,
            "latent_imputation_rank": self.latent_imputation_rank,
            "latent_imputation_iterations": self.latent_imputation_iterations,
        }

    def compile(self, optimizer):
        if isinstance(optimizer, (list, tuple)):
            if len(optimizer) != len(self.model_list):
                raise ValueError("When optimizer is a list, it must have one optimizer per model.")
            self.optimizers = list(optimizer)
        elif isinstance(optimizer, torch.optim.Optimizer):
            self.optimizers = [optimizer for _ in self.model_list]
        else:
            raise ValueError("optimizer must be a torch optimizer or a list of torch optimizers.")

    def _optimizer_param_ids(self, optimizer):
        ids = set()
        for group in optimizer.param_groups:
            for param in group["params"]:
                ids.add(id(param))
        return ids

    def _ensure_optimizer_has_parameters(self, view_index):
        if self.optimizers is None:
            return

        optimizer = self.optimizers[view_index]
        existing = self._optimizer_param_ids(optimizer)
        missing = [
            param for param in self.model_list[view_index].parameters()
            if param.requires_grad and id(param) not in existing
        ]
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

    def _split_validation(self, tensors, validation_split):
        if not validation_split:
            return tensors, None

        n_samples = tensors[0].shape[0]
        val_count = int(round(n_samples * float(validation_split)))
        if val_count <= 0 or val_count >= n_samples:
            raise ValueError("validation_split must leave at least one train and one validation row.")

        train_tensors = [tensor[:-val_count] for tensor in tensors]
        val_tensors = [tensor[-val_count:] for tensor in tensors]
        return train_tensors, val_tensors

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

    def _reference_input(self, inputs_v):
        if isinstance(inputs_v, (list, tuple)):
            return inputs_v[0]
        return inputs_v

    def _bool_fill(self, reference_tensor, value):
        return torch.full(
            (reference_tensor.shape[0],),
            bool(value),
            dtype=torch.bool,
            device=reference_tensor.device,
        )

    def _zero_scalar(self, reference_tensor=None):
        if reference_tensor is None:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        return torch.zeros((), dtype=reference_tensor.dtype, device=reference_tensor.device)

    def _zero_latents(self, batch_size, reference_tensor):
        return torch.zeros(
            (int(batch_size), self.ndims),
            dtype=torch.float32 if not reference_tensor.is_floating_point() else reference_tensor.dtype,
            device=reference_tensor.device,
        )

    def _row_indices(self, row_mask):
        return torch.nonzero(row_mask, as_tuple=False).reshape(-1)

    def _gather_rows_by_index(self, inputs_v, row_indices):
        if isinstance(inputs_v, (list, tuple)):
            return [self._gather_rows_by_index(tensor, row_indices) for tensor in inputs_v]
        if row_indices.device != inputs_v.device:
            row_indices = row_indices.to(inputs_v.device)
        return inputs_v.index_select(0, row_indices)

    def _scatter_rows_by_index(self, values, row_indices, batch_size, reference_tensor):
        target_device = reference_tensor.device
        if values.device != target_device:
            values = values.to(target_device)
        if row_indices.device != target_device:
            row_indices = row_indices.to(target_device)

        zeros = torch.zeros((int(batch_size), self.ndims), dtype=values.dtype, device=target_device)
        if row_indices.numel() > 0:
            zeros[row_indices] = values
        return zeros

    def _tensor_row_missing_mask(self, tensor, view_index):
        if not tensor.is_floating_point():
            return self._bool_fill(tensor, False)

        nan_mask = torch.isnan(tensor)
        flat_nan_mask = nan_mask.reshape(nan_mask.shape[0], -1)
        row_any_nan = torch.any(flat_nan_mask, dim=1)
        row_all_nan = torch.all(flat_nan_mask, dim=1)
        if bool(torch.any(torch.logical_xor(row_any_nan, row_all_nan)).item()):
            raise ValueError(
                f"View {view_index} contains partially missing rows. "
                "Only all-NaN rows are supported for missing-view handling."
            )
        return row_all_nan

    def _view_row_present_mask(self, inputs_v, view_index):
        tensors = inputs_v if isinstance(inputs_v, (list, tuple)) else [inputs_v]
        reference_tensor = self._reference_input(inputs_v)
        missing_mask = None

        for tensor in tensors:
            if not tensor.is_floating_point():
                continue
            current_missing = self._tensor_row_missing_mask(tensor, view_index)
            if missing_mask is None:
                missing_mask = current_missing
            elif not torch.equal(missing_mask, current_missing):
                raise ValueError(
                    f"View {view_index} has inconsistent missing-row masks across inputs. "
                    "All tensors for a multi-input view must mark the same rows as missing."
                )

        if missing_mask is None:
            return self._bool_fill(reference_tensor, True)
        return torch.logical_not(missing_mask)

    def _encode_view_on_present_rows(self, view_index, inputs_v, row_mask, training):
        reference_tensor = self._reference_input(inputs_v)
        batch_size = reference_tensor.shape[0]
        row_indices = self._row_indices(row_mask)
        if row_indices.numel() == 0:
            return self._zero_latents(batch_size, reference_tensor)

        observed_inputs = self._gather_rows_by_index(inputs_v, row_indices)
        use_training = bool(training) and row_indices.numel() > 1
        with _temporary_mode(self.model_list[view_index], use_training):
            y_obs = self.model_list[view_index](observed_inputs)
        return self._scatter_rows_by_index(y_obs, row_indices, batch_size, reference_tensor)

    def _forward_views_with_missing(self, inputs_nested, training=False):
        y_list = []
        view_present_list = []

        for view_index in range(len(self.model_list)):
            row_present = self._view_row_present_mask(inputs_nested[view_index], view_index)
            view_present_list.append(row_present)
            y_list.append(
                self._encode_view_on_present_rows(
                    view_index,
                    inputs_nested[view_index],
                    row_present,
                    training=training,
                )
            )

        y = torch.stack(y_list, dim=2)
        view_present = torch.stack(view_present_list, dim=1)
        if bool(torch.any(~torch.any(view_present, dim=1)).item()):
            raise ValueError(
                "Every row must have at least one observed view. "
                "Rows with all views missing cannot be handled."
            )
        return y, view_present

    def _use_latent_imputation(self, training):
        return bool(training) and self.missing_strategy == "impute"

    def _low_rank_impute_latents(self, y, input_present):
        input_present = input_present.to(dtype=torch.bool, device=y.device)
        row_has_observed = torch.any(input_present, dim=1)
        view_has_observed = torch.any(input_present, dim=0)
        target_present = input_present | (row_has_observed.unsqueeze(1) & view_has_observed.unsqueeze(0))

        with torch.no_grad():
            completed = y.detach().clone()
            if not bool(torch.any(target_present & ~input_present).item()):
                return completed.detach(), target_present

            observed_float = input_present.to(dtype=completed.dtype)
            column_counts = observed_float.sum(dim=0).clamp(min=1.0)
            rank_limit = min(completed.shape[0], completed.shape[2], self.latent_imputation_rank)

            for dim_index in range(completed.shape[1]):
                original = y[:, dim_index, :].detach()
                column_means = (original * observed_float).sum(dim=0) / column_counts
                work = torch.where(
                    input_present,
                    original,
                    column_means.unsqueeze(0).expand_as(original),
                )
                work = torch.where(target_present, work, torch.zeros_like(work))

                for _ in range(self.latent_imputation_iterations):
                    if rank_limit < 1:
                        break
                    try:
                        u, s, vh = torch.linalg.svd(work, full_matrices=False)
                        reconstructed = (u[:, :rank_limit] * s[:rank_limit]) @ vh[:rank_limit, :]
                    except RuntimeError:
                        reconstructed = work
                        break
                    work = torch.where(
                        input_present,
                        original,
                        torch.where(target_present, reconstructed, torch.zeros_like(work)),
                    )

                completed[:, dim_index, :] = torch.where(
                    input_present,
                    original,
                    torch.where(target_present, work, torch.zeros_like(work)),
                )

        return completed.detach(), target_present

    def _training_latents_and_masks(self, inputs_nested, encoder_training):
        y_projected, input_present = self._forward_views_with_missing(
            inputs_nested,
            training=encoder_training,
        )
        if self._use_latent_imputation(training=True):
            y_target, target_present = self._low_rank_impute_latents(y_projected, input_present)
        else:
            y_target = y_projected
            target_present = input_present
        return y_target, input_present, target_present

    def forward(self, inputs, training=False):
        inputs_nested = self.organize_inputs_by_model(inputs)
        out, _ = self._forward_views_with_missing(inputs_nested, training=training)
        return out

    def _scale_factor_for_rows(self, row_mask, dtype, device):
        row_count = row_mask.to(dtype=dtype).sum()
        row_count_safe = torch.clamp(row_count, min=1.0)
        return torch.as_tensor(float(self.tot_num), dtype=dtype, device=device) / row_count_safe

    def _normalize_view_latents(self, view_index, y_view, row_mask, scale_fact):
        batch_size = y_view.shape[0]
        row_indices = self._row_indices(row_mask)
        if row_indices.numel() == 0:
            return self._zero_latents(batch_size, y_view)

        y_obs = self._gather_rows_by_index(y_view, row_indices)
        y_obs = self.model_list[view_index].weight_normalizer([y_obs, scale_fact, self.train_DLV])
        return self._scatter_rows_by_index(y_obs, row_indices, batch_size, y_view)

    def _weight_normaliser(self, y, view_present):
        y_list = []
        scale_fact_list = []
        for view_index in range(len(self.model_list)):
            y_view = y[:, :, view_index]
            scale_fact = self._scale_factor_for_rows(
                view_present[:, view_index],
                y_view.dtype,
                y_view.device,
            )
            scale_fact_list.append(scale_fact)
            y_list.append(
                self._normalize_view_latents(
                    view_index,
                    y_view,
                    view_present[:, view_index],
                    scale_fact,
                )
            )

        return torch.stack(y_list, dim=2), torch.stack(scale_fact_list)

    def _normalize_pred(self, y_pred, scale_fact):
        denom = torch.sqrt(scale_fact) * torch.sqrt(torch.sum(torch.square(y_pred), dim=0) + self.epsilon)
        return y_pred / denom

    def _batch_structural_matrix(self, y, view_present):
        dtype = y.dtype
        omega = torch.zeros((self.ndims, self.ndims), dtype=dtype, device=y.device)

        for left_view in range(len(self.model_list)):
            y_left = y[:, :, left_view]
            left_mask = view_present[:, left_view].to(dtype=dtype)

            for right_view in range(len(self.model_list)):
                edge_weight = float(self._path_array[left_view, right_view])
                if left_view == right_view or edge_weight == 0.0:
                    continue

                pair_mask = left_mask * view_present[:, right_view].to(dtype=dtype)
                pair_count = pair_mask.sum()
                pair_count_safe = torch.clamp(pair_count, min=1.0)
                pair_mask_exp = pair_mask.unsqueeze(1)

                y_left_masked = y_left * pair_mask_exp
                y_right_masked = y[:, :, right_view] * pair_mask_exp
                sum_left = y_left_masked.sum(dim=0)
                sum_right = y_right_masked.sum(dim=0)
                cross = y_left_masked.T @ y_right_masked
                mean_outer = sum_left.unsqueeze(1) @ sum_right.unsqueeze(0) / pair_count_safe
                pair_cov = (cross - mean_outer) / pair_count_safe
                valid = (pair_count > 1.0).to(dtype=dtype)
                omega = omega + edge_weight * pair_cov * valid

        return 0.5 * (omega + omega.T)

    def _order_strength_metric(self, omega):
        if self.ndims < 2:
            return torch.ones((), dtype=omega.dtype, device=omega.device)
        strengths = torch.diagonal(0.5 * (omega + omega.T))
        correct = 0.0
        total = 0.0
        for i in range(self.ndims):
            for j in range(i + 1, self.ndims):
                correct += float(strengths[i] > strengths[j])
                total += 1.0
        return torch.as_tensor(correct / max(total, 1.0), dtype=omega.dtype, device=omega.device)

    def _update_order_moving_omega(self, omega_batch):
        if not self.order:
            return
        current = self.order_moving_omega[: self.ndims, : self.ndims].to(omega_batch.device, omega_batch.dtype)
        state_mass = torch.sum(torch.abs(current))
        if float(state_mass.detach().cpu()) <= self.epsilon:
            updated = omega_batch.detach()
        else:
            updated = self.momentum * current + (1.0 - self.momentum) * omega_batch.detach()
        self.order_moving_omega[: self.ndims, : self.ndims].copy_(updated.to(self.order_moving_omega.device))

    def _rotation_from_order_moving_omega(self, dtype=None, device=None):
        dtype = dtype or self.order_moving_omega.dtype
        device = device or self.order_moving_omega.device
        omega = self.order_moving_omega[: self.ndims, : self.ndims].to(dtype=dtype, device=device)
        omega = 0.5 * (omega + omega.T)
        omega = omega + self.epsilon * torch.eye(self.ndims, dtype=dtype, device=device)
        _, eigvecs = torch.linalg.eigh(omega)
        return torch.flip(eigvecs, dims=(1,))

    def _apply_structural_rotation(self, rotation):
        seen = set()
        for view_model in self.model_list:
            if id(view_model) in seen:
                continue
            seen.add(id(view_model))
            last_layer = view_model.last_layer()
            if not isinstance(last_layer, ZCALayer) or not last_layer._built:
                continue
            rotation = rotation.to(dtype=last_layer.project.dtype, device=last_layer.project.device)
            with torch.no_grad():
                last_layer.project.copy_(last_layer.project @ rotation)
                rotated_cov = rotation.T @ last_layer.moving_conv2 @ rotation
                last_layer.moving_conv2.copy_(0.5 * (rotated_cov + rotated_cov.T))

    def _retained_dims_from_order_omega(self):
        omega = self.order_moving_omega[: self.ndims, : self.ndims].detach().cpu().numpy()
        omega = 0.5 * (omega + omega.T)
        eigvals = np.linalg.eigvalsh(omega)[::-1]
        strengths = np.maximum(eigvals, 0.0)
        total_strength = float(np.sum(strengths))
        if total_strength <= self.epsilon:
            return int(self.ndims)
        cumulative = np.cumsum(strengths) / total_strength
        retained = int(np.searchsorted(cumulative, self.order_association_cutoff) + 1)
        return max(1, min(int(self.ndims), retained))

    def _resize_ordered_zca_dimensions(self, retained_dims):
        retained_dims = int(retained_dims)
        if retained_dims >= self.ndims:
            return

        seen = set()
        for view_model in self.model_list:
            if id(view_model) in seen:
                continue
            seen.add(id(view_model))
            last_layer = view_model.last_layer()
            if not isinstance(last_layer, ZCALayer) or not last_layer._built:
                continue
            with torch.no_grad():
                last_layer.project = nn.Parameter(last_layer.project[:, :retained_dims].detach().clone())
                last_layer.ndims = retained_dims
                last_layer.moving_conv2 = last_layer.moving_conv2[:retained_dims, :retained_dims].detach().clone()

        self.ndims = retained_dims
        self.retained_order_dims = retained_dims

    def _finalize_ordered_dimensions(self):
        if not self.order:
            return False
        omega_mass = float(torch.sum(torch.abs(self.order_moving_omega[: self.ndims, : self.ndims])).detach().cpu())
        if omega_mass <= self.epsilon:
            return False
        rotation = self._rotation_from_order_moving_omega(device=self.device)
        self._apply_structural_rotation(rotation)
        if self.order_association_cutoff is not None:
            retained_dims = self._retained_dims_from_order_omega()
            self._resize_ordered_zca_dimensions(retained_dims)
            print(
                f"Retained {retained_dims} ordered dimensions "
                f"using omega association mass cutoff {self.order_association_cutoff:.2f}."
            )
        return True

    def _step(self, view_index, inputs_v, y_target, target_present, source_present, scale_fact):
        optimizer = self.optimizers[view_index]
        row_indices = self._row_indices(source_present)
        reference_tensor = self._reference_input(inputs_v)
        zero = self._zero_scalar(y_target)

        if row_indices.numel() <= 1:
            return zero, zero

        observed_inputs = self._gather_rows_by_index(inputs_v, row_indices)
        optimizer.zero_grad(set_to_none=True)

        with _temporary_mode(self.model_list[view_index], True):
            y_pred_obs = self.model_list[view_index](observed_inputs)
        y_pred_obs = self._normalize_pred(y_pred_obs, scale_fact)
        y_pred = self._scatter_rows_by_index(
            y_pred_obs,
            row_indices,
            y_target.shape[0],
            reference_tensor,
        )

        mse_loss = self.mse_loss(y_target, y_pred, view_index, target_present, source_present)
        reg_loss = self.model_list[view_index].regularization_loss(reference=mse_loss)
        loss = mse_loss + reg_loss
        loss.backward()
        optimizer.step()
        self.model_list[view_index].apply_constraints()
        return loss.detach(), mse_loss.detach()

    def build(self, X_list):
        tensors = self._prepare_tensors(X_list, device=self.device)
        inputs_nested = self.organize_inputs_by_model(tensors)
        self.train()
        with torch.no_grad():
            self._forward_views_with_missing(inputs_nested, training=False)
        return self

    def train_step(self, data):
        if self.optimizers is None:
            raise RuntimeError("Call compile(optimizer) before train_step().")

        inputs = self._batch_to_inputs(data, device=self.device)
        self.to(self.device)
        self.build(inputs)
        for view_index in range(len(self.model_list)):
            self._ensure_optimizer_has_parameters(view_index)

        self.train()
        inputs_nested = self.organize_inputs_by_model(inputs)
        with torch.no_grad():
            y_raw, input_present, target_present = self._training_latents_and_masks(
                inputs_nested,
                encoder_training=self.train_DLV,
            )
            y_ortho, scale_fact = self._weight_normaliser(y_raw, target_present)
            if self.order:
                omega_batch = self._batch_structural_matrix(y_raw, target_present)
                self._update_order_moving_omega(omega_batch)

        total_losses = []
        total_mse = []
        total_corr = []
        total_redundancy = []
        for view_index in range(len(self.model_list)):
            loss, mse_loss = self._step(
                view_index,
                inputs_nested[view_index],
                y_ortho.detach(),
                target_present,
                input_present[:, view_index],
                scale_fact[view_index],
            )
            total_losses.append(loss)
            total_mse.append(mse_loss)
            total_corr.append(
                self.corr_metric(
                    y_raw.detach(),
                    y_raw[:, :, view_index].detach(),
                    view_index,
                    target_present,
                    input_present[:, view_index],
                )
            )
            total_redundancy.append(
                self.calculate_redundancy(y_raw[:, :, view_index].detach(), row_mask=target_present[:, view_index])
            )

        return {
            "total_loss": torch.stack(total_losses).mean(),
            "cross_metric": torch.stack(total_corr).mean(),
            "mse_loss": torch.stack(total_mse).mean(),
            "redundancy": torch.stack(total_redundancy).mean(),
        }

    def test_step(self, data):
        inputs = self._batch_to_inputs(data, device=self.device)
        self.to(self.device)
        self.eval()
        inputs_nested = self.organize_inputs_by_model(inputs)
        with torch.no_grad():
            y_raw, view_present = self._forward_views_with_missing(inputs_nested, training=False)
            y_ortho, _ = self._weight_normaliser(y_raw, view_present)
            if self.order:
                omega_batch = self._batch_structural_matrix(y_raw, view_present)
                order_strength = self._order_strength_metric(omega_batch)

            total_losses = []
            total_mse = []
            total_corr = []
            total_redundancy = []
            for view_index in range(len(self.model_list)):
                source_mask = view_present[:, view_index]
                row_indices = self._row_indices(source_mask)
                zero = self._zero_scalar(y_ortho)

                if row_indices.numel() > 0:
                    observed_inputs = self._gather_rows_by_index(inputs_nested[view_index], row_indices)
                    with _temporary_mode(self.model_list[view_index], False):
                        y_pred_obs = self.model_list[view_index](observed_inputs)
                    scale_fact = self._scale_factor_for_rows(source_mask, y_pred_obs.dtype, y_pred_obs.device)
                    y_pred_obs = self._normalize_pred(y_pred_obs, scale_fact)
                    y_pred = self._scatter_rows_by_index(
                        y_pred_obs,
                        row_indices,
                        y_ortho.shape[0],
                        self._reference_input(inputs_nested[view_index]),
                    )
                    mse_loss = self.mse_loss(y_ortho, y_pred, view_index, view_present)
                    reg_loss = self.model_list[view_index].regularization_loss(reference=mse_loss)
                    loss = mse_loss + reg_loss
                    corr = self.corr_metric(y_raw, y_pred, view_index, view_present)
                else:
                    loss = zero
                    mse_loss = zero
                    corr = zero

                total_losses.append(loss)
                total_mse.append(mse_loss)
                total_corr.append(corr)
                total_redundancy.append(
                    self.calculate_redundancy(y_raw[:, :, view_index], row_mask=view_present[:, view_index])
                )

        metrics = {
            "total_loss": torch.stack(total_losses).mean(),
            "cross_metric": torch.stack(total_corr).mean(),
            "mse_loss": torch.stack(total_mse).mean(),
            "redundancy": torch.stack(total_redundancy).mean(),
        }
        if self.order:
            metrics["order_strength"] = order_strength
        return metrics

    def fit(
        self,
        X_train,
        batch_size=32,
        epochs=10,
        verbose=True,
        validation_data=None,
        validation_split=0.0,
        shuffle=True,
        schedulers=None,
        **kwargs,
    ):
        del kwargs
        if self.optimizers is None:
            raise RuntimeError("Call compile(optimizer) before fit().")

        if isinstance(X_train, DataLoader):
            if validation_split:
                raise ValueError("validation_split is only available for array or tensor inputs.")
            loader = X_train
        else:
            tensors = self._prepare_tensors(X_train)
            tensors, split_validation = self._split_validation(tensors, validation_split)
            if validation_data is None and split_validation is not None:
                validation_data = split_validation
            drop_last = tensors[0].shape[0] > batch_size
            loader = self._make_loader(
                tensors,
                batch_size=batch_size,
                shuffle=bool(shuffle),
                drop_last=drop_last,
            )
        if len(loader) == 0:
            raise ValueError("Training data produced no batches.")

        self.to(self.device)
        first_batch = self._batch_to_inputs(next(iter(loader)), device=self.device)
        self.build(first_batch)
        for view_index in range(len(self.model_list)):
            self._ensure_optimizer_has_parameters(view_index)

        history = {
            "total_loss": [],
            "cross_metric": [],
            "mse_loss": [],
            "redundancy": [],
        }

        for epoch in range(int(epochs)):
            self.train()
            sums = {key: 0.0 for key in history}
            batch_count = 0

            for batch_tensors in loader:
                inputs = self._batch_to_inputs(batch_tensors, device=self.device)
                inputs_nested = self.organize_inputs_by_model(inputs)

                with torch.no_grad():
                    y_raw, input_present, target_present = self._training_latents_and_masks(
                        inputs_nested,
                        encoder_training=self.train_DLV,
                    )
                    y_ortho, scale_fact = self._weight_normaliser(y_raw, target_present)
                    if self.order:
                        omega_batch = self._batch_structural_matrix(y_raw, target_present)
                        self._update_order_moving_omega(omega_batch)

                total_losses = []
                total_mse = []
                total_corr = []
                total_redundancy = []

                for view_index in range(len(self.model_list)):
                    loss, mse_loss = self._step(
                        view_index,
                        inputs_nested[view_index],
                        y_ortho.detach(),
                        target_present,
                        input_present[:, view_index],
                        scale_fact[view_index],
                    )
                    total_losses.append(loss)
                    total_mse.append(mse_loss)
                    total_corr.append(
                        self.corr_metric(
                            y_raw.detach(),
                            y_raw[:, :, view_index].detach(),
                            view_index,
                            target_present,
                            input_present[:, view_index],
                        )
                    )
                    total_redundancy.append(
                        self.calculate_redundancy(
                            y_raw[:, :, view_index].detach(),
                            row_mask=target_present[:, view_index],
                        )
                    )

                sums["total_loss"] += float(torch.stack(total_losses).mean().detach().cpu())
                sums["cross_metric"] += float(torch.stack(total_corr).mean().detach().cpu())
                sums["mse_loss"] += float(torch.stack(total_mse).mean().detach().cpu())
                sums["redundancy"] += float(torch.stack(total_redundancy).mean().detach().cpu())
                batch_count += 1

            for key in history:
                history[key].append(sums[key] / max(batch_count, 1))

            if validation_data is not None:
                val_metrics = self.evaluate(validation_data, batch_size=batch_size, verbose=False)
                for key, value in val_metrics.items():
                    history.setdefault(f"val_{key}", []).append(value)

            if verbose:
                message = (
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"total_loss: {history['total_loss'][-1]:.5f} - "
                    f"cross_metric: {history['cross_metric'][-1]:.5f} - "
                    f"mse_loss: {history['mse_loss'][-1]:.5f} - "
                    f"redundancy: {history['redundancy'][-1]:.5f}"
                )
                if validation_data is not None:
                    message += f" - val_cross_metric: {history['val_cross_metric'][-1]:.5f}"
                print(message)

            if schedulers is not None:
                for scheduler in schedulers:
                    scheduler.step()

        self._finalize_ordered_dimensions()
        return history

    def evaluate(self, data, batch_size=256, verbose=True, **kwargs):
        del kwargs
        loader = self._make_loader(data, batch_size=batch_size, shuffle=False, drop_last=False)

        self.to(self.device)
        self.eval()
        sums = {
            "total_loss": 0.0,
            "cross_metric": 0.0,
            "mse_loss": 0.0,
            "redundancy": 0.0,
        }
        if self.order:
            sums["order_strength"] = 0.0
        batch_count = 0

        with torch.no_grad():
            for batch_tensors in loader:
                inputs = self._batch_to_inputs(batch_tensors, device=self.device)
                inputs_nested = self.organize_inputs_by_model(inputs)
                y_raw, view_present = self._forward_views_with_missing(inputs_nested, training=False)
                y_ortho, _ = self._weight_normaliser(y_raw, view_present)
                if self.order:
                    omega_batch = self._batch_structural_matrix(y_raw, view_present)
                    order_strength = self._order_strength_metric(omega_batch)

                total_losses = []
                total_mse = []
                total_corr = []
                total_redundancy = []

                for view_index in range(len(self.model_list)):
                    source_mask = view_present[:, view_index]
                    row_indices = self._row_indices(source_mask)
                    zero = self._zero_scalar(y_ortho)

                    if row_indices.numel() > 0:
                        observed_inputs = self._gather_rows_by_index(inputs_nested[view_index], row_indices)
                        with _temporary_mode(self.model_list[view_index], False):
                            y_pred_obs = self.model_list[view_index](observed_inputs)
                        scale_fact = self._scale_factor_for_rows(source_mask, y_pred_obs.dtype, y_pred_obs.device)
                        y_pred_obs = self._normalize_pred(y_pred_obs, scale_fact)
                        y_pred = self._scatter_rows_by_index(
                            y_pred_obs,
                            row_indices,
                            y_ortho.shape[0],
                            self._reference_input(inputs_nested[view_index]),
                        )
                        mse_loss = self.mse_loss(y_ortho, y_pred, view_index, view_present)
                        reg_loss = self.model_list[view_index].regularization_loss(reference=mse_loss)
                        loss = mse_loss + reg_loss
                        corr = self.corr_metric(y_raw, y_pred, view_index, view_present)
                    else:
                        loss = zero
                        mse_loss = zero
                        corr = zero

                    total_losses.append(loss)
                    total_mse.append(mse_loss)
                    total_corr.append(corr)
                    total_redundancy.append(
                        self.calculate_redundancy(y_raw[:, :, view_index], row_mask=view_present[:, view_index])
                    )

                sums["total_loss"] += float(torch.stack(total_losses).mean().detach().cpu())
                sums["cross_metric"] += float(torch.stack(total_corr).mean().detach().cpu())
                sums["mse_loss"] += float(torch.stack(total_mse).mean().detach().cpu())
                sums["redundancy"] += float(torch.stack(total_redundancy).mean().detach().cpu())
                if self.order:
                    sums["order_strength"] += float(order_strength.detach().cpu())
                batch_count += 1

        metrics = {key: value / max(batch_count, 1) for key, value in sums.items()}
        if verbose:
            print(
                "Eval - "
                f"total_loss: {metrics['total_loss']:.5f} - "
                f"cross_metric: {metrics['cross_metric']:.5f} - "
                f"mse_loss: {metrics['mse_loss']:.5f} - "
                f"redundancy: {metrics['redundancy']:.5f}"
            )
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

    def mse_loss(self, y_true, y_pred, vie, view_present, source_present=None):
        dtype = y_true.dtype
        device = y_true.device
        if source_present is None:
            source_present = view_present[:, vie]
        source_mask = source_present.to(dtype=dtype)
        target_mask = view_present.to(dtype=dtype)
        pair_mask = source_mask.unsqueeze(1) * target_mask

        sq_error = torch.square(y_true - y_pred.unsqueeze(2))
        sq_error = sq_error * pair_mask.unsqueeze(1)

        counts = pair_mask.sum(dim=0)
        counts_safe = torch.clamp(counts, min=1.0)
        se_mean = sq_error.sum(dim=0) / counts_safe.unsqueeze(0)

        connected_mask = self.Path[vie, :].to(dtype=dtype, device=device)
        valid_connected = connected_mask * (counts > 0).to(dtype=dtype)
        se_mean_masked = se_mean * valid_connected.unsqueeze(0)
        return se_mean_masked.sum() / 2.0

    def corr_metric(self, y_true, y_pred, vie, view_present, source_present=None):
        dtype = y_true.dtype
        device = y_true.device
        eps = torch.as_tensor(self.epsilon, dtype=dtype, device=device)
        if source_present is None:
            source_present = view_present[:, vie]
        source_mask = source_present.to(dtype=dtype)
        total_corr = torch.zeros((), dtype=dtype, device=device)
        total_weight = torch.zeros((), dtype=dtype, device=device)

        for target_vie in range(len(self.model_list)):
            connected = self.Path[vie, target_vie].to(dtype=dtype, device=device)
            target_mask = view_present[:, target_vie].to(dtype=dtype)
            pair_mask = source_mask * target_mask
            pair_count = pair_mask.sum()
            pair_count_safe = torch.clamp(pair_count, min=1.0)
            pair_mask_exp = pair_mask.unsqueeze(1)

            y_true_target = y_true[:, :, target_vie]
            y_true_mean = torch.sum(y_true_target * pair_mask_exp, dim=0) / pair_count_safe
            y_pred_mean = torch.sum(y_pred * pair_mask_exp, dim=0) / pair_count_safe

            y_true_c = (y_true_target - y_true_mean) * pair_mask_exp
            y_pred_c = (y_pred - y_pred_mean) * pair_mask_exp

            denom_true = torch.sqrt(torch.sum(torch.square(y_true_c), dim=0) + eps)
            denom_pred = torch.sqrt(torch.sum(torch.square(y_pred_c), dim=0) + eps)
            corr_dim = torch.sum((y_true_c / denom_true) * (y_pred_c / denom_pred), dim=0)
            pair_corr = torch.sum(corr_dim) / float(self.ndims)

            valid_pair = connected * (pair_count > 1.0).to(dtype=dtype)
            total_corr = total_corr + pair_corr * valid_pair
            total_weight = total_weight + valid_pair

        return total_corr / torch.clamp(total_weight, min=1.0)

    def calculate_redundancy(self, Y, epsilon=1e-8, row_mask=None):
        if not torch.is_tensor(Y):
            Y = torch.as_tensor(Y, dtype=torch.float32, device=self.device)
        Y = Y.float()

        if row_mask is None:
            row_mask = torch.ones((Y.shape[0],), dtype=Y.dtype, device=Y.device)
        else:
            row_mask = row_mask.to(dtype=Y.dtype, device=Y.device)
        row_mask = row_mask.unsqueeze(1)

        n_f = row_mask.sum()
        n_safe = torch.clamp(n_f, min=1.0)
        col_mean = torch.sum(Y * row_mask, dim=0, keepdim=True) / n_safe
        Yc = (Y - col_mean) * row_mask

        denom_n = torch.clamp(n_f - 1.0, min=1.0)
        cov = Yc.T @ Yc / denom_n
        var = torch.sum(Yc * Yc, dim=0) / denom_n
        std = torch.sqrt(torch.clamp(var, min=epsilon))
        corr = cov / torch.clamp(std.reshape(-1, 1) * std.reshape(1, -1), min=epsilon)
        corr_abs = torch.abs(corr)
        mask = torch.ones_like(corr_abs) - torch.eye(corr_abs.shape[0], dtype=corr_abs.dtype, device=corr_abs.device)
        num_pairs = max(corr_abs.shape[0] * (corr_abs.shape[0] - 1), 1)
        return (corr_abs * mask).sum() / float(num_pairs) * (n_f > 1.0).to(corr_abs.dtype)

    def calculate_corrmat(self, DLVs):
        if not torch.is_tensor(DLVs):
            DLVs = torch.as_tensor(DLVs, dtype=torch.float32, device=self.device)
        if len(DLVs.shape) != 3:
            raise ValueError("Input must be a 3D tensor")

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

    def plot_structural_model(self, outputname):
        graph = pydot.Dot(graph_type="digraph")
        for idx in range(self.Path.shape[0]):
            graph.add_node(pydot.Node(str(idx)))
        path = self.Path.detach().cpu().numpy()
        for i in range(path.shape[0]):
            for j in range(path.shape[1]):
                if path[i, j] != 0:
                    graph.add_edge(pydot.Edge(str(i), str(j)))
        graph.write_png(outputname)

    def save(self, path):
        checkpoint = {
            "config": self.get_config(),
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)

    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)

    def load_state_dict_from_path(self, path, strict=True):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        return self.load_state_dict(state_dict, strict=strict)
