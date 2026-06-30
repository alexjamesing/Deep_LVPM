#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integrated gradients utilities for native PyTorch DLVPM models."""

from __future__ import annotations

import numbers

import numpy as np
import torch


def calculate_integrated_gradients(
    structural_model,
    data,
    baseline=0.0,
    dlv_index=0,
    steps=50,
    explain_loss_reduction=True,
    return_numpy=True,
):
    """
    Calculate integrated gradients for one DLV-specific structural loss.

    Parameters
    ----------
    structural_model:
        A trained ``deep_lvpm.model.StructuralModel`` instance.
    data:
        Flat list or tuple of input arrays, in the same order used by
        ``StructuralModel.fit`` and ``StructuralModel.predict``.
    baseline:
        Either a single numeric value, such as ``0.0``, or a flat list/tuple of
        baseline arrays. Per-input baseline values can have the same shape as
        the matching data array or any shape that broadcasts to it, such as a
        one-dimensional feature baseline vector.
    dlv_index:
        Zero-based DLV index to explain.
    steps:
        Number of right-endpoint integration steps.
    explain_loss_reduction:
        If ``True``, positive attribution means the feature helps reduce the
        selected DLV loss relative to the baseline.
    return_numpy:
        If ``True``, return NumPy arrays. Otherwise, return PyTorch tensors.

    Returns
    -------
    list or tuple
        Integrated gradients in the same flat structure as ``data``.
    """

    _validate_steps(steps)
    _validate_dlv_index(structural_model, dlv_index)

    flat_data, return_kind = _as_flat_input_list(data)
    view_input_groups = _view_input_groups(structural_model)
    expected_inputs = sum(len(group) for group in view_input_groups)
    if len(flat_data) != expected_inputs:
        raise ValueError(
            f"data must contain {expected_inputs} input tensor(s) for this StructuralModel, "
            f"got {len(flat_data)}."
        )

    device = _model_device(structural_model)
    data_tensors = [_to_torch_tensor(value, device=device) for value in flat_data]
    _validate_float_tensors(data_tensors)

    baseline_tensors = _prepare_baseline_tensors(baseline, data_tensors, device=device)
    _validate_matching_shapes(data_tensors, baseline_tensors, name="baseline")

    state_snapshot = None
    training_modes = _snapshot_training_modes(structural_model)
    try:
        _ensure_model_is_built(structural_model, data_tensors)
        state_snapshot = _snapshot_state_dict(structural_model)
        attributions = _calculate_integrated_gradients_torch(
            structural_model,
            data_tensors,
            baseline_tensors,
            view_input_groups,
            int(dlv_index),
            int(steps),
            bool(explain_loss_reduction),
        )
    finally:
        if state_snapshot is not None:
            _restore_state_dict(structural_model, state_snapshot)
        _restore_training_modes(training_modes)

    if return_numpy:
        attributions = [value.detach().cpu().numpy() for value in attributions]

    return _restore_return_structure(attributions, return_kind)


def _calculate_integrated_gradients_torch(
    structural_model,
    data_tensors,
    baseline_tensors,
    view_input_groups,
    dlv_index,
    steps,
    explain_loss_reduction,
):
    data_nested = _group_flat_values(data_tensors, view_input_groups)
    with torch.no_grad():
        target_dlvs, view_present = _fixed_target_dlvs(structural_model, data_nested)
    target_dlvs = target_dlvs.detach()
    view_present = view_present.detach()

    deltas = [
        torch.where(torch.isnan(data_tensor), torch.zeros_like(data_tensor), data_tensor - baseline_tensor)
        for data_tensor, baseline_tensor in zip(data_tensors, baseline_tensors)
    ]
    accumulated_gradients = [torch.zeros_like(data_tensor) for data_tensor in data_tensors]

    for view_index, flat_indices in enumerate(view_input_groups):
        row_mask = view_present[:, view_index]
        if not bool(torch.any(row_mask).item()):
            continue

        row_indices = structural_model._row_indices(row_mask)
        batch_size = target_dlvs.shape[0]

        for step_index in range(1, steps + 1):
            alpha = step_index / float(steps)
            interpolated_inputs = []
            for flat_index in flat_indices:
                interpolated = baseline_tensors[flat_index] + alpha * deltas[flat_index]
                interpolated = interpolated.detach().clone().requires_grad_(True)
                interpolated_inputs.append(interpolated)

            model_inputs = _view_model_inputs(interpolated_inputs)
            observed_inputs = structural_model._gather_rows_by_index(model_inputs, row_indices)

            with _temporary_eval_mode(structural_model.model_list[view_index]):
                y_pred_obs = structural_model.model_list[view_index](observed_inputs)

            scale_fact = structural_model._scale_factor_for_rows(
                row_mask,
                y_pred_obs.dtype,
                y_pred_obs.device,
            )
            y_pred_obs = structural_model._normalize_pred(y_pred_obs, scale_fact)
            y_pred = structural_model._scatter_rows_by_index(
                y_pred_obs,
                row_indices,
                batch_size,
                interpolated_inputs[0],
            )
            selected_loss = _selected_dlv_loss(
                structural_model,
                target_dlvs,
                y_pred,
                view_index,
                view_present,
                dlv_index,
            )
            target_scalar = -selected_loss if explain_loss_reduction else selected_loss

            if target_scalar.requires_grad:
                gradients = torch.autograd.grad(
                    target_scalar,
                    interpolated_inputs,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
            else:
                gradients = [None for _ in interpolated_inputs]

            for flat_index, gradient in zip(flat_indices, gradients):
                if gradient is None:
                    gradient = torch.zeros_like(data_tensors[flat_index])
                accumulated_gradients[flat_index] = accumulated_gradients[flat_index] + gradient.detach()

    return [
        deltas[index] * (accumulated_gradients[index] / float(steps))
        for index in range(len(data_tensors))
    ]


def _fixed_target_dlvs(structural_model, data_nested):
    y_raw, view_present = structural_model._forward_views_with_missing(data_nested, training=False)
    target_dlvs, _ = structural_model._weight_normaliser(y_raw, view_present)
    return target_dlvs, view_present


def _selected_dlv_loss(structural_model, target_dlvs, y_pred, view_index, view_present, dlv_index):
    dtype = target_dlvs.dtype
    device = target_dlvs.device

    source_mask = view_present[:, view_index].to(dtype=dtype, device=device)
    target_mask = view_present.to(dtype=dtype, device=device)
    pair_mask = source_mask.unsqueeze(1) * target_mask

    target_dlv = target_dlvs[:, dlv_index, :]
    predicted_dlv = y_pred[:, dlv_index].unsqueeze(1)
    squared_error = torch.square(target_dlv - predicted_dlv)
    squared_error = squared_error * pair_mask

    counts = pair_mask.sum(dim=0)
    counts_safe = torch.clamp(counts, min=1.0)
    mean_squared_error = squared_error.sum(dim=0) / counts_safe

    connected_mask = structural_model.Path[view_index, :].to(dtype=dtype, device=device)
    valid_connected = connected_mask * (counts > 0).to(dtype=dtype)
    return torch.sum(mean_squared_error * valid_connected) / torch.as_tensor(2.0, dtype=dtype, device=device)


def _as_flat_input_list(data):
    if isinstance(data, tuple):
        return list(data), "tuple"
    if isinstance(data, list):
        return list(data), "list"
    return [data], "single"


def _restore_return_structure(values, return_kind):
    if return_kind == "tuple":
        return tuple(values)
    if return_kind == "single":
        return values[0]
    return values


def _view_input_groups(structural_model):
    groups = []
    data_index = 0

    for model in structural_model.model_list:
        num_inputs = int(getattr(model, "n_inputs", 1))
        groups.append(list(range(data_index, data_index + num_inputs)))
        data_index += num_inputs

    return groups


def _group_flat_values(flat_values, view_input_groups):
    grouped = []
    for flat_indices in view_input_groups:
        values = [flat_values[index] for index in flat_indices]
        grouped.append(values[0] if len(values) == 1 else values)
    return grouped


def _view_model_inputs(values):
    return values[0] if len(values) == 1 else values


def _prepare_baseline_tensors(baseline, data_tensors, device):
    if _is_scalar_baseline(baseline):
        return [torch.full_like(data_tensor, float(baseline), device=device) for data_tensor in data_tensors]

    if not isinstance(baseline, (list, tuple)):
        if len(data_tensors) == 1:
            return [_prepare_single_baseline_tensor(baseline, data_tensors[0], device, 0)]
        raise TypeError(
            "baseline must be either a single numeric value or a flat list/tuple matching data."
        )

    if len(baseline) != len(data_tensors):
        raise ValueError(
            f"baseline must contain {len(data_tensors)} tensor(s), got {len(baseline)}."
        )

    return [
        _prepare_single_baseline_tensor(value, data_tensor, device, index)
        for index, (value, data_tensor) in enumerate(zip(baseline, data_tensors))
    ]


def _prepare_single_baseline_tensor(value, data_tensor, device, index):
    baseline_tensor = _cast_like(_to_torch_tensor(value, device=device), data_tensor)
    if baseline_tensor.shape == data_tensor.shape:
        return baseline_tensor

    try:
        return torch.broadcast_to(baseline_tensor, data_tensor.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"baseline[{index}] shape must match or broadcast to data[{index}] shape. "
            f"Expected shape {tuple(data_tensor.shape)}, got {tuple(baseline_tensor.shape)}."
        ) from exc


def _is_scalar_baseline(value):
    if isinstance(value, (list, tuple)):
        return False
    if isinstance(value, numbers.Real):
        return True
    if isinstance(value, (str, bytes)):
        return False
    try:
        array_value = np.asarray(value)
        return array_value.shape == () and np.issubdtype(array_value.dtype, np.number)
    except Exception:
        return False


def _to_torch_tensor(value, device):
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    elif tensor.is_floating_point() and tensor.dtype != torch.float32:
        tensor = tensor.float()
    return tensor.to(device)


def _cast_like(value, reference):
    return value.to(dtype=reference.dtype, device=reference.device)


def _validate_steps(steps):
    if not isinstance(steps, numbers.Integral) or int(steps) <= 0:
        raise ValueError("steps must be a positive integer.")


def _validate_dlv_index(structural_model, dlv_index):
    if not isinstance(dlv_index, numbers.Integral):
        raise TypeError("dlv_index must be an integer.")
    if int(dlv_index) < 0 or int(dlv_index) >= int(structural_model.ndims):
        raise ValueError(
            f"dlv_index must be between 0 and {int(structural_model.ndims) - 1}, got {dlv_index}."
        )


def _validate_float_tensors(data_tensors):
    for index, tensor in enumerate(data_tensors):
        if not tensor.is_floating_point():
            raise TypeError(
                f"Integrated gradients require floating-point inputs; data[{index}] has dtype {tensor.dtype}."
            )


def _validate_matching_shapes(data_tensors, baseline_tensors, name):
    for index, (data_tensor, baseline_tensor) in enumerate(zip(data_tensors, baseline_tensors)):
        data_shape = tuple(data_tensor.shape)
        baseline_shape = tuple(baseline_tensor.shape)
        if data_shape != baseline_shape:
            raise ValueError(
                f"{name}[{index}] shape must match data[{index}] shape. "
                f"Expected {data_shape}, got {baseline_shape}."
            )


def _ensure_model_is_built(structural_model, data_tensors):
    structural_model.to(_model_device(structural_model))
    structural_model.build(data_tensors)


def _model_device(structural_model):
    if hasattr(structural_model, "device"):
        return torch.device(structural_model.device)
    try:
        return next(structural_model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _snapshot_state_dict(structural_model):
    return {
        key: value.detach().clone()
        for key, value in structural_model.state_dict().items()
    }


def _restore_state_dict(structural_model, state_snapshot):
    structural_model.load_state_dict(state_snapshot, strict=True)


def _snapshot_training_modes(structural_model):
    return [(module, module.training) for module in structural_model.modules()]


def _restore_training_modes(training_modes):
    for module, training in training_modes:
        module.train(training)


class _temporary_eval_mode:
    def __init__(self, module):
        self.module = module
        self.old_mode = module.training

    def __enter__(self):
        self.module.eval()
        return self.module

    def __exit__(self, exc_type, exc_value, traceback):
        self.module.train(self.old_mode)
        return False
