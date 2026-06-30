#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Moore-Penrose DLVPM projection layer implemented in PyTorch."""

from __future__ import annotations

import torch
import torch.nn as nn

from deep_lvpm import regularizers


def soft_threshold(weights: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.sign(weights) * torch.clamp(torch.abs(weights) - float(threshold), min=0.0)


class FactorLayer(nn.Module):
    """Generate orthogonal deep latent variables using iterative deflation."""

    def __init__(
        self,
        kernel_regularizer=None,
        epsilon: float = 1e-3,
        momentum: float = 0.99,
        tot_num: int | None = None,
        ndims: int | None = None,
        sparse_l1: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.kernel_regularizer = kernel_regularizer
        self.epsilon = float(epsilon)
        self.momentum = float(momentum)
        self.tot_num = tot_num
        self.ndims = int(ndims) if ndims is not None else None
        self.sparse_l1 = float(sparse_l1)
        self._built = False

    def build(self, input_shape_or_dim):
        if self._built:
            return

        if isinstance(input_shape_or_dim, int):
            input_dim = input_shape_or_dim
        else:
            input_dim = int(input_shape_or_dim[-1])

        if self.ndims is None:
            raise ValueError("FactorLayer requires ndims.")

        self.batch_norm1 = nn.BatchNorm1d(
            input_dim,
            eps=self.epsilon,
            momentum=1.0 - self.momentum,
        )
        self.linear_layer_list = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim, 1)) for _ in range(self.ndims)]
        )

        for dim_index in range(self.ndims):
            self.register_buffer(f"static_projection_weight_{dim_index}", torch.randn(input_dim, 1))

        self.register_buffer("run", torch.zeros(()))
        self.register_buffer("DLV_mean", torch.zeros(self.ndims, 1))
        self.register_buffer("DLV_var", torch.ones(self.ndims, 1))
        self.register_buffer("moving_convX", torch.zeros(self.ndims, input_dim))
        self._built = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self.build(inputs.shape[-1])
            self.to(inputs.device)

        X = self.batch_norm1(inputs)

        if self.training:
            DLV_all = self.calculate_batch_DLV_static(X)
            out = self.calculate_batch_DLV_train(X, DLV_all)
            with torch.no_grad():
                self.update_moving_variables([X, DLV_all])
        else:
            out = self.calculate_DLV_test(X)

        return out

    def _static_weight(self, dim_index: int) -> torch.Tensor:
        return getattr(self, f"static_projection_weight_{dim_index}")

    def apply_constraints(self):
        if self.sparse_l1 <= 0.0:
            return
        with torch.no_grad():
            for weight in self.linear_layer_list:
                weight.copy_(soft_threshold(weight, self.sparse_l1))

    def regularization_loss(self, reference: torch.Tensor | None = None) -> torch.Tensor:
        return regularizers.penalty(self.linear_layer_list, self.kernel_regularizer, reference=reference)

    def weight_normalizer(self, inputs):
        y, scale_fact, train_DLV = inputs
        del train_DLV

        with torch.no_grad():
            scale_fact = torch.as_tensor(scale_fact, dtype=y.dtype, device=y.device)
            for dim_index in range(self.ndims):
                yi = y[:, dim_index]
                denom = torch.sqrt(scale_fact * torch.sum(torch.square(yi)) + self.epsilon)
                self.linear_layer_list[dim_index].copy_(self.linear_layer_list[dim_index] / denom)
                self._static_weight(dim_index).copy_(self.linear_layer_list[dim_index])

            y_denom = torch.sqrt(scale_fact * torch.sum(torch.square(y), dim=0) + self.epsilon)
            out_y = y / y_denom

        return out_y

    def update_moving_variables(self, inputs):
        X, DLV_all = inputs
        batch_size = torch.as_tensor(X.shape[0], dtype=X.dtype, device=X.device)
        scale_fact = torch.as_tensor(float(self.tot_num), dtype=X.dtype, device=X.device) / batch_size

        first = (self.run == 0).to(dtype=X.dtype)
        momentum = (1.0 - first) * torch.as_tensor(self.momentum, dtype=X.dtype, device=X.device)

        batch_DLV_mean = torch.mean(DLV_all, dim=0, keepdim=False).reshape(self.ndims, 1)
        batch_DLV_var = torch.var(DLV_all, dim=0, unbiased=False).reshape(self.ndims, 1)

        self.DLV_mean.copy_(momentum * self.DLV_mean + (1.0 - momentum) * batch_DLV_mean)
        self.DLV_var.copy_(momentum * self.DLV_var + (1.0 - momentum) * batch_DLV_var)

        batch_DLV_norm = (DLV_all - batch_DLV_mean.T) / (torch.sqrt(batch_DLV_var).T + self.epsilon)
        self.moving_convX.copy_(
            momentum * self.moving_convX
            + scale_fact * (1.0 - momentum) * (batch_DLV_norm.T @ X)
        )
        self.run.fill_(1.0)

    def orthogonalisation_train(self, inputs):
        X, DLV_prev = inputs
        DLV_batch = (DLV_prev - torch.mean(DLV_prev, dim=0)) / (torch.std(DLV_prev, dim=0, unbiased=False) + self.epsilon)
        denom = torch.as_tensor(X.shape[0], dtype=X.dtype, device=X.device)
        beta = DLV_batch.T @ X / denom
        return X - DLV_batch @ beta

    def orthogonalisation_test(self, inputs):
        X, DLV_prev = inputs
        dim_count = DLV_prev.shape[1]
        DLV_norm = (
            DLV_prev - self.DLV_mean[:dim_count, :].T
        ) / (torch.sqrt(self.DLV_var[:dim_count, :]).T + self.epsilon)
        beta = self.moving_convX[:dim_count, :] / float(self.tot_num)
        return X - DLV_norm @ beta

    def calculate_batch_DLV_static(self, X: torch.Tensor) -> torch.Tensor:
        DLV_all = None
        for dim_index in range(self.ndims):
            if dim_index == 0:
                DLV = X @ self._static_weight(dim_index)
                DLV_all = DLV
            else:
                ortho_output = self.orthogonalisation_train([X, DLV_all])
                DLV = ortho_output @ self._static_weight(dim_index)
                DLV_all = torch.cat([DLV_all, DLV], dim=1)
        return DLV_all

    def calculate_batch_DLV_train(self, X: torch.Tensor, DLV_all: torch.Tensor) -> torch.Tensor:
        out = None
        for dim_index in range(self.ndims):
            if dim_index == 0:
                out = X @ self.linear_layer_list[dim_index]
            else:
                ortho_output = self.orthogonalisation_train([X, DLV_all[:, :dim_index]])
                out_i = ortho_output @ self.linear_layer_list[dim_index]
                out = torch.cat([out, out_i], dim=1)
        return out

    def calculate_DLV_test(self, X: torch.Tensor) -> torch.Tensor:
        out = None
        for dim_index in range(self.ndims):
            weight = self.linear_layer_list[dim_index]
            if dim_index == 0:
                out = X @ weight
            else:
                ortho_output = self.orthogonalisation_test([X, out])
                out_i = ortho_output @ weight
                out = torch.cat([out, out_i], dim=1)
        return out

    def get_config(self):
        return {
            "kernel_regularizer": self.kernel_regularizer,
            "epsilon": self.epsilon,
            "momentum": self.momentum,
            "tot_num": self.tot_num,
            "ndims": self.ndims,
            "sparse_l1": self.sparse_l1,
        }
