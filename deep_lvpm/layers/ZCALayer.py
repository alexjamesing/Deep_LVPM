#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZCA DLVPM projection layer implemented in PyTorch."""

from __future__ import annotations

import torch
import torch.nn as nn

from deep_lvpm import regularizers
from deep_lvpm.layers.FactorLayer import soft_threshold


class ZCALayer(nn.Module):
    """Project measurement-model features and whiten them with ZCA statistics."""

    def __init__(
        self,
        kernel_regularizer=None,
        epsilon: float = 1e-3,
        momentum: float = 0.95,
        diag_offset: float = 1e-3,
        tot_num: int | None = None,
        ndims: int | None = None,
        sparse_l1: float = 0.0,
        newton_schulz_iters: int = 20,
        **kwargs,
    ):
        super().__init__()
        self.kernel_regularizer = kernel_regularizer
        self.epsilon = float(epsilon)
        self.momentum = float(momentum)
        self.diag_offset = float(diag_offset)
        self.tot_num = tot_num
        self.ndims = int(ndims) if ndims is not None else None
        self.sparse_l1 = float(sparse_l1)
        self.newton_schulz_iters = int(newton_schulz_iters)
        self._built = False

    def build(self, input_shape_or_dim):
        if self._built:
            return

        if isinstance(input_shape_or_dim, int):
            input_dim = input_shape_or_dim
        else:
            input_dim = int(input_shape_or_dim[-1])

        if self.ndims is None:
            raise ValueError("ZCALayer requires ndims.")

        self.batch_norm1 = nn.BatchNorm1d(
            input_dim,
            eps=self.epsilon,
            momentum=1.0 - self.momentum,
        )
        self.project = nn.Parameter(torch.randn(input_dim, self.ndims))
        self.register_buffer("run", torch.zeros(()))
        self.register_buffer("moving_conv2", torch.eye(self.ndims))
        self._built = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self.build(inputs.shape[-1])
            self.to(inputs.device)

        X = self.batch_norm1(inputs)

        if self.training:
            with torch.no_grad():
                self.update_moving_variables(X)

        return X @ self.project

    def apply_constraints(self):
        if self.sparse_l1 <= 0.0:
            return
        with torch.no_grad():
            self.project.copy_(soft_threshold(self.project, self.sparse_l1))

    def regularization_loss(self, reference: torch.Tensor | None = None) -> torch.Tensor:
        if not self._built:
            if reference is None:
                return torch.tensor(0.0)
            return torch.zeros((), dtype=reference.dtype, device=reference.device)
        return regularizers.penalty([self.project], self.kernel_regularizer, reference=reference)

    def inv_sqrt_newton_schulz(self, matrix: torch.Tensor) -> torch.Tensor:
        matrix_sym = 0.5 * (matrix + matrix.T)
        eye = torch.eye(matrix_sym.shape[0], dtype=matrix_sym.dtype, device=matrix_sym.device)
        matrix_sym = matrix_sym + self.epsilon * eye

        norm = torch.sqrt(torch.sum(matrix_sym * matrix_sym))
        norm = torch.clamp(norm, min=self.epsilon)
        y = matrix_sym / norm
        z = eye.clone()

        for _ in range(self.newton_schulz_iters):
            update = 0.5 * (3.0 * eye - z @ y)
            y = y @ update
            z = update @ z

        inv_sqrt = z / torch.sqrt(norm)
        return 0.5 * (inv_sqrt + inv_sqrt.T)

    def weight_normalizer(self, inputs):
        y, scale_fact, train_DLV = inputs
        scale_fact = torch.as_tensor(scale_fact, dtype=y.dtype, device=y.device)

        with torch.no_grad():
            denom = torch.sqrt(scale_fact * torch.sum(torch.square(y), dim=0) + self.epsilon)
            self.project.copy_(self.project / denom)
            y = y / denom

            eye = torch.eye(self.ndims, dtype=y.dtype, device=y.device)
            if train_DLV is False:
                covariance = self.moving_conv2.to(dtype=y.dtype, device=y.device) + self.diag_offset * eye
            else:
                covariance = scale_fact * (y.T @ y) + self.diag_offset * eye

            sqrt_inv_y = self.inv_sqrt_newton_schulz(covariance)
            out_y = y @ sqrt_inv_y

        return out_y

    def update_moving_variables(self, X: torch.Tensor):
        scale_fact = torch.as_tensor(float(self.tot_num), dtype=X.dtype, device=X.device)
        scale_fact = scale_fact / torch.as_tensor(X.shape[0], dtype=X.dtype, device=X.device)
        y = X @ self.project

        momentum = torch.as_tensor(self.momentum, dtype=X.dtype, device=X.device)
        if float(self.run.detach().cpu()) == 0.0:
            momentum = torch.zeros_like(momentum)

        self.moving_conv2.copy_(
            momentum * self.moving_conv2 + scale_fact * (1.0 - momentum) * (y.T @ y)
        )
        self.run.add_(1.0)

    def get_config(self):
        return {
            "kernel_regularizer": self.kernel_regularizer,
            "epsilon": self.epsilon,
            "momentum": self.momentum,
            "diag_offset": self.diag_offset,
            "tot_num": self.tot_num,
            "ndims": self.ndims,
            "sparse_l1": self.sparse_l1,
            "newton_schulz_iters": self.newton_schulz_iters,
        }
