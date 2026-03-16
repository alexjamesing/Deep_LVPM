#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZCALayer — pure PyTorch implementation.

Alternative orthogonalisation layer using ZCA (Zero-phase Component
Analysis) whitening instead of the Moore-Penrose approach in
FactorLayer. Appended to each measurement model in StructuralModel.
"""

import torch
import torch.nn as nn


class ZCALayer(nn.Module):
    """
    Produces DLVs via linear projection followed by ZCA whitening.

    Orthogonalisation is performed outside this layer as part of
    StructuralModel (via ``weight_normalizer``), making the approach
    more convenient than the sequential Gram-Schmidt used in
    FactorLayer.

    Attributes
    ----------
    kernel_regularizer : tuple (l1, l2) or None
    epsilon : float
    momentum : float  (Keras convention)
    diag_offset : float
        Added to the diagonal of the covariance matrix to ensure
        invertibility.
    tot_num : int
    ndims : int
    """

    def __init__(
        self,
        kernel_regularizer=None,
        epsilon: float = 1e-3,
        momentum: float = 0.95,
        diag_offset: float = 1e-3,
        tot_num: int = None,
        ndims: int = None,
    ):
        super().__init__()
        self.kernel_regularizer = kernel_regularizer
        self.epsilon = epsilon
        self.momentum = momentum
        self.diag_offset = diag_offset
        self.tot_num = tot_num
        self.ndims = ndims
        self._built = False

    # ------------------------------------------------------------------
    # Lazy build
    # ------------------------------------------------------------------

    def build(self, input_dim: int):
        if self._built:
            return

        # BatchNorm: PyTorch momentum = 1 - Keras momentum
        self.batch_norm1 = nn.BatchNorm1d(
            input_dim,
            momentum=1.0 - self.momentum,
            eps=self.epsilon,
        )

        self.project = nn.Parameter(torch.randn(input_dim, self.ndims))

        self.register_buffer("moving_conv2", torch.eye(self.ndims))
        self.register_buffer("run", torch.zeros(()))

        self._built = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self.build(inputs.shape[1])
            self.to(inputs.device)

        X = self.batch_norm1(inputs)

        if self.training:
            with torch.no_grad():
                self._update_moving_variables(X)
        return X @ self.project

    # ------------------------------------------------------------------
    # Weight normaliser (called from StructuralModel, already no_grad)
    # ------------------------------------------------------------------

    def weight_normalizer(
        self,
        y: torch.Tensor,
        scale_fact: torch.Tensor,
        train_DLV: bool,
    ) -> torch.Tensor:
        """Normalise projection weights; return ZCA-whitened DLVs."""
        with torch.no_grad():
            denom = torch.sqrt(scale_fact * (y ** 2).sum(dim=0))
            self.project.data.copy_(self.project / denom)

            if not train_DLV:
                # Use moving average covariance
                C = self.moving_conv2 + self.diag_offset * torch.eye(
                    self.ndims, device=self.moving_conv2.device, dtype=self.moving_conv2.dtype
                )
            else:
                # Use batch-level covariance
                C = scale_fact * (y.T @ y) + self.diag_offset * torch.eye(
                    self.ndims, device=y.device, dtype=y.dtype
                )

            sqrt_inv = self._inv_sqrt_via_eigh(C)
            out_y = y @ sqrt_inv
        return out_y

    # ------------------------------------------------------------------
    # M^{-1/2} via eigendecomposition
    # ------------------------------------------------------------------

    def _inv_sqrt_via_eigh(self, M: torch.Tensor) -> torch.Tensor:
        """Compute M^{-1/2} = V * diag(lam^{-1/2}) * V^T."""
        M_sym = 0.5 * (M + M.T)
        eigvals, eigvecs = torch.linalg.eigh(M_sym)
        eigvals = eigvals.clamp(min=self.epsilon)
        inv_sqrt_vals = 1.0 / eigvals.sqrt()
        V_scaled = eigvecs * inv_sqrt_vals.unsqueeze(0)
        return V_scaled @ eigvecs.T

    # ------------------------------------------------------------------
    # Moving variable update
    # ------------------------------------------------------------------

    def _update_moving_variables(self, X: torch.Tensor) -> None:
        scale_fact = float(self.tot_num) / float(X.shape[0])
        y = X @ self.project

        m = self.momentum
        one_m = 1.0 - m

        self.moving_conv2.copy_(
            m * self.moving_conv2
            + scale_fact * one_m * (y.T @ y)
        )
        self.run.copy_(self.run + 1.0)

    # ------------------------------------------------------------------
    # Regularisation loss
    # ------------------------------------------------------------------

    def regularization_loss(self) -> torch.Tensor:
        device = self.project.device
        dtype = self.project.dtype
        penalty = torch.zeros((), device=device, dtype=dtype)

        if self.kernel_regularizer is None:
            return penalty

        l1, l2 = self.kernel_regularizer
        if l1 > 0:
            penalty = penalty + l1 * self.project.abs().sum()
        if l2 > 0:
            penalty = penalty + l2 * (self.project ** 2).sum()
        return penalty
