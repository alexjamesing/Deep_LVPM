#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FactorLayer — pure PyTorch implementation.

Appended to each measurement model in StructuralModel. Applies batch
normalization then Moore-Penrose orthogonalisation to produce `ndims`
orthogonal Deep Latent Variables (DLVs). Maintains moving statistics
for test-time inference.
"""

import torch
import torch.nn as nn


class FactorLayer(nn.Module):
    """
    Produces orthogonal DLVs via sequential linear projection +
    Gram-Schmidt orthogonalisation using both trainable and static
    (non-gradient) projection weight vectors.

    Attributes
    ----------
    kernel_regularizer : tuple (l1, l2) or None
        L1/L2 penalty applied to the trainable projection weights.
    epsilon : float
        Numerical stability offset for normalisation steps.
    momentum : float  (Keras convention: close to 1 = slow update)
        EMA momentum for moving statistics.  The underlying
        ``nn.BatchNorm1d`` receives ``1 - momentum`` so that both use
        the same convention.
    tot_num : int
        Total training-set size; used to scale covariance estimates.
    ndims : int
        Number of DLVs to extract.
    """

    def __init__(
        self,
        kernel_regularizer=None,
        epsilon: float = 1e-3,
        momentum: float = 0.99,
        tot_num: int = None,
        ndims: int = None,
    ):
        super().__init__()
        self.kernel_regularizer = kernel_regularizer
        self.epsilon = epsilon
        self.momentum = momentum
        self.tot_num = tot_num
        self.ndims = ndims
        self._built = False

    # ------------------------------------------------------------------
    # Lazy build — called on first forward() or explicitly
    # ------------------------------------------------------------------

    def build(self, input_dim: int):
        """Initialise all weights for the given feature dimension."""
        if self._built:
            return

        # BatchNorm: PyTorch momentum = 1 - Keras momentum
        self.batch_norm1 = nn.BatchNorm1d(
            input_dim,
            momentum=1.0 - self.momentum,
            eps=self.epsilon,
        )

        # Trainable projection vectors (one per DLV)
        self.linear_layer_list = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim, 1)) for _ in range(self.ndims)]
        )

        # Non-trainable static copies used for orthogonalisation
        for i in range(self.ndims):
            self.register_buffer(f"static_{i}", torch.randn(input_dim, 1))

        # Moving statistics
        self.register_buffer("DLV_mean", torch.zeros(self.ndims, 1))
        self.register_buffer("DLV_var", torch.ones(self.ndims, 1))
        self.register_buffer("moving_convX", torch.zeros(self.ndims, input_dim))
        self.register_buffer("run", torch.zeros(()))

        self._built = True

    # ------------------------------------------------------------------
    # Helpers: static buffer list access
    # ------------------------------------------------------------------

    def _static(self, i: int) -> torch.Tensor:
        return getattr(self, f"static_{i}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self.build(inputs.shape[1])
            self.to(inputs.device)

        X = self.batch_norm1(inputs)

        if self.training:
            DLV_all = self._calculate_batch_DLV_static(X)
            out = self._calculate_batch_DLV_train(X, DLV_all)
            with torch.no_grad():
                self._update_moving_variables(X, DLV_all)
        else:
            out = self._calculate_DLV_test(X)

        return out

    # ------------------------------------------------------------------
    # Weight normaliser (called from StructuralModel, already no_grad)
    # ------------------------------------------------------------------

    def weight_normalizer(
        self,
        y: torch.Tensor,
        scale_fact: torch.Tensor,
        train_DLV: bool,
    ) -> torch.Tensor:
        """Normalise projection weights; return normalised DLVs."""
        with torch.no_grad():
            for i in range(self.ndims):
                yi = y[:, i]
                denom = torch.sqrt(scale_fact * (yi ** 2).sum())
                new_w = self.linear_layer_list[i] / denom
                self.linear_layer_list[i].data.copy_(new_w)
                self._static(i).copy_(new_w)

            y_denom = torch.sqrt(scale_fact * (y ** 2).sum(dim=0))
            out_y = y / y_denom
        return out_y

    # ------------------------------------------------------------------
    # Moving variable update
    # ------------------------------------------------------------------

    def _update_moving_variables(
        self, X: torch.Tensor, DLV_all: torch.Tensor
    ) -> None:
        batch_size = X.shape[0]
        scale_fact = float(self.tot_num) / float(batch_size)

        # Zero momentum on first call
        is_first = self.run.item() == 0.0
        m = 0.0 if is_first else self.momentum
        one_m = 1.0 - m

        batch_DLV_mean = DLV_all.mean(dim=0).unsqueeze(1)   # (ndims, 1)
        batch_DLV_var = DLV_all.var(dim=0).unsqueeze(1)  # (ndims, 1)

        self.DLV_mean.copy_(m * self.DLV_mean + one_m * batch_DLV_mean)
        self.DLV_var.copy_(m * self.DLV_var + one_m * batch_DLV_var)

        batch_DLV_norm = (
            (DLV_all - batch_DLV_mean.T)
            / (batch_DLV_var.sqrt().T + self.epsilon)
        )

        self.moving_convX.copy_(
            m * self.moving_convX
            + scale_fact * one_m * (batch_DLV_norm.T @ X)
        )

        self.run.fill_(1.0)

    # ------------------------------------------------------------------
    # Orthogonalisation helpers
    # ------------------------------------------------------------------

    def _orthogonalise_train(
        self, X: torch.Tensor, DLV_prev: torch.Tensor
    ) -> torch.Tensor:
        """Orthogonalise X w.r.t. previous DLVs using batch statistics."""
        DLV_batch = (DLV_prev - DLV_prev.mean(dim=0)) / (
            DLV_prev.std(dim=0) + self.epsilon
        )
        denom = float(X.shape[0])
        beta = (DLV_batch.T @ X) / denom
        return X - DLV_batch @ beta

    def _orthogonalise_test(
        self, X: torch.Tensor, DLV_prev: torch.Tensor
    ) -> torch.Tensor:
        """Orthogonalise X w.r.t. previous DLVs using moving statistics."""
        i = DLV_prev.shape[1]
        DLV_norm = (
            DLV_prev - self.DLV_mean[:i, :].T
        ) / (self.DLV_var[:i, :].sqrt().T + self.epsilon)
        beta = self.moving_convX[:i, :] / float(self.tot_num)
        return X - DLV_norm @ beta

    # ------------------------------------------------------------------
    # DLV calculation methods
    # ------------------------------------------------------------------

    def _calculate_batch_DLV_static(self, X: torch.Tensor) -> torch.Tensor:
        """Compute DLVs with static (non-trainable) projection vectors."""
        DLV_all = None
        for i in range(self.ndims):
            if i == 0:
                DLV = X @ self._static(i)
                DLV_all = DLV
            else:
                ortho = self._orthogonalise_train(X, DLV_all)
                DLV = ortho @ self._static(i)
                DLV_all = torch.cat([DLV_all, DLV], dim=1)
        return DLV_all

    def _calculate_batch_DLV_train(
        self, X: torch.Tensor, DLV_all: torch.Tensor
    ) -> torch.Tensor:
        """Compute DLVs with trainable projection vectors."""
        out = None
        for i in range(self.ndims):
            if i == 0:
                out = X @ self.linear_layer_list[i]
            else:
                ortho = self._orthogonalise_train(X, DLV_all[:, :i])
                out_i = ortho @ self.linear_layer_list[i]
                out = torch.cat([out, out_i], dim=1)
        return out

    def _calculate_DLV_test(self, X: torch.Tensor) -> torch.Tensor:
        """Compute DLVs at test time using moving statistics."""
        out = None
        for i in range(self.ndims):
            w = self.linear_layer_list[i]
            if i == 0:
                out = X @ w
            else:
                ortho = self._orthogonalise_test(X, out)
                out_i = ortho @ w
                out = torch.cat([out, out_i], dim=1)
        return out

    # ------------------------------------------------------------------
    # Regularisation loss
    # ------------------------------------------------------------------

    def regularization_loss(self) -> torch.Tensor:
        """L1/L2 penalty on trainable projection weights, or zero."""
        device = self.linear_layer_list[0].device
        dtype = self.linear_layer_list[0].dtype
        penalty = torch.zeros((), device=device, dtype=dtype)

        if self.kernel_regularizer is None:
            return penalty

        l1, l2 = self.kernel_regularizer
        for w in self.linear_layer_list:
            if l1 > 0:
                penalty = penalty + l1 * w.abs().sum()
            if l2 > 0:
                penalty = penalty + l2 * (w ** 2).sum()
        return penalty
