#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZCALayer — pure PyTorch implementation.

Algorithm
---------
``ZCALayer`` is appended to the end of each measurement model and produces
``ndims`` DLVs from the model's output.  Unlike ``FactorLayer``, DLVs are
produced by a single matrix projection followed by ZCA whitening applied
*outside* this layer (in ``StructuralModel._weight_normaliser``).

The computation inside this layer is:

  1. Batch-normalize input X.
  2. Project: Y = X W  where W ∈ ℝ^{n_features × ndims} is learned.

ZCA whitening (``weight_normalizer``)
--------------------------------------
After the projection, ``StructuralModel._weight_normaliser`` calls
``weight_normalizer``, which applies ZCA whitening to Y:

  Y_white = Y · C^{-1/2}

where C = Y^T Y is the within-batch (or moving) covariance of the projected
output.  This enforces approximate orthogonality of the DLVs.  C^{-1/2} is
computed via eigendecomposition (``_inv_sqrt_via_eigh``).

A diagonal offset is added to C before inversion for numerical stability:

  C^{-1/2} = (C + δI)^{-1/2}

Moving covariance
-----------------
The layer accumulates a moving estimate of the projection covariance
``moving_conv2 ≈ E[Y^T Y]`` during training.  When ``train_DLV=False``,
``weight_normalizer`` uses this moving covariance instead of the current
batch, making it suitable for stable test-time whitening.
"""

import torch
import torch.nn as nn


class ZCALayer(nn.Module):
    """Orthogonalization layer implementing ZCA whitening.

    Appended automatically to each view's measurement model by
    ``StructuralModel``.  Produces ``ndims`` DLVs from the measurement
    model's output via a linear projection followed by ZCA whitening.

    Unlike ``FactorLayer``, the orthogonalization happens outside this layer:
    ``ZCALayer`` only performs batch normalization and the linear projection.
    The ZCA whitening step is applied in ``weight_normalizer``, which is
    called by ``StructuralModel._weight_normaliser`` at the end of pass 1.

    Attributes
    ----------
    kernel_regularizer : tuple (l1, l2) or None
        L1/L2 regularization coefficients for the projection matrix W.
    epsilon : float
        Numerical stability constant for batch normalization and the
        eigendecomposition (eigenvalues are clamped to this value).
    momentum : float
        Momentum for the moving covariance estimate (Keras convention:
        higher = slower update).  PyTorch BatchNorm1d uses ``1 - momentum``.
    diag_offset : float
        Diagonal regularization δ added to the covariance before inverting
        (prevents singular matrices).
    tot_num : int
        Total number of training samples; used to scale the batch covariance
        to dataset scale (scale_fact = tot_num / batch_size).
    ndims : int
        Number of DLVs to extract.
    project : nn.Parameter
        Trainable projection matrix W of shape ``(n_features, ndims)``.
    moving_conv2 : torch.Tensor (buffer)
        Moving estimate of Y^T Y, shape ``(ndims, ndims)``.  Initialized to
        the identity so the first whitening step is a no-op.  Used in
        ``weight_normalizer`` when ``train_DLV=False``.
    run : torch.Tensor (buffer)
        Counter incremented each time ``_update_moving_variables`` is called.
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
        self._initialized: torch.Tensor

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
        self.register_buffer("_initialized", torch.tensor(False))

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
            denom = torch.sqrt(scale_fact * (y**2).sum(dim=0))
            self.project.data.copy_(self.project / denom)

            if not train_DLV:
                # Use moving average covariance
                C = self.moving_conv2 + self.diag_offset * torch.eye(
                    self.ndims,
                    device=self.moving_conv2.device,
                    dtype=self.moving_conv2.dtype,
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

        # training set size divided by batch size, used to scale covariance estimates
        scale_fact = float(self.tot_num) / float(X.shape[0])

        # Zero momentum on the first call
        m = 0.0 if not self._initialized else self.momentum
        one_m = 1.0 - m

        y = X @ self.project
        conv2_new = m * self.moving_conv2 + scale_fact * one_m * (y.T @ y)
        self.moving_conv2.copy_(conv2_new)

        # set flag after first update
        self._initialized.fill_(True)

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
            penalty = penalty + l2 * (self.project**2).sum()
        return penalty
