#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConfoundLayer — pure PyTorch implementation.

Algorithm
---------
``ConfoundLayer`` takes two inputs — data X and a confound matrix C — and
returns the residuals of X after regressing out C.  This is equivalent to
ordinary least squares (OLS) regression of X on C:

  X_clean = X − C_aug β

where C_aug = [1 | C] is the confound matrix augmented with a bias column,
and β is the OLS coefficient matrix:

  β = (C_aug^T C_aug)^{-1} C_aug^T X

Because gradient-based optimization requires that β be available as a
differentiable or at least stable estimate throughout training, the layer
accumulates *moving* estimates of the two key statistics:

  moving_conv2 ≈ E[C_aug^T C_aug]   (confound Gram matrix, dataset-scaled)
  moving_convX ≈ E[C_aug^T X]       (confound-data cross-product, dataset-scaled)

These are updated each training batch via exponential moving average and used
directly to compute β.

Usage
-----
Place ``ConfoundLayer`` at the beginning of a view's measurement model:

  confound_layer = ConfoundLayer(tot_num=N)
  X_clean = confound_layer([data_tensor, confound_tensor])
  ...
"""

import torch
import torch.nn as nn


class ConfoundLayer(nn.Module):
    """Remove confound effects from a data representation via OLS regression.

    Takes two inputs: the data to be cleaned and the confound matrix.
    Returns the OLS residuals X − C_aug β, where C_aug = [1 | C] includes
    a bias column and β is estimated from accumulated moving statistics.

    Attributes
    ----------
    tot_num : int
        Total number of training samples; used to scale the batch-level
        statistics to dataset scale.
    epsilon : float
        Numerical stability constant for batch normalisation.
    momentum : float
        Momentum for the exponential moving average of statistics
        (Keras convention: close to 1 = slow update).
    diag_offset : float
        Diagonal regularization added to the confound Gram matrix before
        inversion (prevents singular matrices when confounds are collinear).
    batch_norm1 : nn.BatchNorm1d
        BatchNormalization applied to the data input.
    batch_norm2 : nn.BatchNorm1d
        BatchNormalization applied to the confound input.
    moving_conv2 : Tensor
        Moving estimate of C_aug^T C_aug,
        shape ``(n_confounds+1, n_confounds+1)``.
    moving_convX : Tensor
        Moving estimate of C_aug^T X,
        shape ``(n_confounds+1, n_features)``.
    """

    def __init__(
        self,
        tot_num: int,
        epsilon: float = 1e-4,
        momentum: float = 0.95,
        diag_offset: float = 1e-3,
    ):
        super().__init__()
        self.tot_num = tot_num
        self.epsilon = epsilon
        self.momentum = momentum
        self.diag_offset = diag_offset
        self._is_built = False
        # Type annotations for registered buffers (set in build()).
        self.moving_conv2: torch.Tensor
        self.moving_convX: torch.Tensor
        self._initialized: torch.Tensor

    # ------------------------------------------------------------------
    # Lazy build
    # ------------------------------------------------------------------

    def build(self, input_dim_data: int, input_dim_confound: int):
        """Create batch-normalisation layers and moving-statistic buffers.

        Parameters
        ----------
        input_dim_data : int
            Number of features in the data input (n_features).
        input_dim_confound : int
            Number of confound variables (n_confounds).  The augmented
            dimension will be n_confounds + 1 because a bias column of
            ones is prepended to C.
        """
        if self._is_built:
            raise RuntimeError("ConfoundLayer is already built.")

        # n_confounds + 1 because a bias column of ones is appended to C.
        aug_dim = input_dim_confound + 1

        # BatchNorm: Keras default momentum=0.99 (EMA) ≡ momentum=0.01 in
        # PyTorch convention (PyTorch uses 1 - Keras_momentum).
        self.batch_norm1 = nn.BatchNorm1d(
            input_dim_data,
            momentum=0.01,
            eps=self.epsilon,
        )
        self.batch_norm2 = nn.BatchNorm1d(
            input_dim_confound,
            momentum=0.01,
            eps=self.epsilon,
        )

        # Gram matrix of the augmented confounds: C_aug^T C_aug.
        self.register_buffer("moving_conv2", torch.zeros(aug_dim, aug_dim))
        # Cross-product of augmented confounds and data: C_aug^T X.
        self.register_buffer("moving_convX", torch.zeros(aug_dim, input_dim_data))
        # Flag to detect the first forward pass (used to bootstrap moving
        # statistics from scratch rather than blending with zero-initialized
        # weights).
        self.register_buffer("_initialized", torch.tensor(False))

        self._is_built = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs):
        """Regress confounds out of the data input.

        Parameters
        ----------
        inputs : list or tuple of two tensors
            ``[X, C]`` where ``X`` is the data ``(batch, n_features)`` and
            ``C`` is the confound matrix ``(batch, n_confounds)``.

        Returns
        -------
        Tensor
            Cleaned data ``X − C_aug β``, same shape as ``X``.
        """
        input1, input2 = inputs

        if not self._is_built:
            self.build(
                input_dim_data=input1.shape[1],
                input_dim_confound=input2.shape[1],
            )

        X = self.batch_norm1(input1)
        C = self.batch_norm2(input2)

        # Augment confounds with a column of ones (intercept / bias term).
        ones = torch.ones(C.shape[0], 1, dtype=C.dtype, device=C.device)
        C_aug = torch.cat([ones, C], dim=1)

        # Update moving statistics so β stays current with the data
        # distribution.  Only update during training (unlike the Keras version
        # which updated unconditionally; inference should use frozen stats).
        if self.training:
            with torch.no_grad():
                self._update_moving_variables(X, C_aug)

        # Compute OLS coefficient matrix: β = (C_aug^T C_aug + δI)^{-1} C_aug^T X
        aug_dim = C_aug.shape[1]
        eye = torch.eye(aug_dim, dtype=C_aug.dtype, device=C_aug.device)
        inv = torch.linalg.inv(self.moving_conv2 + self.diag_offset * eye)
        beta = inv @ self.moving_convX

        # Return residuals: X with the estimated confound contribution removed.
        return X - C_aug @ beta

    # ------------------------------------------------------------------
    # Moving statistics update
    # ------------------------------------------------------------------

    def _update_moving_variables(self, X: torch.Tensor, C_aug: torch.Tensor) -> None:
        """Update moving estimates of the confound Gram matrix and cross-product.

        On the first call (``_initialized == False``), momentum is set to 0
        so the moving statistics are initialized directly from the first batch
        rather than being blended with the zero-initialized buffers.

        Parameters
        ----------
        X : Tensor
            Batch-normalized data, shape ``(batch, n_features)``.
        C_aug : Tensor
            Augmented (bias-prepended) confound matrix,
            shape ``(batch, n_confounds+1)``.
        """
        # Scale batch statistics to dataset scale (equivalent to computing
        # full-dataset covariance from a single batch estimate).
        scale_fact = float(self.tot_num) / float(X.shape[0])

        # Bootstrap: zero momentum on the very first call so the buffers are
        # set directly from the first batch rather than blended with zeros.
        m = 0.0 if not self._initialized else self.momentum
        one_m = 1.0 - m

        C_aug_T = C_aug.T
        conv2_new = m * self.moving_conv2 + scale_fact * one_m * (C_aug_T @ C_aug)
        convX_new = m * self.moving_convX + scale_fact * one_m * (C_aug_T @ X)
        self.moving_conv2.copy_(conv2_new)
        self.moving_convX.copy_(convX_new)

        self._initialized.fill_(True)
