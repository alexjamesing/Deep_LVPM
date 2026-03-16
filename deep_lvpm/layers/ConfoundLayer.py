#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConfoundLayer — pure PyTorch implementation.

Orthogonalises a data view with respect to a set of confound variables
by estimating and removing their linear contribution via moving-average
regression statistics.

Inputs:
    inputs[0] : data tensor  (batch, n_features)
    inputs[1] : confound tensor  (batch, n_confounds)
"""

import torch
import torch.nn as nn


class ConfoundLayer(nn.Module):
    """
    Remove confound effects from data inputs.

    The layer maintains moving-average estimates of the confound
    cross-covariance (C^T C) and cross-data covariance (C^T X), then
    regresses C out of X each forward pass.

    Parameters
    ----------
    tot_num : int
        Total training-set size (used to scale covariance estimates).
    epsilon : float
        Numerical stability offset.
    momentum : float  (Keras convention: close to 1 = slow update)
    diag_offset : float
        Regularisation added to diagonal of confound covariance before
        inversion.
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
        self._built = False

    # ------------------------------------------------------------------
    # Lazy build
    # ------------------------------------------------------------------

    def build(self, input_dim_data: int, input_dim_confound: int):
        if self._built:
            return

        aug_dim = input_dim_confound + 1   # +1 for the bias column

        # BatchNorm: old Keras code used BN default momentum=0.99 (EMA),
        # which is momentum=0.01 in PyTorch convention.
        self.batch_norm1 = nn.BatchNorm1d(input_dim_data, momentum=0.01, eps=self.epsilon)
        self.batch_norm2 = nn.BatchNorm1d(input_dim_confound, momentum=0.01, eps=self.epsilon)

        self.register_buffer("moving_conv2", torch.zeros(aug_dim, aug_dim))
        self.register_buffer("moving_convX", torch.zeros(aug_dim, input_dim_data))
        self.register_buffer("run", torch.zeros(()))

        self._built = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs, training: bool = False):
        """
        Parameters
        ----------
        inputs : list/tuple of two tensors
            [data (batch, n_features), confound (batch, n_confounds)]
        """
        input1, input2 = inputs

        if not self._built:
            self.build(input1.shape[1], input2.shape[1])

        bn_input1 = self.batch_norm1(input1)
        bn_input2 = self.batch_norm2(input2)

        X = bn_input1
        conv = bn_input2

        # Augment confound with bias column of ones
        ones = torch.ones(conv.shape[0], 1, dtype=conv.dtype, device=conv.device)
        conv_aug = torch.cat([ones, conv], dim=1)

        # Update moving statistics (no gradient needed)
        with torch.no_grad():
            self._update_moving_variables(X, conv_aug)

        # beta = (C^T C)^{-1} (C^T X)
        aug_dim = conv_aug.shape[1]
        eye = torch.eye(aug_dim, dtype=conv_aug.dtype, device=conv_aug.device)
        inv = torch.linalg.inv(self.moving_conv2 + self.diag_offset * eye)
        beta = inv @ self.moving_convX

        # Regress out confounds
        X_out = X - conv_aug @ beta
        return X_out

    # ------------------------------------------------------------------
    # Moving variable update
    # ------------------------------------------------------------------

    def _update_moving_variables(self, X: torch.Tensor, conv: torch.Tensor) -> None:
        x_dtype = X.dtype
        batch_size = X.shape[0]
        scale_fact = float(self.tot_num) / float(batch_size)

        # Zero momentum on the very first call (bug fix: was checking == 1)
        is_first = self.run.item() == 0.0
        m = 0.0 if is_first else self.momentum
        one_m = 1.0 - m

        conv_T = conv.T
        new_conv2 = m * self.moving_conv2 + scale_fact * one_m * (conv_T @ conv)
        new_convX = m * self.moving_convX + scale_fact * one_m * (conv_T @ X)

        self.moving_conv2.copy_(new_conv2)
        self.moving_convX.copy_(new_convX)
        self.run.fill_(1.0)
