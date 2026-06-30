#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Confound residualization layer implemented in PyTorch."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConfoundLayer(nn.Module):
    """Orthogonalise one input tensor with respect to a set of confounds."""

    def __init__(
        self,
        tot_num,
        epsilon: float = 1e-4,
        momentum: float = 0.95,
        diag_offset: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        self.tot_num = int(tot_num)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.diag_offset = float(diag_offset)
        self._built = False

    def build(self, input_shape):
        if self._built:
            return

        x_shape, confound_shape = input_shape
        num_output_features = int(x_shape[-1])
        num_confound_features = int(confound_shape[-1]) + 1

        self.batch_norm1 = nn.BatchNorm1d(
            num_output_features,
            eps=self.epsilon,
            momentum=1.0 - self.momentum,
        )
        self.batch_norm2 = nn.BatchNorm1d(
            num_confound_features - 1,
            eps=self.epsilon,
            momentum=1.0 - self.momentum,
        )

        self.register_buffer("run", torch.zeros(()))
        self.register_buffer("moving_conv2", torch.zeros(num_confound_features, num_confound_features))
        self.register_buffer("moving_convX", torch.zeros(num_confound_features, num_output_features))
        self._built = True

    def forward(self, inputs):
        input_x, input_confound = inputs

        if not self._built:
            self.build([input_x.shape, input_confound.shape])
            self.to(input_x.device)

        x = self.batch_norm1(input_x)
        confound = self.batch_norm2(input_confound)

        ones = torch.ones((confound.shape[0], 1), dtype=confound.dtype, device=confound.device)
        confound_aug = torch.cat([ones, confound], dim=1)

        if self.training:
            with torch.no_grad():
                self.update_moving_variables(x, confound_aug)

        eye = torch.eye(
            self.moving_conv2.shape[0],
            dtype=self.moving_conv2.dtype,
            device=self.moving_conv2.device,
        )
        beta = torch.linalg.solve(
            self.moving_conv2 + self.diag_offset * eye,
            self.moving_convX,
        )
        return x - confound_aug @ beta

    def update_moving_variables(self, x, confound_aug):
        matrix_dtype = self.moving_conv2.dtype
        scale_fact = torch.as_tensor(float(self.tot_num), dtype=matrix_dtype, device=x.device)
        scale_fact = scale_fact / torch.as_tensor(x.shape[0], dtype=matrix_dtype, device=x.device)

        momentum = torch.as_tensor(self.momentum, dtype=matrix_dtype, device=x.device)
        if float(self.run.detach().cpu()) == 0.0:
            momentum = torch.zeros_like(momentum)

        confound_transpose = confound_aug.T
        new_conv2 = momentum * self.moving_conv2 + scale_fact * (1.0 - momentum) * (
            confound_transpose @ confound_aug
        )
        new_convX = momentum * self.moving_convX + scale_fact * (1.0 - momentum) * (
            confound_transpose @ x
        )

        self.moving_conv2.copy_(new_conv2)
        self.moving_convX.copy_(new_convX)
        self.run.fill_(1.0)

    def get_config(self):
        return {
            "tot_num": self.tot_num,
            "epsilon": self.epsilon,
            "momentum": self.momentum,
            "diag_offset": self.diag_offset,
        }
