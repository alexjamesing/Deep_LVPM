#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:24:12 2023

@author: ing
"""

import keras
from keras import ops


@keras.utils.register_keras_serializable(package="deep_lvpm", name="ConfoundLayer")
class ConfoundLayer(keras.layers.Layer):

    """Orthogonalise one input tensor with respect to a set of confounds."""

    def __init__(self, tot_num, epsilon=1e-4, momentum=0.95, diag_offset=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.tot_num = tot_num
        self.momentum = momentum
        self.epsilon = epsilon
        self.diag_offset = diag_offset

    def build(self, input_shape):
        x_shape, confound_shape = input_shape

        if x_shape[-1] is None or confound_shape[-1] is None:
            raise ValueError("ConfoundLayer requires known feature dimensions for both inputs.")

        self.batch_norm1 = keras.layers.BatchNormalization(
            name="batch_norm1_confound",
            momentum=self.momentum,
            epsilon=self.epsilon,
        )
        self.batch_norm2 = keras.layers.BatchNormalization(
            name="batch_norm2_confound",
            momentum=self.momentum,
            epsilon=self.epsilon,
        )

        num_confound_features = int(confound_shape[-1]) + 1
        num_output_features = int(x_shape[-1])

        self.run = self.add_weight(
            name="confoundlayer_run",
            shape=(),
            initializer="zeros",
            trainable=False,
        )
        self.moving_conv2 = self.add_weight(
            name="moving_conv2",
            shape=[num_confound_features, num_confound_features],
            initializer="zeros",
            trainable=False,
        )
        self.moving_convX = self.add_weight(
            name="moving_convX",
            shape=[num_confound_features, num_output_features],
            initializer="zeros",
            trainable=False,
        )

        super().build(input_shape)

    def call(self, inputs, training=False):
        input_x, input_confound = inputs

        x = self.batch_norm1(input_x, training=training)
        confound = self.batch_norm2(input_confound, training=training)

        batch_size = ops.shape(confound)[0]
        ones = ops.ones((batch_size, 1), dtype=ops.dtype(confound))
        confound_aug = ops.concatenate([ones, confound], axis=1)

        self.update_moving_variables(x, confound_aug)

        matrix_dtype = ops.dtype(self.moving_conv2)
        eye = ops.eye(
            self.moving_conv2.shape[0],
            self.moving_conv2.shape[1],
            dtype=matrix_dtype,
        )
        diag_offset = ops.convert_to_tensor(self.diag_offset, dtype=matrix_dtype)
        beta = ops.matmul(
            ops.linalg.inv(self.moving_conv2 + diag_offset * eye),
            self.moving_convX,
        )

        return x - ops.matmul(confound_aug, beta)

    def update_moving_variables(self, x, confound_aug):
        matrix_dtype = ops.dtype(self.moving_conv2)
        one = ops.convert_to_tensor(1.0, dtype=matrix_dtype)
        zero = ops.convert_to_tensor(0.0, dtype=matrix_dtype)
        momentum_value = ops.convert_to_tensor(self.momentum, dtype=matrix_dtype)

        momentum = ops.where(
            ops.equal(self.run, one),
            zero,
            momentum_value,
        )

        batch_size = ops.cast(ops.shape(x)[0], matrix_dtype)
        scale_fact = ops.cast(self.tot_num, matrix_dtype) / batch_size
        one_minus_momentum = one - momentum

        confound_transpose = ops.transpose(confound_aug, (1, 0))
        new_conv2 = (
            momentum * self.moving_conv2
            + scale_fact * one_minus_momentum * ops.matmul(confound_transpose, confound_aug)
        )
        new_convX = (
            momentum * self.moving_convX
            + scale_fact * one_minus_momentum * ops.matmul(confound_transpose, x)
        )

        self.moving_conv2.assign(new_conv2)
        self.moving_convX.assign(new_convX)
        self.run.assign(one)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "tot_num": self.tot_num,
                "epsilon": self.epsilon,
                "momentum": self.momentum,
                "diag_offset": self.diag_offset,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
