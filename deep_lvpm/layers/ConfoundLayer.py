#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:24:12 2023

@author: ing
"""

import keras 
from keras import saving
from keras import ops

@keras.utils.register_keras_serializable(package="deep_lvpm",name="ConfoundLayer")
class ConfoundLayer(keras.layers.Layer):
    
    """ The purpose of this layer is to orthogonalise data-inputs with respect
    to a set of input confounds
    
    call_inputs:
    input[0]: This should be the data input that we want to orthogonalise with
    respect to input[1]
    input[1]: We orthogonlise input[0] with respect to input[1]
    
    """
    
    
    def __init__(self, tot_num, epsilon=1e-4, momentum=0.95, diag_offset=1e-3, **kwargs):
        
        """
        Initialize the custom layer.

        Parameters:
        tot_num (int): Total number.
        epsilon (float): Offset for batch normalization.
        momentum (float): Momentum for covariance matrices.
        diag_offset (float): Offset added to the diagonal of the covariance matrix.
        """

        super().__init__(**kwargs)
        self.tot_num = tot_num
        self.momentum = momentum ## This is the amount of momentum that covariance matrices are subject to (see pseudo-code for more details)
        self.epsilon = epsilon ## This is the offset determined during batch normalisation
        self.diag_offset =diag_offset ## This is a offset added to the diagonal of the covariance matrix between confounds, to ensure that this matrix is invertable
       
    def build(self, input_shape):
        
        """ In this function, the model builds and assigns values to the weights used in the Deep-PLS analysis.
        The function builds the list of projection vectors used to map associations between different data-views. 
        The function also builds the moving mean and moving standard deviation used to normalise the input data.
        """

        self.batch_norm1 = keras.layers.BatchNormalization(name = "batch_norm1_confound")
        self.batch_norm2 = keras.layers.BatchNormalization(name = "batch_norm2_confound")
        self.run=self.add_weight(shape = (), initializer = 'zeros',trainable=False, name = 'confoundlayer_run') ## This variable tracks the number of runs we 
      
        self.moving_conv2 = self.add_weight(name = 'moving_conv2', shape=[input_shape[1][1]+1, input_shape[1][1]+1], initializer='zeros', trainable=False)
        self.moving_convX = self.add_weight(name = 'moving_convX', shape=[input_shape[1][1]+1, input_shape[0][1]], initializer='zeros', trainable=False)
        
    def call(self, inputs, training=None):    
        
        """ We run the call function during model training. This call function starts with an initialisation,
        which uses the tf.init_scope() function, which takes the process out of backpropagation. Note that the 
        function performs differently during training and testing.
        
        """

        input1, input2 = inputs

        # Apply batch normalization to each input, this increases model stability
        bn_input1 = self.batch_norm1(input1, training=training)
        bn_input2 = self.batch_norm2(input2, training=training)
    
        # # # Concatenate the batch-normalized inputs
        inputs = [bn_input1, bn_input2]

        X=inputs[0]
        conv = inputs[1]

        ones = tf.ones((tf.shape(conv)[0], 1))
        conv = tf.concat([ones, conv], axis=1)
      
        self.update_moving_variables([X, conv]) ## update parameters for calculating beta
             
        beta = tf.matmul(tf.linalg.inv(self.moving_conv2+self.diag_offset*tf.eye(conv.shape[1])),self.moving_convX) ## calculate beta for confound regression
        
        X_out = tf.subtract(X,tf.matmul(conv, beta)) ## remove confounds

        return X_out
    
    from keras import ops

    def call(self, inputs, training=False):
        """
        Backend-agnostic Keras 3 version of the call() method.
        Uses keras.ops instead of tf.* so it works with TF / Torch / JAX.
        """
        input1, input2 = inputs

        # Batch norm (Keras layers are already backend-agnostic)
        bn_input1 = self.batch_norm1(input1, training=training)
        bn_input2 = self.batch_norm2(input2, training=training)

        X = bn_input1
        conv = bn_input2

        # Augment conv with a bias column of ones
        batch = ops.shape(conv)[0]
        feat  = ops.shape(conv)[1]
        ones  = ops.ones((batch, 1), dtype=ops.dtype(conv))
        conv_aug = ops.concatenate([ones, conv], axis=1)

        # Update moving variables used for confound regression stats
        # (Make sure update_moving_variables internally uses keras.ops too.)
        self.update_moving_variables([X, conv_aug])

        # Compute beta = (C^T C + λI)^{-1} (C^T X) using stored moving stats
        # Here: self.moving_conv2 ≈ C^T C, self.moving_convX ≈ C^T X
        width = ops.shape(conv_aug)[1]
        eye   = ops.eye(width, dtype=ops.dtype(conv_aug))
        inv   = ops.linalg.inv(self.moving_conv2 + self.diag_offset * eye)
        beta  = ops.matmul(inv, self.moving_convX)

        # Regress out confounds
        X_out = X - ops.matmul(conv_aug, beta)
        return X_out
    
    from keras import ops

    def update_moving_variables(self, inputs):
        """
        update variables used for orthogonalisation
        
        """
        X, conv = inputs

        # momentum = 0.0 on first run, otherwise self.momentum
        mom_dtype = ops.dtype(self.momentum)
        momentum = ops.where(
            ops.equal(self.run, 1),
            ops.asarray(0.0, dtype=mom_dtype),
            self.momentum
        )

        # scale_fact = tot_num / batch_size, cast to X dtype
        x_dtype = ops.dtype(X)
        batch_size = ops.shape(X)[0]
        scale_fact = ops.cast(self.tot_num, x_dtype) / ops.cast(batch_size, x_dtype)

        # Precompute (1 - momentum) with matching dtype
        one_minus_m = ops.asarray(1.0, dtype=mom_dtype) - momentum

        # Update moving stats: moving_conv2 ≈ C^T C, moving_convX ≈ C^T X
        conv_T = ops.transpose(conv, (1, 0))
        new_conv2 = momentum * self.moving_conv2 + scale_fact * one_minus_m * ops.matmul(conv_T, conv)
        new_convX = momentum * self.moving_convX + scale_fact * one_minus_m * ops.matmul(conv_T, X)

        # Assign back to state variables
        self.moving_conv2.assign(new_conv2)
        self.moving_convX.assign(new_convX)

        # Mark that we've run once
        self.run.assign(1)
       
        
    
    def get_config(self):
        """
        Returns the configuration of the custom layer for saving and loading.

        Returns:
        config (dict): A Python dictionary containing the layer configuration.
        """
        config = super().get_config().copy()
        config.update({
            'tot_num': self.tot_num,
            'epsilon': self.epsilon,
            'momentum': self.momentum,
            'diag_offset': self.diag_offset,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a layer instance from its configuration.

        Parameters:
        config (dict): A Python dictionary containing the layer configuration.

        Returns:
        An instance of the layer.
        """
        return cls(**config)