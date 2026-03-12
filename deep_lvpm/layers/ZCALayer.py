#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 14:53:57 2022

@author: ing
"""

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import keras
from keras import saving
from keras import ops


@keras.utils.register_keras_serializable(package="deep_lvpm", name="SoftThreshold")
class SoftThreshold(keras.constraints.Constraint):
    """
    Applies the Soft Thresholding operator (Proximal operator for L1 norm).
    This forces weights with magnitude < threshold to become exactly zero.
    """
    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def __call__(self, w):
        # Using keras.ops for backend-agnostic operations
        return ops.sign(w) * ops.maximum(ops.abs(w) - self.threshold, 0.0)

    def get_config(self):
        return {'threshold': self.threshold}


@keras.utils.register_keras_serializable(package="deep_lvpm",name="ZCALayer")
class ZCALayer(keras.layers.Layer):
    
    """This layer should be placed at the end of DLVPM models. The layer 
    generates orthogonal factors that are highly correlated between data-views. 
    
    This layer is constructed of two basic parts. The first set of operations
    involve carrying out batch normalisation on the inputs. We then use a linear layer to 
    project the output of the neural network into a space where it correlates 
    with the outputs of other data-views. In contrast to the FactorLayer, orthogonalisation
    is carried out outside of this layer, as part of the StructuralModel class. 
    This is much more convinient in this case.
    
    The ordering of the layer calculations is: batch normalisation > 
    > linear projection. 
    
    Similar to some other layers, such as the batch normalisation layer, this
    layer performs differently during training and testing.
    
    Args:
        
    kernel regulariser: this parameter determines the amount of regularisation 
    applied to the projection layers
        
    momentum: a single value that should be greater than zero but less than one.
    momentum is used to ascribe global mean and variance normalisation values during
    the initial batch normalisation step, and the values of covariance matrices 
    during their update. Default value is momentum = 0.95.
    
    epsilon: This is the offset value used during the initial batch normalisation
    step, which ensures stability. Default value is set to 1e-6.
    
    tot_num: This is the total number of samples that training is carried out over. 
    This value is used to ensure that covariance matrices are optimally scaled.
    
    ndims: parameter that defines the number of DLVPM factor dimensions 
    we wish to extract
    
    
    Call arguments:
    inputs: A single tensor, which is used for the purposes of projecting to 
    other data-views, identifying factors that are highly correlated between 
    data-views. 
    
    """
    
    
    def __init__(self, 
                 kernel_regularizer=keras.regularizers.l1_l2(l1=0, l2=0), 
                 epsilon=1e-3, 
                 momentum=0.95, 
                 diag_offset=1e-3, 
                 tot_num=None, 
                 ndims=None, 
                 sparse_l1=0.0,
                 **kwargs):
        
        super().__init__(**kwargs)

        self.kernel_regularizer = kernel_regularizer 
        self.momentum = momentum 
        self.epsilon = epsilon 
        self.diag_offset = diag_offset 
        self.tot_num = tot_num 
        self.ndims = ndims 
        self.sparse_l1 = float(sparse_l1)


    def build(self, input_shape):
        
        self.batch_norm1 = keras.layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon)
        self.run = self.add_weight(shape=(), initializer="zeros", trainable=False, name="zcalayer_run")

        # Determine if we should apply the proximal constraint
        proj_constraint = None
        if self.sparse_l1 > 0.0:
            proj_constraint = SoftThreshold(self.sparse_l1)

        self.project = self.add_weight(
            name="projection_weight_",
            shape=[input_shape[1], self.ndims],
            initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            regularizer=self.kernel_regularizer,
            constraint=proj_constraint,  # <--- Apply constraint here
            trainable=True,
        )

        self.moving_conv2 = self.add_weight(
            name="moving_conv2", shape=[self.ndims, self.ndims], initializer="zeros", trainable=False
        )
        self.moving_conv2.assign(ops.eye(self.ndims, self.ndims, dtype=self.compute_dtype))

    def call(self, inputs, training=None):

        """ We run the call function during model training. This call function starts with an initialisation,
        which uses the tf.init_scope() function, which takes the process out of backpropagation. Note that the 
        function performs differently during training and testing.

        """

        X = self.batch_norm1(inputs, training=training)

        if training:
            self.update_moving_variables(X)       
            out = ops.matmul(X, self.project)
        else:  
            out = ops.matmul(X, self.project)
            # out = ops.matmul(out, self.inv_sqrt_via_cholesky(self.moving_conv2))
       

        return out


    def inv_sqrt_via_cholesky(self, M):
        
        """
        Backend-agnostic M^{-1/2} via eigendecomposition:
        M = V diag(lam) V^T  ->  M^{-1/2} = V diag(lam^{-1/2}) V^T
        """
        M_sym = 0.5 * (M + ops.transpose(M))
        eps = ops.convert_to_tensor(self.epsilon, dtype=ops.dtype(M_sym))
        eigvals, eigvecs = ops.linalg.eigh(M_sym)
        eigvals = ops.maximum(eigvals, eps)
        inv_sqrt_vals = 1.0 / ops.sqrt(eigvals)
        V_scaled = eigvecs * ops.expand_dims(inv_sqrt_vals, axis=0)

        return ops.matmul(V_scaled, ops.transpose(eigvecs))
    

   
    def weight_normalizer(self, inputs):

        #     """ The purpose of this function is to re-normalize weights weight vectors. This 
        #     prevents a collapse to a trivial solution. The inputs here are DLVs for this data view. 
        #     """
 
        y, scale_fact, train_DLV = inputs


        eps = ops.convert_to_tensor(self.epsilon, dtype=ops.dtype(y))
        denom = ops.sqrt(scale_fact * ops.sum(ops.square(y), axis=0) + eps)
        self.project.assign(self.project / denom)
        y = y / denom

        if train_DLV == False: # if train_DLV is False, we use moving averages

            sqrt_inv_y = self.inv_sqrt_via_cholesky(
                self.moving_conv2 + self.diag_offset*ops.eye(self.moving_conv2.shape[0], self.moving_conv2.shape[0], dtype=self.compute_dtype)   
            )

        else: # if train_DLV is True (default), we use batch level statistics

            sqrt_inv_y = self.inv_sqrt_via_cholesky(
                scale_fact*ops.matmul(ops.transpose(y), y) + self.diag_offset*ops.eye(self.moving_conv2.shape[0], self.moving_conv2.shape[0], dtype=self.compute_dtype)   
            )

        
        out_y = ops.matmul(ops.squeeze(y), sqrt_inv_y)

        return out_y



    def update_moving_variables(self, X):

        scale_fact = ops.cast(self.tot_num, self.compute_dtype) / ops.cast(ops.shape(X)[0], self.compute_dtype)
        y = ops.matmul(X, self.project)

        momentum = ops.convert_to_tensor(self.momentum, dtype=self.compute_dtype)
        one = ops.convert_to_tensor(1.0, dtype=self.compute_dtype)

        self.moving_conv2.assign(
            momentum * self.moving_conv2
            + scale_fact * (one - momentum) * ops.matmul(ops.transpose(y), y)
        )

        self.run.assign(self.run + ops.cast(1.0, self.compute_dtype))

        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'epsilon': self.epsilon,
            'momentum': self.momentum,
            'diag_offset': self.diag_offset,
            'tot_num': self.tot_num,
            'ndims': self.ndims,
            'sparse_l1': self.sparse_l1,
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
        config['kernel_regularizer'] = keras.regularizers.deserialize(config['kernel_regularizer'])
        return cls(**config)
   
