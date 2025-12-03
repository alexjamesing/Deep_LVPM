#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:56:45 2021

@author: ing
"""

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import keras
from keras import saving
from keras import ops


@keras.utils.register_keras_serializable(package='deep_lvpm', name='FactorLayer')
class FactorLayer(keras.layers.Layer):
    
    """This layer should be placed at the end of DLVPM models. The layer 
    generates orthogonal factors that are highly correlated between data-views. 
    
    This layer is constructed of three basic parts. The first set of operations
    involve carrying out batch normalisation on the inputs. In the second set of 
    operations, we orthogonalise inputs with respect to the first DLV.
    We then use a linear layer to project the output of the neural network into a 
    space where it correlates with the outputs of other data-views.

    Similar to some other layers, such as the batch normalisation layer, this
    layer performs differently during training and testing.
    
    Attributes:
        kernel_regularizer (keras.regularizers.Regularizer or None): Regularizer function applied to the projection layer's kernel weights.
        epsilon (float): Small constant added to variance to avoid dividing by zero in the batch normalization step. Defaults to 1e-6.
        momentum (float): Momentum for the moving average and moving variance in the batch normalization step. Defaults to 0.95.
        tot_num (int or None): Total number of samples used for training. This is used for optimal scaling of covariance matrices.
        ndims (int or None): Number of DLVs to extract.
        run (tf.Variable): Tracks the number of runs to initialize moving variables on the first call.
    
    
    Call arguments:
    inputs: A single tensor, which is used for the purposes of projecting to 
    other data-views, identifying factors that are highly correlated between 
    data-views. 
    
    """
    
    
    def __init__(self, kernel_regularizer=None, epsilon=1e-3, momentum=0.99, tot_num=None, ndims=None, **kwargs):
        
        
        """
        Initializes the FactorLayer.

        Args:
            kernel_regularizer (keras.regularizers.Regularizer, optional): Regularizer function applied to the projection layer's kernel weights.
            epsilon (float, optional): Small constant added to variance to avoid dividing by zero in the batch normalization step. Defaults to 1e-6.
            momentum (float, optional): Momentum for the moving average and moving variance in the batch normalization step. Defaults to 0.95.
            tot_num (int, optional): Total number of samples used for training. Used for optimal scaling of covariance matrices.
            ndims (int, optional): Number of DLVPM factor dimensions to extract.
            run (int, optional): Initial value for the run tracker. Defaults to 0.
            **kwargs: Additional keyword arguments inherited from keras.layers.Layer.
        """
        
        super().__init__(**kwargs)

        self.kernel_regularizer = kernel_regularizer ## This kernel regularizer variable determines the degree of regularization that projection weight vectors are subject to
        self.epsilon = epsilon ## This is the offset determined during batch normalisation
        self.momentum = momentum ## This is the amount of momentum that covariance matrices are subject to (see pseudo-code for more details)
        # # Additional custom parameters
        self.tot_num = tot_num #kwargs.get("tot_num") ## This is the total number of samples in the full dataset
        self.ndims = ndims #kwargs.get("ndims") ## This is the total number of factors we wish to extract
      
       
    def build(self, input_shape):
        
        """
        Creates the weights of the layer.

        This function initializes the list of projection vectors, moving mean, moving standard deviation,
        and other variables required for the orthogonalization and normalization processes.

        Args:
            input_shape (tuple): Shape of the input tensor.
        """

        self.batch_norm1 = keras.layers.BatchNormalization(name='batch_norm1_factorlayer', momentum=self.momentum, epsilon=self.epsilon)

        self.run= self.add_weight(name = 'factorlayer_run', shape = (), initializer = 'zeros',trainable=False) ## This variable tracks the number of runs

        self.linear_layer_list = [] ## A list of projection layers
        self.linear_layer_static = [] ## A list containing projection layer weights which are assigned as non-trainable
        
        ## This loop creates n=tot_num projection layers, which are used to construct DLVPM factors 
        for i in range(self.ndims):
            linear_layer = self.add_weight(name = 'projection_weight_' + str(i), shape = [input_shape[1],1], initializer=keras.initializers.RandomNormal(mean=0., stddev=1.), regularizer=self.kernel_regularizer, trainable=True)
            self.linear_layer_list.append(linear_layer)
        
        ## This loop creates n=tot_num static projection layers, which are non-trainable and used in orthogonalisation processes  
        for i in range(self.ndims):
            static_layer = self.add_weight(name = 'static_projection_weight_' + str(i), shape = [input_shape[1],1], initializer=keras.initializers.RandomNormal(mean=0., stddev=1.), trainable=False)
            self.linear_layer_static.append(static_layer)
        
        self.DLV_mean = self.add_weight(name = 'DLV_moving_mean', shape = [self.ndims,1], initializer='zeros', trainable=False) 
        self.DLV_var = self.add_weight(name = 'DLV_moving_std', shape = [self.ndims,1], initializer='ones', trainable=False) 
        
        self.moving_convX = self.add_weight(name = 'moving_convX', shape=[self.ndims, input_shape[1]], initializer='zeros', trainable=False)
        #self.i=tf.Variable(0,trainable=False)

        super(FactorLayer, self).build(input_shape) ## ensures that the layer registers as built
        #self.run=tf.Variable(0,trainable=False)


    def call(self, inputs, training=False):
        """
        Forward pass of the FactorLayer.

        This function applies the projection, batch normalization, orthogonalization, and correlation enhancement steps.
        """
        from keras import ops

        X = self.batch_norm1(inputs, training=training)

        if training:
            DLV_all = self.calculate_batch_DLV_static(X)
            out = self.calculate_batch_DLV_train(X, DLV_all)
            self.update_moving_variables([X, DLV_all])
        else:
            out = self.calculate_DLV_test(X)

        return out


    def weight_normalizer(self, inputs):
        """Re-normalize projection weight vectors; return normalized DLVs."""

        y, scale_fact, train_DLV = inputs

        for i in range(self.ndims):
            yi = y[:, i]
            denom = ops.sqrt(scale_fact * ops.sum(ops.square(yi)))
            self.linear_layer_list[i].assign(self.linear_layer_list[i] / denom)
            self.linear_layer_static[i].assign(self.linear_layer_list[i] / denom)

        y_denom = ops.sqrt(scale_fact * ops.sum(ops.square(y), axis=0))
        out_y = y / y_denom
        return out_y

    def update_moving_variables(self, inputs):

        """Update moving variables using batch-level statistics."""

        X, DLV_all = inputs
        batch_size = ops.cast(ops.shape(X)[0], self.compute_dtype)
        scale_fact = ops.cast(self.tot_num, self.compute_dtype) / batch_size

        # momentum = 0 on first call, else self.momentum
        first = ops.cast(ops.equal(self.run, ops.zeros_like(self.run)), self.compute_dtype)
        momentum = (1.0 - first) * ops.convert_to_tensor(self.momentum, dtype=self.compute_dtype)

        batch_DLV_mean = ops.expand_dims(ops.mean(DLV_all, axis=0), axis=1)   # (ndims,1)
        batch_DLV_var  = ops.expand_dims(ops.var(DLV_all, axis=0),  axis=1)   # (ndims,1)

        one = ops.convert_to_tensor(1.0, dtype=self.compute_dtype)

        self.DLV_mean.assign(momentum * self.DLV_mean + (one - momentum) * batch_DLV_mean)
        self.DLV_var.assign(momentum * self.DLV_var + (one - momentum) * batch_DLV_var)

        batch_DLV_norm = (DLV_all - ops.transpose(batch_DLV_mean)) / (ops.transpose(ops.sqrt(batch_DLV_var)) + self.epsilon)

        self.moving_convX.assign(
            momentum * self.moving_convX
            + scale_fact * (one - momentum) * ops.matmul(ops.transpose(batch_DLV_norm), X)
        )

        # # keep static copies in sync
        # for i in range(self.ndims):
        #     self.linear_layer_static[i].assign(self.linear_layer_list[i])

        self.run.assign(ops.cast(1.0, self.compute_dtype))


    def orthogonalisation_train(self, inputs):
        """Orthogonalize X w.r.t. previous DLVs using batch stats."""

        X, DLV_prev = inputs
        DLV_batch = (DLV_prev - ops.mean(DLV_prev, axis=0)) / (ops.std(DLV_prev, axis=0) + self.epsilon)

        denom = ops.cast(ops.shape(X)[0], self.compute_dtype)
        beta = ops.matmul(ops.transpose(DLV_batch), X) / denom
        ortho_output = X - ops.matmul(DLV_batch, beta)
        return ortho_output


    def orthogonalisation_test(self, inputs):
        """Orthogonalize X w.r.t. previous DLVs using moving variables."""
        from keras import ops

        X, DLV_prev = inputs
        i = DLV_prev.shape[1]

        DLV_norm = (DLV_prev - ops.transpose(self.DLV_mean[:i, :])) / (ops.transpose(ops.sqrt(self.DLV_var)[:i, :]) + self.epsilon)

        denom = self.tot_num
        beta = self.moving_convX[:i, :] / denom
        ortho_output = X - ops.matmul(DLV_norm, beta)
        return ortho_output
    

    def calculate_batch_DLV_static(self, X):
        """Compute batch DLVs with STATIC (non-trainable) projection vectors."""

        for i in range(self.ndims):
            if i == 0:
                DLV = ops.matmul(X, self.linear_layer_static[i])
                DLV_all = DLV
            else:
                ortho_output = self.orthogonalisation_train([X, DLV_all])
                DLV = ops.matmul(ortho_output, self.linear_layer_static[i])
                DLV_all = ops.concatenate([DLV_all, DLV], axis=1)
        return DLV_all
    


    def calculate_batch_DLV_train(self, X, DLV_all):
        """Compute batch DLVs with TRAINABLE projection vectors."""
        from keras import ops

        for i in range(self.ndims):
            if i == 0:
                out = ops.matmul(X, self.linear_layer_list[i])
            else:
                ortho_output = self.orthogonalisation_train([X, DLV_all[:, :i]])
                out_i = ops.matmul(ortho_output, self.linear_layer_list[i])
                out = ops.concatenate([out, out_i], axis=1)
        return out


    

    def calculate_DLV_test(self, X):
        """Compute DLVs at test time using moving statistics."""
        from keras import ops

        for i in range(self.ndims):
            w = self.linear_layer_list[i]
            if i == 0:
                out = ops.matmul(X, w)
            else:
                ortho_output = self.orthogonalisation_test([X, out])
                out_i = ops.matmul(ortho_output, w)
                out = ops.concatenate([out, out_i], axis=1)
        return out
    

    def get_config(self):
        
        base_config = super().get_config()
        
        config={
            'kernel_regularizer':  keras.regularizers.serialize(self.kernel_regularizer),
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'tot_num': self.tot_num,
            'ndims': self.ndims,
            }
        
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        
        config['kernel_regularizer'] = keras.regularizers.deserialize(config['kernel_regularizer'])

        return cls(**config)
    
    def build_from_config(self,config):
         self.build(config["input_shape"])
        
    
