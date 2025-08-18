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
    
    
    def __init__(self, kernel_regularizer=keras.regularizers.l1_l2(l1=0, l2=0), epsilon=1e-3, momentum=0.95, diag_offset=1e-3, tot_num=None, ndims=None, **kwargs):
        
        """
        Initialize the custom layer.

        Parameters:
        kernel_regularizer: Regularizer function for the kernel weights (default: L1L2 regularizer).
        epsilon: Small float added to variance to avoid dividing by zero in batch normalization.
        momentum: Momentum for the moving average in batch normalization.
        diag_offset: Small float added to the diagonal of covariance matrix to ensure it's invertible.
        tot_num: Total number of samples in the full dataset.
        ndims: Total number of factors to extract.
        run: Variable tracking the number of runs.
        """

        super().__init__()

        self.kernel_regularizer = kernel_regularizer ## This kernel regularizer variable determines the degree of regularization that projection weight vectors are subject to
        self.momentum = momentum ## This is the amount of momentum that covariance matrices are subject to (see pseudo-code for more details)
        self.epsilon = epsilon ## This is the offset determined during batch normalisation
        self.diag_offset =diag_offset ## This is a offset added to the diagonal of the covariance matrix between DLVs, to ensure that this matrix is invertable
         # # Additional custom parameters
        self.tot_num = tot_num #kwargs.get("tot_num") ## This is the total number of samples in the full dataset
        self.ndims = ndims #kwargs.get("ndims") ## This is the total number of factors we wish to extract
        


    # def build(self, input_shape):
        
    #     """ In this function, the model builds and assigns values to the weights used in the DLVPM analysis.
    #     The function builds the list of projection vectors used to map associations between different data-views. 
    #     The function also builds the moving mean and moving standard deviation used to normalise the input data.
    #     """
       
    #     self.batch_norm1 = keras.layers.BatchNormalization(momentum=self.momentum,epsilon=self.epsilon)
    #     self.run=self.add_weight(shape = (), initializer = 'zeros',trainable=False, name = 'zcalayer_run') ## This variable tracks the number of runs we 

    #     ## self.project is the weight projection layer, trained to project variables into a space where they are optimally correlated
    #     self.project = self.add_weight(name = 'projection_weight_', shape = [input_shape[1],self.ndims], initializer=keras.initializers.RandomNormal(mean=0., stddev=1.), regularizer=self.kernel_regularizer, trainable=True)
        
    #     ## self.moving_conv2 and self.moving_convX are covaraince matrices used in the orthonalisation process. These matrices are only used in the testing/prediction phase. self.moving_conv2 is a covaraince matrix expressing the covariances between DLVPM factors, elf.moving_convX is a covariance matrix expressing the covariances between DLVPM factors and the last layer of the neural network
    #     self.moving_conv2 = self.add_weight(name = 'moving_conv2', shape=[self.ndims, self.ndims], initializer='zeros', trainable=False)
    #     #init_offset = self.diag_offset*100
    #     self.moving_conv2.assign(tf.eye(num_rows=self.ndims)) ## this variable is initialised under the assumption that DLVPM factors are uncorrelated with one another

    def build(self, input_shape):

        """ In this function, the model builds and assigns values to the weights used in the DLVPM analysis.
        The function builds the list of projection vectors used to map associations between different data-views. 
        The function also builds the moving mean and moving standard deviation used to normalise the input data.
        """
    
        self.batch_norm1 = keras.layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon)
        self.run = self.add_weight(shape=(), initializer="zeros", trainable=False, name="zcalayer_run")

        self.project = self.add_weight(
            name="projection_weight_",
            shape=[input_shape[1], self.ndims],
            initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        self.moving_conv2 = self.add_weight(
            name="moving_conv2", shape=[self.ndims, self.ndims], initializer="zeros", trainable=False
        )
        # was: tf.eye(...)
        self.moving_conv2.assign(ops.eye(self.ndims, self.ndims, dtype=self.compute_dtype))



         
    # @tf.function
    # def call(self, inputs, training=None):    
        
    #     """ We run the call function during model training. This call function starts with an initialisation,
    #     which uses the tf.init_scope() function, which takes the process out of backpropagation. Note that the 
    #     function performs differently during training and testing.
        
    #     """

    #     #tf.print("Running the new version of my custom layer!")
    
    #     X = self.batch_norm1(inputs, training=training)
    
    #     if training:
            
    #         ## The algorithm runs differently in training and testing modes. In the training mode,
    #         ## normalisation and orthogonalisation are carried out using batch-level statistics
        
    #         #self.project_static.assign(self.project)
    #         self.update_moving_variables(X)
    #         out = tf.matmul(X,self.project)


    #     else:

    #         out = tf.matmul(X,self.project)

    #     return out

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

        return out


    # def inv_sqrt_via_cholesky(self, M):
    #     """
    #     Computes a triangular factor that squares to M^{-1} for a positive-definite matrix M.
        
    #     Specifically:
    #         1) L = cholesky(M)
    #         2) L_inv = triangular_solve(L, I)
    #         3) M^{-1} = L_inv^T @ L_inv
    #         4) A valid 'square root' of M^{-1} is L_inv^T (or cholesky(M_inv)), 
    #         which you can multiply by vectors to get M^{-1/2} * x.
            
    #     Returns:
    #         A (upper) triangular factor R such that R^T R = M^{-1}.
    #         i.e. R = L_inv^T and M^{-1} = R^T R.
    #     """
    #     # 1) Cholesky factor: M = L L^T
    #     L = tf.linalg.cholesky(M)  
    #     # 2) Invert L by solving L * X = I
    #     n = tf.shape(M)[0]
    #     I = tf.eye(n, dtype=M.dtype)
    #     L_inv = tf.linalg.triangular_solve(L, I, lower=True)  
    #     # M^{-1} = L_inv^T @ L_inv
        
    #     # If we define R = L_inv^T, then:
    #     #   R^T R = (L_inv^T)^T (L_inv^T) = L_inv L_inv^T = M^{-1}.
    #     # But we typically use R^T R form. If we want R R^T = M^{-1},
    #     # you could return L_inv instead.
        
    #     # We'll return the upper-triangular factor, R = L_inv^T
    #     return tf.transpose(L_inv)

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


    
    # def weight_normalizer(self, inputs):

    #     """ The purpose of this function is to re-normalize weights weight vectors. This 
    #     prevents a collapse to a trivial solution. The inputs here are DLVs for this data view. 
        
    #     """

    #     y = inputs[0]
    #     scale_fact = inputs[1]

    #     diag = self.diag_offset
    #     denom = tf.math.sqrt(tf.math.multiply(scale_fact,tf.math.reduce_sum(tf.math.square(y),axis=0)))
    #     self.project.assign(tf.divide(self.project,denom)) ## Here, we normalize the DLVPM weights

    
    #     # #sqrt_inv_y =tf.where(tf.equal(self.run, 0),tf.linalg.sqrtm(tf.linalg.inv(tf.matmul(tf.transpose(y),y)+100*diag*tf.eye(self.moving_conv2.shape[0]))), tf.linalg.sqrtm(tf.linalg.inv(self.moving_conv2+diag*tf.eye(self.moving_conv2.shape[0])))) ## pseudo inverse called on first batch for improved numeric stability
    #     #sqrt_inv_y =tf.linalg.sqrtm(tf.linalg.inv(self.moving_conv2+diag*tf.eye(self.moving_conv2.shape[0]))) ## pseudo inverse called on first batch for improved numeric stability
    #     #sqrt_inv_y  = self.inv_sqrt_via_cholesky(self.moving_conv2+diag*tf.eye(self.moving_conv2.shape[0]))  # shape = [dim, dim]
    #     sqrt_inv_y  = self.inv_sqrt_via_cholesky(self.moving_conv2+self.decaying_diagonal(self.run, self.moving_conv2.shape[0]))  # shape = [dim, dim]
        
        
    #     #scale_fact*(tf.matmul(tf.transpose(y),y))
    #     #sqrt_inv_y  =  self.inv_sqrt_via_cholesky(scale_fact*(tf.matmul(tf.transpose(y),y))+diag*tf.eye(self.moving_conv2.shape[0]))

    #     out_y = tf.matmul(tf.squeeze(y),sqrt_inv_y) ## Here, we normalize the output DLVs

    #     return out_y

    def weight_normalizer(self, inputs):

        #     """ The purpose of this function is to re-normalize weights weight vectors. This 
        #     prevents a collapse to a trivial solution. The inputs here are DLVs for this data view. 
        #     """
 
        y = inputs[0]
        scale_fact = inputs[1]

        denom = ops.sqrt(scale_fact * ops.sum(ops.square(y), axis=0))
        self.project.assign(self.project / denom)

        sqrt_inv_y = self.inv_sqrt_via_cholesky(
            self.moving_conv2 + self.decaying_diagonal(self.run, self.moving_conv2.shape[0])
        )

        out_y = ops.matmul(ops.squeeze(y), sqrt_inv_y)
        
        return out_y



    # def update_moving_variables(self, X):
        
    #     """ This function is called for every batch the model sees during training. This function
    #     updates the moving variables using batch-level statistics.
        
    #     """


    #     scale_fact = tf.cast(self.tot_num/tf.shape(X)[0],dtype=float)

    #     DLVs = tf.matmul(X, self.project)

    #     #momentum = tf.where(tf.equal(self.run, 0), 0.0, self.momentum) ## initialise inputs on first call
    #     #denom = tf.where(tf.equal(self.run, 0), 0.0, tf.math.sqrt(tf.math.multiply(scale_fact,tf.math.reduce_sum(tf.math.square(DLVs),axis=0))))

    #     #tf.where(tf.equal(self.run, 0), self.project.assign(tf.divide(self.project,tf.math.sqrt(tf.math.multiply(scale_fact,tf.math.reduce_sum(tf.math.square(DLVs),axis=0))))), self.project)
        
    #     momentum = self.momentum

    #     self.moving_conv2.assign(momentum*self.moving_conv2 + scale_fact*(tf.constant(1,dtype=float)-momentum)*(tf.matmul(tf.transpose(DLVs),DLVs)))
       
    #     self.run.assign_add(1.0)

    def update_moving_variables(self, X):

        scale_fact = ops.cast(self.tot_num, self.compute_dtype) / ops.cast(ops.shape(X)[0], self.compute_dtype)
        DLVs = ops.matmul(X, self.project)

        momentum = ops.convert_to_tensor(self.momentum, dtype=self.compute_dtype)
        one = ops.convert_to_tensor(1.0, dtype=self.compute_dtype)

        self.moving_conv2.assign(
            momentum * self.moving_conv2
            + scale_fact * (one - momentum) * ops.matmul(ops.transpose(DLVs), DLVs)
        )

        self.run.assign(self.run + ops.cast(1.0, self.compute_dtype))



    # def decaying_diagonal(self, step, dim, final_eps=1e-4, decay_rate=0.1):
    #     """
    #     Returns a (dim x dim) identity matrix scaled by an epsilon value that decays 
    #     exponentially over 'step'.
        
    #     Parameters:
    #     -----------
    #     step : tf.Tensor or int
    #         Current step or iteration count.
    #     dim : int
    #         The dimension for the identity matrix.
    #     initial_eps : float
    #         The initial (large) epsilon value.
    #     final_eps : float
    #         The final (small) epsilon value after many steps.
    #     decay_rate : float
    #         Exponential decay rate.

    #     Returns:
    #     --------
    #     tf.Tensor
    #         A (dim x dim) diagonal matrix with exponentially decaying scale.
    #     """
    #     # Convert step to float in case it's an integer or scalar tensor
    #     step = tf.cast(step, tf.float32)-tf.cast(1, tf.float32)

    #     tf.print()

    #     initial_eps = self.diag_offset

    #     # Exponential decay from initial_eps down to final_eps
    #     current_eps = final_eps + (initial_eps - final_eps) * tf.exp(-decay_rate * step)

    #     #tf.print(current_eps)

    #     # Return scaled identity matrix
    #     return current_eps * tf.eye(dim, dtype=tf.float32)

    def decaying_diagonal(self, step, dim, final_eps=1e-4, decay_rate=0.1):

        """
         Returns a (dim x dim) identity matrix scaled by an epsilon value that decays 
         exponentially over 'step'.

        """
        step = ops.cast(step, "float32") - ops.cast(1.0, "float32")

        initial_eps = ops.convert_to_tensor(self.diag_offset, dtype="float32")
        final_eps = ops.convert_to_tensor(final_eps, dtype="float32")
        current_eps = final_eps + (initial_eps - final_eps) * ops.exp(-decay_rate * step)

        return ops.cast(current_eps, self.compute_dtype) * ops.eye(dim, dim, dtype=self.compute_dtype)     

        
    def get_config(self):
        """
        Returns the configuration of the custom layer for saving and loading.

        Returns:
        config (dict): A Python dictionary containing the layer configuration.
        """
        config = super().get_config().copy()
        config.update({
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'epsilon': self.epsilon,
            'momentum': self.momentum,
            'diag_offset': self.diag_offset,
            'tot_num': self.tot_num,
            'ndims': self.ndims,
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
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Multi-backend Keras 3 version of ZCALayer (uses keras.ops)

# This layer (to be placed at the end of each measurement model) projects inputs
# to DLVs and maintains running covariance statistics for orthogonalization that
# is performed outside (in the StructuralModel). It behaves differently in train
# vs. inference mode (batch norm + moving covariances).
# """

# import numpy as np
# import pydot  # not used here, but kept for symmetry with other files if desired

# import keras
# from keras import ops
# from keras import backend as K
# from keras.layers import Layer, BatchNormalization
# from keras.initializers import RandomNormal, Zeros
# from keras.regularizers import l1_l2, serialize as serialize_reg, deserialize as deserialize_reg
# from keras.saving import register_keras_serializable


# @register_keras_serializable(package="deep_lvpm", name="ZCALayer")
# class ZCALayer(Layer):
#     """
#     ZCA-style projection layer for DLVPM models.

#     Parameters
#     ----------
#     kernel_regularizer : keras.regularizers.Regularizer
#         Regularizer applied to the projection weights.
#     epsilon : float
#         Small constant for numerical stability (used in norms/stds).
#     momentum : float
#         Exponential moving average momentum for running covariances.
#     diag_offset : float
#         Initial diagonal stabilization magnitude for covariance inverse-sqrt.
#     tot_num : int
#         Total dataset size (used to scale batch covariances).
#     ndims : int
#         Number of DLVs (projection output dims).
#     """

#     def __init__(
#         self,
#         kernel_regularizer=l1_l2(l1=0.0, l2=0.0),
#         epsilon=1e-3,
#         momentum=0.95,
#         diag_offset=1e-3,
#         tot_num=None,
#         ndims=None,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.kernel_regularizer = kernel_regularizer
#         self.momentum = float(momentum)
#         self.epsilon = float(epsilon)
#         self.diag_offset = float(diag_offset)
#         self.tot_num = None if tot_num is None else int(tot_num)
#         if ndims is None:
#             raise ValueError("ZCALayer requires ndims (number of DLVs).")
#         self.ndims = int(ndims)

#         # Sub-layers
#         self.batch_norm1 = BatchNormalization(momentum=self.momentum, epsilon=self.epsilon)

#         # State tensors — created in build()
#         self.run = None
#         self.project = None
#         self.moving_conv2 = None

#     # ------------- helpers (backend-agnostic via keras.ops) -------------

#     def _inv_sqrt_spd(self, M):
#         """
#         Compute M^{-1/2} for a symmetric positive-definite matrix M using eigendecomposition:
#             M = V diag(λ) V^T  ->  M^{-1/2} = V diag(λ^{-1/2}) V^T
#         This path is supported across TF / Torch / JAX via keras.ops.
#         """
#         # Symmetrize just in case of tiny asymmetry
#         M_sym = 0.5 * (M + ops.transpose(M))
#         # Stabilize spectrum
#         eps = ops.convert_to_tensor(self.epsilon, dtype=ops.dtype(M_sym))
#         # eigh returns (eigvals, eigvecs) in ascending order
#         eigvals, eigvecs = ops.linalg.eigh(M_sym)
#         eigvals = ops.maximum(eigvals, eps)
#         inv_sqrt_vals = 1.0 / ops.sqrt(eigvals)
#         # Scale columns of eigvecs by inv_sqrt_vals, then V * diag(inv_sqrt) * V^T
#         V_scaled = eigvecs * ops.expand_dims(inv_sqrt_vals, axis=0)
#         return ops.matmul(V_scaled, ops.transpose(eigvecs))

#     def _eye(self, n, dtype):
#         # Prefer ops.eye if available; fall back to NumPy constant.
#         try:
#             return ops.eye(n, n, dtype=dtype)
#         except Exception:
#             return ops.convert_to_tensor(np.eye(n, dtype=keras.dtypes.serialize(dtype)))

#     def _decaying_diagonal(self, step, dim, final_eps=1e-4, decay_rate=0.1):
#         """
#         Returns eps(step) * I, where eps decays exponentially from diag_offset to final_eps.
#         """
#         step = ops.cast(step, "float32") - ops.cast(1.0, "float32")
#         final_eps = ops.convert_to_tensor(final_eps, dtype="float32")
#         init_eps = ops.convert_to_tensor(self.diag_offset, dtype="float32")
#         current_eps = final_eps + (init_eps - final_eps) * ops.exp(-decay_rate * step)
#         return ops.cast(current_eps, self.compute_dtype) * self._eye(dim, self.compute_dtype)

#     # ---------------------------- Keras plumbing ----------------------------

#     def build(self, input_shape):
#         if not isinstance(input_shape, (tuple, list)):
#             in_dim = int(input_shape[-1])
#         else:
#             in_dim = int(input_shape[0][-1])

#         # Tracks number of updates (batches seen)
#         self.run = self.add_weight(
#             name="zcalayer_run",
#             shape=(),
#             initializer=Zeros(),
#             trainable=False,
#             dtype=self.compute_dtype,
#         )

#         # Projection matrix W: (in_dim x ndims)
#         self.project = self.add_weight(
#             name="projection_weight_",
#             shape=(in_dim, self.ndims),
#             initializer=RandomNormal(mean=0.0, stddev=1.0),
#             regularizer=self.kernel_regularizer,
#             trainable=True,
#             dtype=self.compute_dtype,
#         )

#         # Running covariance of DLVs (ndims x ndims) — start as identity
#         self.moving_conv2 = self.add_weight(
#             name="moving_conv2",
#             shape=(self.ndims, self.ndims),
#             initializer=Zeros(),
#             trainable=False,
#             dtype=self.compute_dtype,
#         )
#         # Initialize to identity (orthogonal initial assumption)
#         self.moving_conv2.assign(self._eye(self.ndims, self.compute_dtype))

#         super().build(input_shape)

#     # --------------------------------- call ---------------------------------

#     def call(self, inputs, training=None):
#         """
#         Training: BN on inputs, update moving covariance from current batch, then project.
#         Inference: BN on inputs, then project (uses running stats collected during training).
#         """
#         X = self.batch_norm1(inputs, training=training)

#         if training:
#             self._update_moving_variables(X)

#         out = ops.matmul(X, self.project)  # (batch, ndims)
#         return out

#     # ----------------------- public utility (used externally) ----------------

#     def weight_normalizer(self, inputs):
#         """
#         Renormalize projection weights and return normalized DLVs for this view.

#         inputs: [y, scale_fact]
#           y           : (batch, ndims)  current DLVs for this view
#           scale_fact  : scalar = tot_num / batch_size
#         """
#         y, scale_fact = inputs
#         y = ops.convert_to_tensor(y)
#         scale_fact = ops.cast(scale_fact, ops.dtype(y))

#         # Normalize columns of W by sqrt(scale_fact * ||y||^2) to avoid collapse
#         denom = ops.sqrt(scale_fact * (ops.sum(ops.square(y), axis=0) + self.epsilon))  # (ndims,)
#         self.project.assign(self.project / denom)  # broadcast over columns

#         # Compute inverse sqrt of running covariance with a small, decaying diagonal stabilizer
#         R = self._inv_sqrt_spd(self.moving_conv2 + self._decaying_diagonal(self.run, self.ndims))
#         out_y = ops.matmul(y, R)  # (batch, ndims)
#         return out_y

#     # ------------------------------ internals ------------------------------

#     def _update_moving_variables(self, X):
#         """
#         Update running covariance of DLVs using batch statistics.
#         """
#         # Scale factor = tot_num / batch_size
#         if self.tot_num is None:
#             raise ValueError("ZCALayer.update_moving_variables requires tot_num be set.")
#         bsz = ops.cast(ops.shape(X)[0], self.compute_dtype)
#         scale_fact = ops.cast(self.tot_num, self.compute_dtype) / bsz

#         DLVs = ops.matmul(X, self.project)  # (batch, ndims)

#         momentum = ops.convert_to_tensor(self.momentum, dtype=self.compute_dtype)
#         one = ops.convert_to_tensor(1.0, dtype=self.compute_dtype)

#         # Cov update: m <- m * momentum + (1 - momentum) * scale_fact * (Y^T Y)
#         yt_y = ops.matmul(ops.transpose(DLVs), DLVs)  # (ndims, ndims)
#         updated = momentum * self.moving_conv2 + (one - momentum) * scale_fact * yt_y
#         self.moving_conv2.assign(updated)

#         # Step counter
#         self.run.assign(self.run + ops.cast(1.0, self.compute_dtype))

#     # ----------------------------- serialization -----------------------------

#     def get_config(self):
#         config = super().get_config().copy()
#         config.update(
#             {
#                 "kernel_regularizer": serialize_reg(self.kernel_regularizer),
#                 "epsilon": self.epsilon,
#                 "momentum": self.momentum,
#                 "diag_offset": self.diag_offset,
#                 "tot_num": self.tot_num,
#                 "ndims": self.ndims,
#             }
#         )
#         return config

#     @classmethod
#     def from_config(cls, config):
#         config["kernel_regularizer"] = deserialize_reg(config["kernel_regularizer"])
#         return cls(**config)
    
    
    
    
    