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
    
    
    def __init__(self, kernel_regularizer=None, epsilon=1e-3, momentum=0.99, tot_num=None, ndims=None, run=0, **kwargs):
        
        
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
        #self.first_run =tf.Variable(True,trainable=False)
       
    def build(self, input_shape):
        
        """
        Creates the weights of the layer.

        This function initializes the list of projection vectors, moving mean, moving standard deviation,
        and other variables required for the orthogonalization and normalization processes.

        Args:
            input_shape (tuple): Shape of the input tensor.
        """

        self.batch_norm1 = keras.layers.BatchNormalization(name='batch_norm1_factorlayer', momentum=self.momentum, epsilon=self.epsilon)

        self.run= self.add_weight(name = 'factorlayer_run', shape = (), initializer = 'zeros',trainable=False) ## This variable tracks the number of runs we 

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

       
         
    #@tf.function
    # def call(self, inputs, training=False):    
        
    #     """
    #     Forward pass of the FactorLayer.

    #     This function applies the projection, batch normalization, orthogonalization, and correlation enhancement steps.

    #     Args:
    #         inputs (tf.Tensor): Input tensor.
    #         training (bool, optional): Indicator for training or inference mode. Defaults to False. The layer performs differently 
    #         during training and testing.

    #     Returns:
    #         tf.Tensor: The output tensor after applying the transformations.
    #     """

    #     X = self.batch_norm1(inputs, training=training)

    #     if training:
            
    #         ## The algorithm runs differently in training and testing modes. In the training mode,
    #         ## normalisation and orthogonalisation are carried out using batch-level statistics

    #         DLV_all = self.calculate_batch_DLV_static(X)

    #         out = self.calculate_batch_DLV_train(X, DLV_all) 

    #         self.update_moving_variables([X, DLV_all])
            
        
    #     else:
            
    #         out = self.calculate_DLV_test(X) ## Here, we calculate DLVs during testing, using population level statistics

    #     return out

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

   
    # def weight_normalizer(self, inputs):

    #     """ The purpose of this function is to re-normalize weights weight vectors. This 
    #     prevents a collapse to a trivial solution. The inputs here are DLVs for this data view. 
        
    #     """

    #     y = inputs[0]
    #     scale_fact = inputs[1]

    #     for i in range(self.ndims): # Here, we loop through weight projection vectors
            
    #         denom = tf.math.sqrt(tf.math.multiply(scale_fact,tf.math.reduce_sum(tf.math.square(y[:,i]))))

    #         self.linear_layer_list[i].assign(tf.divide(self.linear_layer_list[i],denom))
    #         self.linear_layer_static[i].assign(tf.divide(self.linear_layer_static[i],denom))

    #     y_denom = tf.math.sqrt(tf.math.multiply(scale_fact,tf.math.reduce_sum(tf.math.square(y),axis=0)))
    #     out_y = tf.divide(y, y_denom)

    #     return out_y

    def weight_normalizer(self, inputs):
        """Re-normalize projection weight vectors; return normalized DLVs."""

        y, scale_fact = inputs

        for i in range(self.ndims):
            yi = y[:, i]
            denom = ops.sqrt(scale_fact * ops.sum(ops.square(yi)))
            self.linear_layer_list[i].assign(self.linear_layer_list[i] / denom)
            self.linear_layer_static[i].assign(self.linear_layer_static[i] / denom)

        y_denom = ops.sqrt(scale_fact * ops.sum(ops.square(y), axis=0))
        out_y = y / y_denom
        return out_y


    # def update_moving_variables(self, inputs):
        
    #     """ This function is called for every batch the model sees during training. This function
    #     updates the moving variables using batch-level statistics.
        
    #     """

    #     momentum = tf.where(tf.equal(self.run, 0), 0.0, self.momentum) ## initialise inputs on first call
    #     scale_fact = tf.cast(self.tot_num/tf.shape(inputs[0])[0],dtype=float)

    #     batch_DLV_mean = tf.expand_dims(tf.math.reduce_mean(inputs[1], axis=0),axis=1)
    #     batch_DLV_var = tf.expand_dims(tf.math.reduce_variance(inputs[1], axis=0),axis=1)

    #     self.DLV_mean.assign(momentum*self.DLV_mean + (tf.constant(1,dtype=float)-momentum)*batch_DLV_mean)
    #     self.DLV_var.assign(momentum*self.DLV_var + (tf.constant(1,dtype=float)-momentum)*batch_DLV_var)

    #     batch_DLV_norm = tf.divide(tf.subtract(inputs[1],tf.transpose(batch_DLV_mean)),tf.transpose(tf.math.sqrt(batch_DLV_var)))

    #     self.moving_convX.assign(momentum*self.moving_convX + scale_fact*(tf.constant(1,dtype=float)-momentum)*tf.matmul(tf.transpose(batch_DLV_norm),inputs[0]))
        
    #     for i in range(self.ndims): # Here, we loop through weight projection vectors 
    #         self.linear_layer_static[i].assign(self.linear_layer_list[i])

    #     self.run.assign(1)

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

        # keep static copies in sync
        for i in range(self.ndims):
            self.linear_layer_static[i].assign(self.linear_layer_list[i])

        self.run.assign(ops.cast(1.0, self.compute_dtype))

        
    # def orthogonalisation_train(self, inputs):
        
    #     """ This function is called multiple times during model training. The purpose of this function is to 
    #     orthogonalise the data with respect to previous DLVs using batch-level statistics

    #     """
        
    #     #X = tf.divide(tf.subtract(inputs[0], tf.math.reduce_mean(inputs[0], axis=0)),tf.math.reduce_std(inputs[0], axis=0)+self.epsilon) 
    #     DLV_batch = tf.divide(tf.subtract(inputs[1], tf.math.reduce_mean(inputs[1], axis=0)),tf.math.reduce_std(inputs[1], axis=0)+self.epsilon) ## Here, we z-normalise the input features to have mean of zero and standard deviation of one 
        
    #     denom = tf.cast(tf.shape(inputs[0])[0],dtype=float)
    #     beta = tf.matmul(tf.transpose(DLV_batch),inputs[0])/denom
    #     ortho_output = tf.subtract(inputs[0],tf.matmul(DLV_batch, beta)) ## This is the input matrix, orthogonalised with respect to previous DLVs
        
    #     return ortho_output 

    def orthogonalisation_train(self, inputs):
        """Orthogonalize X w.r.t. previous DLVs using batch stats."""

        X, DLV_prev = inputs
        DLV_batch = (DLV_prev - ops.mean(DLV_prev, axis=0)) / (ops.std(DLV_prev, axis=0) + self.epsilon)

        denom = ops.cast(ops.shape(X)[0], self.compute_dtype)
        beta = ops.matmul(ops.transpose(DLV_batch), X) / denom
        ortho_output = X - ops.matmul(DLV_batch, beta)
        return ortho_output
    
    # def orthogonalisation_test(self, inputs):
        
    #     """ This function is called during model testing. This function orthogonalises the data with 
    #     respect to previous DLVs, using moving variables.

    #     """
        
    #     i = inputs[1].shape[1]

    #     DLV_norm = tf.divide(tf.subtract(inputs[1],tf.transpose(self.DLV_mean[:i,:])),(tf.transpose(tf.math.sqrt(self.DLV_var)[:i,:])+self.epsilon))

    #     denom = self.tot_num
    #     beta = self.moving_convX[:i,:]/denom
    #     #beta = self.moving_convX[:i,:]

    #     ortho_output = tf.subtract(inputs[0],tf.matmul(DLV_norm, beta))

    #     return ortho_output 

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
        

    # def calculate_batch_DLV_static(self, X):
        
    #     """ This function is used to calculate DLVs at the batch level. These batch level DLVs 
    #     can then be used for orthogonalisation in training. Note that this is done using static 
    #     layers. This means that the backprop algorithm does not see this.

    #     """
        
    #     for i in range(self.ndims):
    #         if i == 0:
    #             DLV=tf.matmul(X,self.linear_layer_static[i])
    #             DLV_all=DLV
    #         else:
    #             ortho_output=self.orthogonalisation_train([X, DLV_all])
    #             DLV=tf.matmul(ortho_output,self.linear_layer_static[i])
    #             DLV_all=tf.concat([DLV_all,DLV],axis=1)
                
    #     return DLV_all

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
    

    # def calculate_batch_DLV_train(self, X, DLV_all):
        
    #     """ This function is used to calculate DLVs at the batch level. These batch level DLVs 
    #     can then be used for orthogonalisation in training. Note that this is done using the training 
    #     layers, so backprop does see the projections here

    #     """
        
    #     for i in range(self.ndims):
    #         if i == 0:
    #             out=tf.matmul(X,self.linear_layer_list[i])
    #         else:
    #             ortho_output=self.orthogonalisation_train([X, DLV_all[:,:i]])
    #             out_i=tf.matmul(ortho_output,self.linear_layer_list[i])
    #             out=tf.concat([out,out_i],axis=1)
                
    #     return out

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


    

    # def calculate_DLV_test(self, X):

    #     """ This function is used to calculate DLVs at test time. This function uses the moving 
    #     variables assigned as class attributes
    #     """

    #     for i in range(self.ndims):
                          
    #         linear_layer = self.linear_layer_list[i] ## select weight projection vector
            
    #         if i == 0: 
    #             out = tf.matmul(X,linear_layer)
    #         else:
    #             ortho_output=self.orthogonalisation_test([X, out])
    #             out_i = tf.matmul(ortho_output,linear_layer)
    #             out=tf.concat([out,out_i],axis=1)
    
    #     return out

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
        
    


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Multi-backend Keras 3 version of FactorLayer (uses keras.ops)

# Place this layer at the end of each measurement model. It:
# 1) batch-normalizes inputs,
# 2) constructs DLVs iteratively with orthogonalization (train/test modes differ),
# 3) maintains running stats for test-time orthogonalization.
# """

# import keras
# from keras import ops
# from keras.layers import Layer, BatchNormalization
# from keras.initializers import RandomNormal, Zeros
# from keras.regularizers import serialize as serialize_reg, deserialize as deserialize_reg
# from keras.saving import register_keras_serializable


# @register_keras_serializable(package="deep_lvpm", name="FactorLayer")
# class FactorLayer(Layer):
#     """
#     Parameters
#     ----------
#     kernel_regularizer : keras.regularizers.Regularizer or None
#     epsilon : float
#         Small constant for numerical stability.
#     momentum : float
#         EMA momentum for running stats.
#     tot_num : int
#         Total dataset size (used to scale batch covariances).
#     ndims : int
#         Number of DLVs to extract (iterative factors).
#     """

#     def __init__(
#         self,
#         kernel_regularizer=None,
#         epsilon=1e-3,
#         momentum=0.99,
#         tot_num=None,
#         ndims=None,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         if ndims is None:
#             raise ValueError("FactorLayer requires ndims (number of factors).")

#         self.kernel_regularizer = kernel_regularizer
#         self.epsilon = float(epsilon)
#         self.momentum = float(momentum)
#         self.tot_num = None if tot_num is None else int(tot_num)
#         self.ndims = int(ndims)

#         # Sub-layers
#         self.batch_norm1 = BatchNormalization(
#             name="batch_norm1_factorlayer",
#             momentum=self.momentum,
#             epsilon=self.epsilon,
#         )

#         # State — created in build()
#         self.run = None
#         self.linear_layer_list = None          # list of trainable (in_dim x 1)
#         self.linear_layer_static = None        # list of non-trainable (in_dim x 1)
#         self.DLV_mean = None                   # (ndims, 1)
#         self.DLV_var = None                    # (ndims, 1)
#         self.moving_convX = None               # (ndims, in_dim)

#     # ---------------------------- Keras plumbing ----------------------------

#     def build(self, input_shape):
#         in_dim = int(input_shape[-1]) if not isinstance(input_shape, (tuple, list)) else int(input_shape[0][-1])

#         # Tracks whether we've seen a batch (0 on first call → special momentum handling)
#         self.run = self.add_weight(
#             name="factorlayer_run",
#             shape=(),
#             initializer=Zeros(),
#             trainable=False,
#             dtype=self.compute_dtype,
#         )

#         # Trainable projection vectors (one per factor): shape (in_dim, 1)
#         self.linear_layer_list = []
#         for i in range(self.ndims):
#             w = self.add_weight(
#                 name=f"projection_weight_{i}",
#                 shape=(in_dim, 1),
#                 initializer=RandomNormal(mean=0.0, stddev=1.0),
#                 regularizer=self.kernel_regularizer,
#                 trainable=True,
#                 dtype=self.compute_dtype,
#             )
#             self.linear_layer_list.append(w)

#         # Static copies for orthogonalization during the batch factor construction
#         self.linear_layer_static = []
#         for i in range(self.ndims):
#             w_stat = self.add_weight(
#                 name=f"static_projection_weight_{i}",
#                 shape=(in_dim, 1),
#                 initializer=RandomNormal(mean=0.0, stddev=1.0),
#                 trainable=False,
#                 dtype=self.compute_dtype,
#             )
#             self.linear_layer_static.append(w_stat)

#         # Moving stats of DLVs
#         self.DLV_mean = self.add_weight(
#             name="DLV_moving_mean",
#             shape=(self.ndims, 1),
#             initializer=Zeros(),
#             trainable=False,
#             dtype=self.compute_dtype,
#         )
#         self.DLV_var = self.add_weight(
#             name="DLV_moving_std",
#             shape=(self.ndims, 1),
#             initializer=keras.initializers.Ones(),
#             trainable=False,
#             dtype=self.compute_dtype,
#         )

#         # Running cross-covariance between (normalized) DLVs and inputs
#         self.moving_convX = self.add_weight(
#             name="moving_convX",
#             shape=(self.ndims, in_dim),
#             initializer=Zeros(),
#             trainable=False,
#             dtype=self.compute_dtype,
#         )

#         super().build(input_shape)

#     # --------------------------------- call ---------------------------------

#     def call(self, inputs, training=False):
#         """
#         Train:  BN -> build batch DLVs (static) -> compute train DLVs (grad flows) -> update moving stats.
#         Test:   BN -> compute DLVs using moving stats.
#         """
#         X = self.batch_norm1(inputs, training=training)

#         if training:
#             DLV_all_static = self._calculate_batch_DLV_static(X)      # (batch, ndims); no grad through static weights
#             out = self._calculate_batch_DLV_train(X, DLV_all_static)  # (batch, ndims); grad through trainable weights
#             self._update_moving_variables(X, DLV_all_static)
#         else:
#             out = self._calculate_DLV_test(X)

#         return out

#     # ----------------------- public utility (used externally) ----------------

#     def weight_normalizer(self, inputs):
#         """
#         Re-normalize projection vectors to avoid collapse and return normalized DLVs.

#         inputs: [y, scale_fact]
#             y          : (batch, ndims)   current DLVs for this view
#             scale_fact : scalar = tot_num / batch_size
#         """
#         y, scale_fact = inputs
#         y = ops.convert_to_tensor(y, dtype=self.compute_dtype)
#         scale_fact = ops.cast(scale_fact, self.compute_dtype)

#         # Normalize each projection vector by sqrt(scale * ||y_i||^2)
#         for i in range(self.ndims):
#             yi = y[:, i]  # (batch,)
#             denom = ops.sqrt(scale_fact * (ops.sum(ops.square(yi)) + self.epsilon))
#             self.linear_layer_list[i].assign(self.linear_layer_list[i] / denom)
#             self.linear_layer_static[i].assign(self.linear_layer_static[i] / denom)

#         # Also return normalized y for downstream usage
#         y_denom = ops.sqrt(scale_fact * (ops.sum(ops.square(y), axis=0) + self.epsilon))  # (ndims,)
#         out_y = y / y_denom
#         return out_y

#     # ------------------------------ internals ------------------------------

#     def _update_moving_variables(self, X, DLV_all):
#         """
#         Update moving mean/var of DLVs and running cross-covariance with inputs.
#         """
#         if self.tot_num is None:
#             raise ValueError("FactorLayer.update_moving_variables requires tot_num be set.")

#         # momentum = 0 on first call, else self.momentum
#         is_first = ops.cast(ops.equal(self.run, ops.zeros_like(self.run)), self.compute_dtype)
#         momentum = (1.0 - is_first) * ops.convert_to_tensor(self.momentum, dtype=self.compute_dtype)

#         batch_size = ops.cast(ops.shape(X)[0], self.compute_dtype)
#         scale_fact = ops.cast(self.tot_num, self.compute_dtype) / batch_size

#         # Batch moments over batch axis (axis=0)
#         batch_mean = ops.expand_dims(ops.mean(DLV_all, axis=0), axis=1)            # (ndims, 1)
#         batch_var = ops.expand_dims(ops.var(DLV_all, axis=0), axis=1)              # (ndims, 1)

#         one = ops.convert_to_tensor(1.0, dtype=self.compute_dtype)

#         self.DLV_mean.assign(momentum * self.DLV_mean + (one - momentum) * batch_mean)
#         self.DLV_var.assign(momentum * self.DLV_var + (one - momentum) * batch_var)

#         # Normalize batch DLVs with batch stats for cross-cov update
#         DLV_norm = (DLV_all - ops.transpose(batch_mean)) / (ops.transpose(ops.sqrt(batch_var) + self.epsilon))
#         # Update running cross-cov: E[DLV_norm^T X] scaled by scale_fact
#         cov_update = ops.matmul(ops.transpose(DLV_norm), X)                         # (ndims, in_dim)
#         self.moving_convX.assign(momentum * self.moving_convX + (one - momentum) * scale_fact * cov_update)

#         # Mark that we've initialized
#         self.run.assign(ops.cast(1.0, self.compute_dtype))

#     def _orthogonalisation_train(self, X, DLV_prev):
#         """
#         Orthogonalize X w.r.t. previous DLVs (batch stats).
#         X        : (batch, in_dim)
#         DLV_prev : (batch, k)   (k < ndims)
#         """
#         # Z-normalize DLV_prev over batch
#         mu = ops.mean(DLV_prev, axis=0)                         # (k,)
#         std = ops.sqrt(ops.var(DLV_prev, axis=0) + self.epsilon)
#         DLV_z = (DLV_prev - mu) / std                           # (batch, k)

#         denom = ops.cast(ops.shape(X)[0], self.compute_dtype)   # batch_size
#         beta = ops.matmul(ops.transpose(DLV_z), X) / denom      # (k, in_dim)
#         ortho_output = X - ops.matmul(DLV_z, beta)              # (batch, in_dim)
#         return ortho_output

#     def _orthogonalisation_test(self, X, DLV_prev):
#         """
#         Orthogonalize X w.r.t. previous DLVs using moving stats.
#         X        : (batch, in_dim)
#         DLV_prev : (batch, k)
#         """
#         k = int(DLV_prev.shape[1])
#         mean_T = ops.transpose(self.DLV_mean[:k, :])                     # (1, k)
#         std_T = ops.transpose(ops.sqrt(self.DLV_var[:k, :]) + self.epsilon)  # (1, k)
#         DLV_norm = (DLV_prev - mean_T) / std_T                           # (batch, k)

#         beta = self.moving_convX[:k, :] / ops.cast(self.tot_num, self.compute_dtype)  # (k, in_dim)
#         ortho_output = X - ops.matmul(DLV_norm, beta)                     # (batch, in_dim)
#         return ortho_output

#     def _calculate_batch_DLV_static(self, X):
#         """
#         Build batch DLVs using STATIC (non-trainable) projection vectors.
#         Returns (batch, ndims). No gradient flows through these weights.
#         """
#         for i in range(self.ndims):
#             if i == 0:
#                 DLV = ops.matmul(X, self.linear_layer_static[i])          # (batch, 1)
#                 DLV_all = DLV                                             # (batch, 1)
#             else:
#                 ortho_output = self._orthogonalisation_train(X, DLV_all[:, :i])
#                 DLV = ops.matmul(ortho_output, self.linear_layer_static[i])  # (batch, 1)
#                 DLV_all = ops.concatenate([DLV_all, DLV], axis=1)         # (batch, i+1)
#         return DLV_all

#     def _calculate_batch_DLV_train(self, X, DLV_all_static):
#         """
#         Build batch DLVs using TRAINABLE projection vectors; grad flows through these.
#         Returns (batch, ndims).
#         """
#         for i in range(self.ndims):
#             if i == 0:
#                 out = ops.matmul(X, self.linear_layer_list[i])               # (batch, 1)
#             else:
#                 ortho_output = self._orthogonalisation_train(X, DLV_all_static[:, :i])
#                 out_i = ops.matmul(ortho_output, self.linear_layer_list[i])  # (batch, 1)
#                 out = ops.concatenate([out, out_i], axis=1)
#         return out

#     def _calculate_DLV_test(self, X):
#         """
#         Compute DLVs at test time using moving statistics.
#         """
#         for i in range(self.ndims):
#             w = self.linear_layer_list[i]  # (in_dim, 1)
#             if i == 0:
#                 out = ops.matmul(X, w)                             # (batch, 1)
#             else:
#                 ortho_output = self._orthogonalisation_test(X, out[:, :i])  # (batch, in_dim)
#                 out_i = ops.matmul(ortho_output, w)                # (batch, 1)
#                 out = ops.concatenate([out, out_i], axis=1)
#         return out

#     # ----------------------------- serialization -----------------------------

#     def get_config(self):
#         base = super().get_config()
#         cfg = {
#             "kernel_regularizer": serialize_reg(self.kernel_regularizer),
#             "momentum": self.momentum,
#             "epsilon": self.epsilon,
#             "tot_num": self.tot_num,
#             "ndims": self.ndims,
#         }
#         return {**base, **cfg}

#     @classmethod
#     def from_config(cls, config):
#         config["kernel_regularizer"] = deserialize_reg(config["kernel_regularizer"])
#         return cls(**config)

#     # (Optional) for parity with your original
#     def build_from_config(self, config):
#         self.build(config["input_shape"])
