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
import tensorflow as tf
import tensorflow.keras.layers

# changes to git


@tf.keras.utils.register_keras_serializable(package="deep_lvpm",name="ZCALayer")
class ZCALayer(tf.keras.layers.Layer):
    
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
    
    
    def __init__(self, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0), epsilon=1e-3, momentum=0.95, diag_offset=1e-3, tot_num=None, ndims=None, **kwargs):
        
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
        


    def build(self, input_shape):
        
        """ In this function, the model builds and assigns values to the weights used in the DLVPM analysis.
        The function builds the list of projection vectors used to map associations between different data-views. 
        The function also builds the moving mean and moving standard deviation used to normalise the input data.
        """
       
        self.batch_norm1 = tf.keras.layers.BatchNormalization(momentum=self.momentum,epsilon=self.epsilon)
        self.run=self.add_weight(shape = (), initializer = 'zeros',trainable=False, name = 'zcalayer_run') ## This variable tracks the number of runs we 

        ## self.project is the weight projection layer, trained to project variables into a space where they are optimally correlated
        self.project = self.add_weight(name = 'projection_weight_', shape = [input_shape[1],self.ndims], initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.), regularizer=self.kernel_regularizer, trainable=True)
        
        ## self.moving_conv2 and self.moving_convX are covaraince matrices used in the orthonalisation process. These matrices are only used in the testing/prediction phase. self.moving_conv2 is a covaraince matrix expressing the covariances between DLVPM factors, elf.moving_convX is a covariance matrix expressing the covariances between DLVPM factors and the last layer of the neural network
        self.moving_conv2 = self.add_weight(name = 'moving_conv2', shape=[self.ndims, self.ndims], initializer='zeros', trainable=False)
        #init_offset = self.diag_offset*100
        self.moving_conv2.assign(tf.eye(num_rows=self.ndims)) ## this variable is initialised under the assumption that DLVPM factors are uncorrelated with one another

         
    @tf.function
    def call(self, inputs, training=None):    
        
        """ We run the call function during model training. This call function starts with an initialisation,
        which uses the tf.init_scope() function, which takes the process out of backpropagation. Note that the 
        function performs differently during training and testing.
        
        """

        #tf.print("Running the new version of my custom layer!")
    
        X = self.batch_norm1(inputs, training=training)
    
        if training:
            
            ## The algorithm runs differently in training and testing modes. In the training mode,
            ## normalisation and orthogonalisation are carried out using batch-level statistics
        
            #self.project_static.assign(self.project)
            self.update_moving_variables(X)
            out = tf.matmul(X,self.project)


        else:

            out = tf.matmul(X,self.project)

        return out
    

    import tensorflow as tf

    def inv_sqrt_via_cholesky(self, M):
        """
        Computes a triangular factor that squares to M^{-1} for a positive-definite matrix M.
        
        Specifically:
            1) L = cholesky(M)
            2) L_inv = triangular_solve(L, I)
            3) M^{-1} = L_inv^T @ L_inv
            4) A valid 'square root' of M^{-1} is L_inv^T (or cholesky(M_inv)), 
            which you can multiply by vectors to get M^{-1/2} * x.
            
        Returns:
            A (upper) triangular factor R such that R^T R = M^{-1}.
            i.e. R = L_inv^T and M^{-1} = R^T R.
        """
        # 1) Cholesky factor: M = L L^T
        L = tf.linalg.cholesky(M)  
        # 2) Invert L by solving L * X = I
        n = tf.shape(M)[0]
        I = tf.eye(n, dtype=M.dtype)
        L_inv = tf.linalg.triangular_solve(L, I, lower=True)  
        # M^{-1} = L_inv^T @ L_inv
        
        # If we define R = L_inv^T, then:
        #   R^T R = (L_inv^T)^T (L_inv^T) = L_inv L_inv^T = M^{-1}.
        # But we typically use R^T R form. If we want R R^T = M^{-1},
        # you could return L_inv instead.
        
        # We'll return the upper-triangular factor, R = L_inv^T
        return tf.transpose(L_inv)

    
    def weight_normalizer(self, inputs):

        """ The purpose of this function is to re-normalize weights weight vectors. This 
        prevents a collapse to a trivial solution. The inputs here are DLVs for this data view. 
        
        """

        y = inputs[0]
        scale_fact = inputs[1]

        diag = self.diag_offset
        denom = tf.math.sqrt(tf.math.multiply(scale_fact,tf.math.reduce_sum(tf.math.square(y),axis=0)))
        self.project.assign(tf.divide(self.project,denom)) ## Here, we normalize the DLVPM weights

    
        # #sqrt_inv_y =tf.where(tf.equal(self.run, 0),tf.linalg.sqrtm(tf.linalg.inv(tf.matmul(tf.transpose(y),y)+100*diag*tf.eye(self.moving_conv2.shape[0]))), tf.linalg.sqrtm(tf.linalg.inv(self.moving_conv2+diag*tf.eye(self.moving_conv2.shape[0])))) ## pseudo inverse called on first batch for improved numeric stability
        #sqrt_inv_y =tf.linalg.sqrtm(tf.linalg.inv(self.moving_conv2+diag*tf.eye(self.moving_conv2.shape[0]))) ## pseudo inverse called on first batch for improved numeric stability
        #sqrt_inv_y  = self.inv_sqrt_via_cholesky(self.moving_conv2+diag*tf.eye(self.moving_conv2.shape[0]))  # shape = [dim, dim]
        sqrt_inv_y  = self.inv_sqrt_via_cholesky(self.moving_conv2+self.decaying_diagonal(self.run, self.moving_conv2.shape[0]))  # shape = [dim, dim]
        
        
        #scale_fact*(tf.matmul(tf.transpose(y),y))
        #sqrt_inv_y  =  self.inv_sqrt_via_cholesky(scale_fact*(tf.matmul(tf.transpose(y),y))+diag*tf.eye(self.moving_conv2.shape[0]))

        out_y = tf.matmul(tf.squeeze(y),sqrt_inv_y) ## Here, we normalize the output DLVs

        return out_y



    def update_moving_variables(self, X):
        
        """ This function is called for every batch the model sees during training. This function
        updates the moving variables using batch-level statistics.
        
        """


        scale_fact = tf.cast(self.tot_num/tf.shape(X)[0],dtype=float)

        DLVs = tf.matmul(X, self.project)

        #momentum = tf.where(tf.equal(self.run, 0), 0.0, self.momentum) ## initialise inputs on first call
        #denom = tf.where(tf.equal(self.run, 0), 0.0, tf.math.sqrt(tf.math.multiply(scale_fact,tf.math.reduce_sum(tf.math.square(DLVs),axis=0))))

        #tf.where(tf.equal(self.run, 0), self.project.assign(tf.divide(self.project,tf.math.sqrt(tf.math.multiply(scale_fact,tf.math.reduce_sum(tf.math.square(DLVs),axis=0))))), self.project)
        
        momentum = self.momentum

        self.moving_conv2.assign(momentum*self.moving_conv2 + scale_fact*(tf.constant(1,dtype=float)-momentum)*(tf.matmul(tf.transpose(DLVs),DLVs)))
       
        self.run.assign_add(1.0)



    def decaying_diagonal(self, step, dim, final_eps=1e-4, decay_rate=0.1):
        """
        Returns a (dim x dim) identity matrix scaled by an epsilon value that decays 
        exponentially over 'step'.
        
        Parameters:
        -----------
        step : tf.Tensor or int
            Current step or iteration count.
        dim : int
            The dimension for the identity matrix.
        initial_eps : float
            The initial (large) epsilon value.
        final_eps : float
            The final (small) epsilon value after many steps.
        decay_rate : float
            Exponential decay rate.

        Returns:
        --------
        tf.Tensor
            A (dim x dim) diagonal matrix with exponentially decaying scale.
        """
        # Convert step to float in case it's an integer or scalar tensor
        step = tf.cast(step, tf.float32)-tf.cast(1, tf.float32)

        tf.print()

        initial_eps = self.diag_offset

        # Exponential decay from initial_eps down to final_eps
        current_eps = final_eps + (initial_eps - final_eps) * tf.exp(-decay_rate * step)

        #tf.print(current_eps)

        # Return scaled identity matrix
        return current_eps * tf.eye(dim, dtype=tf.float32)

        
    def get_config(self):
        """
        Returns the configuration of the custom layer for saving and loading.

        Returns:
        config (dict): A Python dictionary containing the layer configuration.
        """
        config = super().get_config().copy()
        config.update({
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
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
        config['kernel_regularizer'] = tf.keras.regularizers.deserialize(config['kernel_regularizer'])
        return cls(**config)
    
    
    
    
    
    