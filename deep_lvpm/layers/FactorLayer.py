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
import tensorflow as tf
import tensorflow.keras.layers
import keras
from keras import saving
# changes to git again change


@tf.keras.utils.register_keras_serializable(package='YourPackageName', name='YourCustomName')
class FactorLayer(tf.keras.layers.Layer):
    
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
        kernel_regularizer (tf.keras.regularizers.Regularizer or None): Regularizer function applied to the projection layer's kernel weights.
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
            kernel_regularizer (tf.keras.regularizers.Regularizer, optional): Regularizer function applied to the projection layer's kernel weights.
            epsilon (float, optional): Small constant added to variance to avoid dividing by zero in the batch normalization step. Defaults to 1e-6.
            momentum (float, optional): Momentum for the moving average and moving variance in the batch normalization step. Defaults to 0.95.
            tot_num (int, optional): Total number of samples used for training. Used for optimal scaling of covariance matrices.
            ndims (int, optional): Number of Deep-PLS factor dimensions to extract.
            run (int, optional): Initial value for the run tracker. Defaults to 0.
            **kwargs: Additional keyword arguments inherited from tf.keras.layers.Layer.
        """
        
        super().__init__(**kwargs)

        self.kernel_regularizer = kernel_regularizer ## This kernel regularizer variable determines the degree of regularization that projection weight vectors are subject to
        self.epsilon = epsilon ## This is the offset determined during batch normalisation
        self.momentum = momentum ## This is the amount of momentum that covariance matrices are subject to (see pseudo-code for more details)
        self.batch_norm1 = tf.keras.layers.BatchNormalization(momentum=momentum,epsilon=epsilon)

        # # Additional custom parameters
        self.tot_num = tot_num #kwargs.get("tot_num") ## This is the total number of samples in the full dataset
        self.ndims = ndims #kwargs.get("ndims") ## This is the total number of factors we wish to extract
        self.run=self.add_weight(shape = (), initializer = 'zeros',trainable=False) ## This variable tracks the number of runs we 
        self.first_run =tf.Variable(True,trainable=False)
       
    def build(self, input_shape):
        
        """
        Creates the weights of the layer.

        This function initializes the list of projection vectors, moving mean, moving standard deviation,
        and other variables required for the orthogonalization and normalization processes.

        Args:
            input_shape (tuple): Shape of the input tensor.
        """

        self.linear_layer_list = [None]*self.ndims ## A list of projection layers
        self.linear_layer_static = [None]*self.ndims ## A list containing projection layer weights which are assigned as non-trainable
        
        ## This loop creates n=tot_num projection layers, which are used to construct Deep-PLS factors 
        for i in range(self.ndims):
            linear_layer = self.add_weight(name = 'projection_weight_' + str(i), shape = [input_shape[1],1], initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.), regularizer=self.kernel_regularizer, trainable=True)
            self.linear_layer_list[i]=linear_layer
        
        ## This loop creates n=tot_num static projection layers, which are non-trainable and used in orthogonalisation processes  
        for i in range(self.ndims):
            static_layer = self.add_weight(name = 'static_projection_weight_' + str(i), shape = [input_shape[1],1], initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.), trainable=False)
            self.linear_layer_static[i]=static_layer
        
        self.DLV_mean = self.add_weight(name = 'DLV_moving_mean', shape = [self.ndims,1], initializer='zeros', trainable=False) 
        self.DLV_var = self.add_weight(name = 'DLV_moving_std', shape = [self.ndims,1], initializer='ones', trainable=False) 
        
        self.moving_convX = self.add_weight(name = 'moving_convX', shape=[self.ndims, input_shape[1]], initializer='zeros', trainable=False)
        self.i=tf.Variable(0,trainable=False)

        super(FactorLayer, self).build(input_shape) ## ensures that the layer registers as built
        #self.run=tf.Variable(0,trainable=False)
         
    @tf.function
    def call(self, inputs, training=False):    
        
        """
        Forward pass of the FactorLayer.

        This function applies the projection, batch normalization, orthogonalization, and correlation enhancement steps.

        Args:
            inputs (tf.Tensor): Input tensor.
            training (bool, optional): Indicator for training or inference mode. Defaults to False. The layer performs differently 
            during training and testing.

        Returns:
            tf.Tensor: The output tensor after applying the transformations.
        """

        X = self.batch_norm1(inputs, training=training)
        
        # def run_initialization():
        #     self.moving_variables_initial_values(X)
        #     return X

        # with tf.init_scope():
        #     if self.first_run.value:
        #         self.moving_variables_initial_values(X)
        #         self.first_run.assign(False)


        #Replace Python if statement with tf.cond for graph mode compatibility
        # tf.cond(tf.equal(self.run, 0), run_initialization, lambda: X)

        if training:
            
            ## The algorithm runs differently in training and testing modes. In the training mode,
            ## normalisation and orthogonalisation are carried out using batch-level statistics

            DLV_all = self.calculate_batch_DLV_static(X)
            out = self.calculate_batch_DLV_train(X, DLV_all) 
            
            self.update_moving_variables([X, DLV_all])

            #print(np.corrcoef(tf.transpose(out).numpy()))
        
        else:
            
            out = self.calculate_DLV_test(X) ## Here, we calculate DLVs during testing, using population level statistics

            #tf.print(np.corrcoef(tf.transpose(out).numpy()))
            #print(tf.norm(out,axis=0))

    

        return out
   
    
    # def moving_variables_initial_values(self, X):

    #     with tf.init_scope():
       
    #         """ This function is called the first time the layer is called with data, i.e. when 
    #         self.run=0. Here, the layer takes the first batch of data, and uses it to calculate
    #         the moving variables used by DLVPM.

    #         """

    #         scale_fact = tf.cast(self.tot_num/tf.shape(X)[0],dtype=float)

    #         for i in range(self.ndims): # Here, we loop through weight projection vectors
                
    #             out_init = tf.matmul(X,self.linear_layer_static[i])
    #             self.linear_layer_list[i].assign(tf.divide(self.linear_layer_static[i],tf.norm(out_init)*tf.math.sqrt(scale_fact)))
    #             self.linear_layer_static[i].assign(tf.divide(self.linear_layer_static[i],tf.norm(out_init)*tf.math.sqrt(scale_fact)))

    #         batch_DLV = self.calculate_batch_DLV_static(X) ## Here, we create the DLVPM factors for orthogonalisation, based on weights estimated from the previous batch 
            
    #         self.DLV_mean.assign(tf.expand_dims(tf.math.reduce_mean(batch_DLV, axis=0),axis=1))
    #         self.DLV_var.assign(tf.expand_dims(tf.math.reduce_variance(batch_DLV, axis=0),axis=1))

    #         batch_DLV_norm = tf.divide(tf.subtract(batch_DLV,tf.transpose(self.DLV_mean)),tf.transpose(tf.math.sqrt(self.DLV_var)+self.epsilon))

    #         #ones = tf.ones((tf.shape(batch_DLV_norm)[0], 1))
    #         #batch_DLV_norm = tf.concat([ones, batch_DLV_norm], axis=1)
            
    #         self.moving_convX.assign(scale_fact*tf.matmul(tf.transpose(batch_DLV_norm),X))  ## update moving_convX, which is used for orthogonalisation during testing
                
    #         self.first_run.assign(False)
    #         tf.print('run')
    #         #self.run.assign(1)
    #         self.run.assign_add(1)

    def weight_normalizer(self, inputs):

        """ The purpose of this function is to re-normalize weights weight vectors. This 
        prevents a collapse to a trivial solution. The inputs here are DLVs for this data view. 
        
        """

        out_init = inputs[0]
        scale_fact = inputs[1]

        for i in range(self.ndims): # Here, we loop through weight projection vectors
            
            out_init = tf.matmul(inputs,self.linear_layer_static[i])
            self.linear_layer_list[i].assign(tf.divide(self.linear_layer_static[i],tf.norm(out_init)*tf.math.sqrt(scale_fact)))
            self.linear_layer_static[i].assign(tf.divide(self.linear_layer_static[i],tf.norm(out_init)*tf.math.sqrt(scale_fact)))


    def update_moving_variables(self, inputs):
        
        """ This function is called for every batch the model sees during training. This function
        updates the moving variables using batch-level statistics.
        
        """
        # momentum = self.momentum
        # with tf.init_scope():
        #     if self.first_run.value:
        #         momentum = 0.0
        #         self.first_run.assign(False)
            
        
        #tf.print(self.first_run.value)

        momentum = tf.where(tf.equal(self.run, 1),0.0,self.momentum) ## initialise inputs on first call
   
        scale_fact = tf.cast(self.tot_num/tf.shape(inputs[0])[0],dtype=float)
        #momentum = tf.where(tf.cast(self.run, tf.float32)> tf.cast(1, tf.float32), self.momentum, tf.zeros_like(tf.cast(0, tf.float32)))

        
        batch_DLV_mean = tf.expand_dims(tf.math.reduce_mean(inputs[1], axis=0),axis=1)
        batch_DLV_var = tf.expand_dims(tf.math.reduce_variance(inputs[1], axis=0),axis=1)

        self.DLV_mean.assign(momentum*self.DLV_mean + (tf.constant(1,dtype=float)-momentum)*batch_DLV_mean)
        self.DLV_var.assign(momentum*self.DLV_var + (tf.constant(1,dtype=float)-momentum)*batch_DLV_var)

        batch_DLV_norm = tf.divide(tf.subtract(inputs[1],tf.transpose(batch_DLV_mean)),tf.transpose(tf.math.sqrt(batch_DLV_var)))
        #batch_DLV_norm = tf.divide(tf.subtract(inputs[1],tf.transpose(self.DLV_mean)),tf.transpose(tf.math.sqrt(self.DLV_var)+self.epsilon))

        #ones = tf.ones((tf.shape(batch_DLV_norm)[0], 1))
        #batch_DLV_norm = tf.concat([ones, batch_DLV_norm], axis=1)

        self.moving_convX.assign(momentum*self.moving_convX + scale_fact*(tf.constant(1,dtype=float)-momentum)*tf.matmul(tf.transpose(batch_DLV_norm),inputs[0]))
        
        for i in range(self.ndims): # Here, we loop through weight projection vectors 
            self.linear_layer_static[i].assign(self.linear_layer_list[i])

        #self.run.assign(self.run.value+1)
        self.run.assign_add(1)
       
        

    def orthogonalisation_train(self, inputs):
        
        """ This function is called multiple times during model training. The purpose of this function is to 
        orthogonalise the data with respect to previous DLVs using batch-level statistics

        """
        
        #X = tf.divide(tf.subtract(inputs[0], tf.math.reduce_mean(inputs[0], axis=0)),tf.math.reduce_std(inputs[0], axis=0)+self.epsilon) 
        DLV_batch = tf.divide(tf.subtract(inputs[1], tf.math.reduce_mean(inputs[1], axis=0)),tf.math.reduce_std(inputs[1], axis=0)+self.epsilon) ## Here, we z-normalise the input features to have mean of zero and standard deviation of one 
        
        #ones = tf.ones((tf.shape(DLV_batch)[0], 1))
        #DLV_batch = tf.concat([ones, DLV_batch], axis=1)
        
        denom = tf.cast(tf.shape(inputs[0])[0],dtype=float)
        beta = tf.matmul(tf.transpose(DLV_batch),inputs[0])/denom
        ortho_output = tf.subtract(inputs[0],tf.matmul(DLV_batch, beta)) ## This is the input matrix, orthogonalised with respect to previous DLVs
        
        return ortho_output 
    
    def orthogonalisation_test(self, inputs):
        
        """ This function is called during model testing. This function orthogonalises the data with 
        respect to previous DLVs, using moving variables.

        """
        
        i = inputs[1].shape[1]

        DLV_norm = tf.divide(tf.subtract(inputs[1],tf.transpose(self.DLV_mean[:i,:])),(tf.transpose(tf.math.sqrt(self.DLV_var)[:i,:])+self.epsilon))

        denom= self.tot_num
        beta = self.moving_convX[:i,:]/denom
        ortho_output = tf.subtract(inputs[0],tf.matmul(DLV_norm, beta))

        return ortho_output 
        

    def calculate_batch_DLV_static(self, X):
        
        """ This function is used to calculate DLVs at the batch level. These batch level DLVs 
        can then be used for orthogonalisation in training. Note that this is done using static 
        layers. This means that the backprop algorithm does not see this.

        """
        
        for i in range(self.ndims):
            if i == 0:
                DLV=tf.matmul(X,self.linear_layer_static[i])
                DLV_all=DLV
            else:
                ortho_output=self.orthogonalisation_train([X, DLV_all])
                DLV=tf.matmul(ortho_output,self.linear_layer_static[i])
                DLV_all=tf.concat([DLV_all,DLV],axis=1)
                
        return DLV_all
    

    def calculate_batch_DLV_train(self, X, DLV_all):
        
        """ This function is used to calculate DLVs at the batch level. These batch level DLVs 
        can then be used for orthogonalisation in training. Note that this is done using the training 
        layers, so backprop does see the projections here

        """
        
        for i in range(self.ndims):
            if i == 0:
                out=tf.matmul(X,self.linear_layer_list[i])
            else:
                ortho_output=self.orthogonalisation_train([X, DLV_all[:,:i]])
                out_i=tf.matmul(ortho_output,self.linear_layer_list[i])
                out=tf.concat([out,out_i],axis=1)
                
        return out
    

    def calculate_DLV_test(self, X):

        """ This function is used to calculate DLVs at test time. This function uses the moving 
        variables assigned as class attributes
        """

        for i in range(self.ndims):
                          
            linear_layer = self.linear_layer_list[i] ## select weight projection vector
            
            if i == 0: 
                out = tf.matmul(X,linear_layer)
            else:
                ortho_output=self.orthogonalisation_test([X, out])
                out_i = tf.matmul(ortho_output,linear_layer)
                out=tf.concat([out,out_i],axis=1)
    
        return out

    def get_config(self):
        
        base_config = super().get_config()
        
        config={
            'kernel_regularizer': tf.keras.saving.serialize_keras_object(self.kernel_regularizer),
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'tot_num': self.tot_num,
            'ndims': self.ndims,
            'run': int(self.run.numpy())
            }
        
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        
        config['kernel_regularizer'] = tf.keras.regularizers.deserialize(config['kernel_regularizer'])

        return cls(**config)
    
    def build_from_config(self,config):
         self.build(config["input_shape"])
        
    


