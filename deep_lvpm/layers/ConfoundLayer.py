#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:24:12 2023

@author: ing
"""

import tensorflow as tf
import scipy.io as sio
import tensorflow.keras as keras


import tensorflow as tf
import scipy.io as sio
import tensorflow.keras as keras


@keras.saving.register_keras_serializable(package="deep_lvpm",name="ConfoundLayer")
class ConfoundLayer(tf.keras.layers.Layer):
    
    """ The purpose of this layer is to orthogonalise data-inputs with respect
    to a set of input confounds
    
    
    
    call_inputs:
    input[0]: This should be the data input that we want to orthogonalise with
    respect to input[1]
    input[1]: We orthogonlise input[0] with respect to input[1]
    
    """
    
    
    def __init__(self, tot_num, epsilon=1e-4, momentum=0.95, diag_offset=1e-6, run=0, **kwargs):
        
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
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.run=tf.Variable(run,trainable=False)

    def build(self, input_shape):
        
        """ In this function, the model builds and assigns values to the weights used in the Deep-PLS analysis.
        The function builds the list of projection vectors used to map associations between different data-views. 
        The function also builds the moving mean and moving standard deviation used to normalise the input data.
        """
      
        ## self.moving_mean and self.moving_std are used to z-normalise the data-inputs to this, last layer, of the neural network, used in the testing/prediction phase only
        self.moving_mean = self.add_weight(name = 'moving_mean', shape = [input_shape[0][1],1], initializer='zeros', trainable=False) ## inputs are normalised using: inputs - self.moving_mean
        self.moving_var = self.add_weight(name = 'moving_std', shape = [input_shape[0][1],1], initializer='ones', trainable=False) ## inputs are
        
        self.moving_conv2 = self.add_weight(name = 'moving_conv2', shape=[input_shape[1][1]+1, input_shape[1][1]+1], initializer='zeros', trainable=False)
       
        self.moving_convX = self.add_weight(name = 'moving_convX', shape=[input_shape[1][1]+1, input_shape[0][1]], initializer='zeros', trainable=False)
        
        
        
    @tf.function
    def call(self, inputs, training=None):    
        
        """ We run the call function during model training. This call function starts with an initialisation,
        which uses the tf.init_scope() function, which takes the process out of backpropagation. Note that the 
        function performs differently during training and testing.
        
        """

        input1, input2 = inputs

        # Apply batch normalization to each input
        bn_input1 = self.batch_norm1(input1, training=training)
        bn_input2 = self.batch_norm2(input2, training=training)

        # Concatenate the batch-normalized inputs
        inputs = [bn_input1, bn_input2]

        conv = inputs[1]

        ones = tf.ones((tf.shape(conv)[0], 1))
        conv = tf.concat([ones, conv], axis=1)
      
        #tf.cond(self.i==0,true_fn=self.moving_variables_initial_values(inputs),false_fn=None)
        tf.cond(tf.equal(self.run, 0),lambda: self.moving_variables_initial_values([inputs[0], conv]),lambda: None)
     
        if training: 
            ## The algorithm runs differently in training and testing modes. In the training mode,

            X=inputs[0]

            #beta = (tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(tf.transpose(conv),conv)+self.diag_offset*tf.eye(conv.shape[1])),tf.transpose(conv)),X))
            beta = tf.matmul(tf.linalg.inv(self.moving_conv2+self.diag_offset*tf.eye(conv.shape[1])),self.moving_convX) 
            X_out = tf.subtract(X,tf.matmul(conv, beta))
               
            self.update_moving_variables([inputs[0], conv])
    
        else:
            
            X=inputs[0]
            #X = tf.divide(tf.subtract(inputs[0], (tf.transpose(self.moving_mean)+self.epsilon)),(tf.transpose(tf.math.sqrt(self.moving_var))+self.epsilon))
            
            beta = tf.matmul(tf.linalg.inv(self.moving_conv2+self.diag_offset*tf.eye(conv.shape[1])),self.moving_convX) 
            X_out = tf.subtract(X,tf.matmul(conv, beta))

        return X_out
   
    
    def moving_variables_initial_values(self, inputs):
       
        """ This function is called the first time the layer is called with data, i.e. when 
        self.i=0. Here, the layer takes the first batch of data, and uses it to calculate
        the moving variables used by Deep-PLS during inference.
        
        """
      
        conv=inputs[1]

        scale_fact = tf.cast(self.tot_num/tf.shape(inputs[0])[0],dtype=float)

        self.moving_mean.assign(tf.expand_dims(tf.math.reduce_mean(inputs[0], axis=0),axis=1))
        self.moving_var.assign(tf.expand_dims(tf.math.reduce_variance(inputs[0], axis=0),axis=1))

        X=inputs[0]
        #X=tf.divide(tf.subtract(inputs[0],tf.transpose(self.moving_mean)),tf.transpose(tf.math.sqrt(self.moving_var)+self.epsilon))

        self.moving_conv2.assign(scale_fact*tf.matmul(tf.transpose(conv),conv))
        self.moving_convX.assign(scale_fact*tf.matmul(tf.transpose(conv),X))  
            
        self.run.assign(1)
        
   
    def update_moving_variables(self, inputs):
        
        """ This function is called for every batch the model sees during training. This function
        updates the moving variables using batch-level statistics.
        
        """
   
        scale_fact = tf.cast(self.tot_num/tf.shape(inputs[0])[0],dtype=float)
        
        self.moving_mean.assign(self.momentum*self.moving_mean + (tf.constant(1,dtype=float)-self.momentum)*tf.expand_dims(tf.math.reduce_mean(inputs[0], axis=0),axis=1))
        self.moving_var.assign(self.momentum*self.moving_var + (tf.constant(1,dtype=float)-self.momentum)*tf.expand_dims(tf.math.reduce_variance(inputs[0], axis=0),axis=1))
        
        #X=tf.divide(tf.subtract(inputs[0],tf.transpose(self.moving_mean)),tf.transpose(tf.math.sqrt(self.moving_var)+self.epsilon))
        X=inputs[0]
        conv = inputs[1]
    
        self.moving_conv2.assign(self.momentum*self.moving_conv2 + scale_fact*(tf.constant(1,dtype=float)-self.momentum)*tf.matmul(tf.transpose(conv),conv))
        self.moving_convX.assign(self.momentum*self.moving_convX + scale_fact*(tf.constant(1,dtype=float)-self.momentum)*tf.matmul(tf.transpose(conv),X))  
    
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
            'run': int(self.run.numpy())
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