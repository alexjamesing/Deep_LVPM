#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:24:12 2023

@author: ing
"""

import tensorflow as tf
import scipy.io as sio
import tensorflow.keras as keras


#@keras.saving.register_keras_serializable()

class ConfoundLayer(tf.keras.layers.Layer):
    
    """ The purpose of this layer is to orthogonalise data-inputs with respect
    to a set of input confounds
    
    
    
    call_inputs:
    input[0]: This should be the data input that we want to orthogonalise with
    respect to input[1]
    input[1]: We orthogonlise input[0] with respect to input[1]
    
    """
    
    
    def __init__(self, tot_num, epsilon=1e-12, momentum=0.95, diag_offset=1e-12):
        
        super().__init__()
        self.tot_num = tot_num
        self.momentum = momentum ## This is the amount of momentum that covariance matrices are subject to (see pseudo-code for more details)
        self.epsilon = epsilon ## This is the offset determined during batch normalisation
        self.diag_offset =diag_offset ## This is a offset added to the diagonal of the covariance matrix between confounds, to ensure that this matrix is invertable
    
    def build(self, input_shape):
        
        """ In this function, the model builds and assigns values to the weights used in the Deep-PLS analysis.
        The function builds the list of projection vectors used to map associations between different data-views. 
        The function also builds the moving mean and moving standard deviation used to normalise the input data.
        """
      
        ## self.moving_mean and self.moving_std are used to z-normalise the data-inputs to this, last layer, of the neural network, used in the testing/prediction phase only
        self.moving_mean = self.add_weight(name = 'moving_mean', shape = [input_shape[0][1],1], initializer='zeros', trainable=False) ## inputs are normalised using: inputs - self.moving_mean
        self.moving_var = self.add_weight(name = 'moving_std', shape = [input_shape[0][1],1], initializer='ones', trainable=False) ## inputs are
        
        ## self.c_mat_mean and self.c_mat_std are used to z-normalise Deep-PLS factors during the orthognalisation process, used in the testing/prediction phase only
        self.c_mat_mean = self.add_weight(name = 'c_mat_moving_mean', shape = [input_shape[1][1],1], initializer='zeros', trainable=False) 
        self.c_mat_var = self.add_weight(name = 'c_mat_moving_std', shape = [input_shape[1][1],1], initializer='ones', trainable=False) 
        
        self.moving_conv2 = self.add_weight(name = 'moving_conv2', shape=[input_shape[1][1], input_shape[1][1]], initializer='zeros', trainable=False)
       
        self.moving_convX = self.add_weight(name = 'moving_convX', shape=[input_shape[1][1], input_shape[0][1]], initializer='zeros', trainable=False)
        
        self.i=tf.Variable(0,trainable=False)
        
        
    @tf.function
    def call(self, inputs, training=None):    
        
        """ We run the call function during model training. This call function starts with an initialisation,
        which uses the tf.init_scope() function, which takes the process out of backpropagation. Note that the 
        function performs differently during training and testing.
        
        """
       
      
        #tf.cond(self.i==0,true_fn=self.moving_variables_initial_values(inputs),false_fn=None)
        if self.i==0:
            self.moving_variables_initial_values(inputs)
        
     
        if training:
            
            ## The algorithm runs differently in training and testing modes. In the training mode,
            ## normalisation and orthogonalisation are carried out using batch-level statistics
        
            X = tf.divide(tf.subtract(inputs[0], tf.math.reduce_mean(inputs[0], axis=0)),tf.math.reduce_std(inputs[0], axis=0)+self.epsilon) 
            conv = tf.divide(tf.subtract(inputs[1], tf.math.reduce_mean(inputs[1], axis=0)),tf.math.reduce_std(inputs[1], axis=0)+self.epsilon) ## Here, we z-normalise the input features to have mean of zero and standard deviation of one 
            
            #beta = (tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(tf.transpose(conv),conv)+self.diag_offset*tf.eye(conv.shape[1])),tf.transpose(conv)),X))
            beta = tf.matmul(tf.linalg.inv(self.moving_conv2+self.diag_offset*tf.eye(conv.shape[1])),self.moving_convX) 
            X_out = tf.subtract(X,tf.matmul(conv, beta))
               
        
            self.update_moving_variables(inputs)
    
        else:
            
            if len(inputs)==2:
            
                X = tf.divide(tf.subtract(inputs[0], (tf.transpose(self.moving_mean)+self.epsilon)),(tf.transpose(tf.math.sqrt(self.moving_var))+self.epsilon))
                conv = tf.divide(tf.subtract(inputs[1],tf.transpose(self.c_mat_mean)),(tf.transpose(tf.math.sqrt(self.c_mat_var)+self.epsilon)))
            
                beta = tf.matmul(tf.linalg.inv(self.moving_conv2+self.diag_offset*tf.eye(conv.shape[1])),self.moving_convX) 
                X_out = tf.subtract(X,tf.matmul(conv, beta))
    
            else:
                
                X_out = tf.divide(tf.subtract(inputs, (tf.transpose(self.moving_mean)+self.epsilon)),(tf.transpose(tf.math.sqrt(self.moving_var))+self.epsilon))
                
                
         
        return X_out
   
    
    def moving_variables_initial_values(self, inputs):
       
       """ This function is called the first time the layer is called with data, i.e. when 
       self.i=0. Here, the layer takes the first batch of data, and uses it to calculate
       the moving variables used by Deep-PLS during inference.
       
       """
      
       scale_fact = tf.cast(self.tot_num/tf.shape(inputs[0])[0],dtype=float)
       
       self.moving_mean.assign(tf.expand_dims(tf.math.reduce_mean(inputs[0], axis=0),axis=1))
       self.moving_var.assign(tf.expand_dims(tf.math.reduce_variance(inputs[0], axis=0),axis=1))
       
       self.c_mat_mean.assign(tf.expand_dims(tf.math.reduce_mean(inputs[1], axis=0),axis=1))
       self.c_mat_var.assign(tf.expand_dims(tf.math.reduce_variance(inputs[1], axis=0),axis=1))
    
       X=tf.divide(tf.subtract(inputs[0],tf.transpose(self.moving_mean)),tf.transpose(tf.math.sqrt(self.moving_var)+self.epsilon))
       conv = tf.divide(tf.subtract(inputs[1],tf.transpose(self.c_mat_mean)),tf.transpose(tf.math.sqrt(self.c_mat_var)+self.epsilon))
       
       self.moving_conv2.assign(scale_fact*tf.matmul(tf.transpose(conv),conv))
       self.moving_convX.assign(scale_fact*tf.matmul(tf.transpose(conv),X))  
          
       self.i.assign(1)
        
   
    def update_moving_variables(self, inputs):
        
        """ This function is called for every batch the model sees during training. This function
        updates the moving variables using batch-level statistics.
        
        """
   
        scale_fact = tf.cast(self.tot_num/tf.shape(inputs[0])[0],dtype=float)
        
        self.moving_mean.assign(self.momentum*self.moving_mean + (tf.constant(1,dtype=float)-self.momentum)*tf.expand_dims(tf.math.reduce_mean(inputs[0], axis=0),axis=1))
        self.moving_var.assign(self.momentum*self.moving_var + (tf.constant(1,dtype=float)-self.momentum)*tf.expand_dims(tf.math.reduce_variance(inputs[0], axis=0),axis=1))
        
        self.c_mat_mean.assign(self.momentum*self.c_mat_mean + (tf.constant(1,dtype=float)-self.momentum)*tf.expand_dims(tf.math.reduce_mean(inputs[1], axis=0),axis=1))
        self.c_mat_var.assign(self.momentum*self.c_mat_var + (tf.constant(1,dtype=float)-self.momentum)*tf.expand_dims(tf.math.reduce_variance(inputs[1], axis=0),axis=1))
        
        X=tf.divide(tf.subtract(inputs[0],tf.transpose(self.moving_mean)),tf.transpose(tf.math.sqrt(self.moving_var)+self.epsilon))
        conv = tf.divide(tf.subtract(inputs[1],tf.transpose(self.c_mat_mean)),tf.transpose(tf.math.sqrt(self.c_mat_var)+self.epsilon))
    
        self.moving_conv2.assign(self.momentum*self.moving_conv2 + scale_fact*(tf.constant(1,dtype=float)-self.momentum)*tf.matmul(tf.transpose(conv),conv))
        self.moving_convX.assign(self.momentum*self.moving_convX + scale_fact*(tf.constant(1,dtype=float)-self.momentum)*tf.matmul(tf.transpose(conv),X))  
        
        