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

#@keras.saving.register_keras_serializable()
class FactorLayer(tf.keras.layers.Layer):
    
    """This layer should be placed at the end of DPLS-PM models. The layer 
    generates orthogonal factors that are highly correlated between data-views. 
    
    This layer is constructed of three basic parts. The first set of operations
    involve carrying out batch normalisation on the inputs. In the second set of 
    operations, we orthogonalise the second set of inputs with respect to the first.
    We then use a linear layer to project the output of the neural network into a 
    space where it correlates with the outputs of other data-views.
    
    The ordering of the layer calculations is: batch normalisation > orthogonalisation 
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
    
    tot_dims: parameter that defines the number of Deep-PLS factor dimensions 
    we wish to extract
    
    
    Call arguments:
    inputs: A single tensor, which is used for the purposes of projecting to 
    other data-views, identifying factors that are highly correlated between 
    data-views. 
    
    """
    
    
    def __init__(self, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0), epsilon=1e-6, momentum=0.95, **kwargs):
        
        super().__init__()

        self.kernel_regularizer = kernel_regularizer ## This kernel regularizer variable determines the degree of regularization that projection weight vectors are subject to
        self.momentum = momentum ## This is the amount of momentum that covariance matrices are subject to (see pseudo-code for more details)
        self.epsilon = epsilon ## This is the offset determined during batch normalisation
        
        
    def global_build(self,tot_num, tot_dims):
        
        """ This function is called internally during the compile step, when we 
        know the total number of samples, and the total number of dimensions we
        wish to use, which are global properties of the model.
        
        """
        
        self.tot_num = tot_num ## Total_num is the total sample size, used to calculate covariance matrices
        self.tot_dims = tot_dims ## Total number of latent factors we wish to extract for each data-view

       
    
    def build(self, input_shape):
        
        """ In this function, the model builds and assigns values to the weights used in the Deep-PLS analysis.
        The function builds the list of projection vectors used to map associations between different data-views. 
        The function also builds the moving mean and moving standard deviation used to normalise the input data.
        """
        
        self.linear_layer_list = [None]*self.tot_dims ## A list of projection layers
        self.linear_layer_static = [None]*self.tot_dims ## A list containing projection layer weights which are assigned as non-trainable
        
        
        ## This loop creates n=tot_num projection layers, which are used to construct Deep-PLS factors 
        for i in range(self.tot_dims):
            linear_layer = self.add_weight(name = 'projection_weight_' + str(i), shape = [input_shape[1],1], initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.), regularizer=self.kernel_regularizer, trainable=True)
            
            self.linear_layer_list[i]=linear_layer
            #self.linear_layer_list[i]=linear_layer
        
        ## This loop creates n=tot_num static projection layers, which are non-trainable and used in orthogonalisation processes  
        for i in range(self.tot_dims):
            static_layer = self.add_weight(name = 'static_projection_weight_' + str(i), shape = [input_shape[1],1], initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.), trainable=False)

            self.linear_layer_static[i]=static_layer
        
        ## self.moving_mean and self.moving_std are used to z-normalise the data-inputs to this, last layer, of the neural network, used in the testing/prediction phase only
        self.moving_mean = self.add_weight(name = 'moving_mean', shape = [input_shape[1],1], initializer='zeros', trainable=False) ## inputs are normalised using: inputs - self.moving_mean
        self.moving_var = self.add_weight(name = 'moving_std', shape = [input_shape[1],1], initializer='ones', trainable=False) ## inputs are
        
        ## self.c_mat_mean and self.c_mat_std are used to z-normalise Deep-PLS factors during the orthognalisation process, used in the testing/prediction phase only
        self.c_mat_mean = self.add_weight(name = 'c_mat_moving_mean', shape = [self.tot_dims,1], initializer='zeros', trainable=False) 
        self.c_mat_var = self.add_weight(name = 'c_mat_moving_std', shape = [self.tot_dims,1], initializer='ones', trainable=False) 
        
        
        ## self.moving_conv2 and self.moving_convX are covaraince matrices used in the orthonalisation process. These matrices are only used in the testing/prediction phase. self.moving_conv2 is a covaraince matrix expressing the covariances between Deep-PLS factors, elf.moving_convX is a covariance matrix expressing the covariances between Deep-PLS factors and the last layer of the neural network
        #self.moving_conv2 = self.add_weight(name = 'moving_conv2', shape=[self.tot_dims, self.tot_dims], initializer='zeros', trainable=False)
        #self.moving_conv2.assign(tf.eye(num_rows=self.tot_dims)) ## this variable is initialised under the assumption that Deep-PLS factors are uncorrelated with one another
        self.moving_convX = self.add_weight(name = 'moving_convX', shape=[self.tot_dims, input_shape[1]], initializer='zeros', trainable=False)
        
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
            c_mat = self.calculate_cmat(inputs)
            
            for i in range(self.tot_dims): # Here, we loop through weight projection vectors 
        
                if i == 0:
                    X = tf.divide(tf.subtract(inputs, tf.math.reduce_mean(inputs, axis=0)),tf.math.reduce_std(inputs, axis=0)+self.epsilon) 
                    out = tf.matmul(X,self.linear_layer_list[i])
                else:
                    lin_input = self.training_run([inputs,c_mat[:,:i]])
                    out_lay = tf.matmul(lin_input,self.linear_layer_list[i])
                    out=tf.concat([out,out_lay],axis=1)
                    
            self.update_moving_variables([inputs,c_mat])
        
        else:
            
            X = tf.divide(tf.subtract(inputs, (tf.transpose(self.moving_mean)+self.epsilon)),(tf.transpose(tf.math.sqrt(self.moving_var))+self.epsilon))
            
            for i in range(self.tot_dims):
                          
                linear_layer = self.linear_layer_list[i] ## select weight projection vector
                
                if i == 0: 
                    out = tf.matmul(X,linear_layer)
                    conv = tf.divide(tf.subtract(out,tf.transpose(self.c_mat_mean[:(i+1),:])),(tf.transpose(tf.math.sqrt(self.c_mat_var)[:(i+1),:])))
                else:
                    #beta = tf.matmul(tf.linalg.inv(self.moving_conv2[:i,:i]),self.moving_convX[:i,:])   
                    beta = self.moving_convX[:i,:]/self.tot_num
                    lin_input = tf.subtract(X,tf.matmul(conv, beta))
                    out_lay = tf.matmul(lin_input,linear_layer)
                    out=tf.concat([out,out_lay],axis=1)
        
                    conv = tf.divide(tf.subtract(out,tf.transpose(self.c_mat_mean[:(i+1),:])),(tf.transpose(tf.math.sqrt(self.c_mat_var)[:(i+1),:])))

         
        return out
   
    
    def moving_variables_initial_values(self, inputs):
       
       """ This function is called the first time the layer is called with data, i.e. when 
       self.i=0. Here, the layer takes the first batch of data, and uses it to calculate
       the moving variables used by Deep-PLS during inference.
       
       """
      
       scale_fact = tf.cast(self.tot_num/tf.shape(inputs)[0],dtype=float)
       
       self.moving_mean.assign(tf.expand_dims(tf.math.reduce_mean(inputs, axis=0),axis=1))
       self.moving_var.assign(tf.expand_dims(tf.math.reduce_variance(inputs, axis=0),axis=1))
       
       X=tf.divide(tf.subtract(inputs,tf.transpose(self.moving_mean)),tf.transpose(tf.math.sqrt(self.moving_var)))
       
       for i in range(self.tot_dims): # Here, we loop through weight projection vectors
           
           out_init = tf.matmul(X,self.linear_layer_static[i])
           self.linear_layer_list[i].assign(tf.divide(self.linear_layer_static[i],tf.math.sqrt(tf.math.multiply(scale_fact,tf.math.reduce_sum(tf.math.square(out_init),axis=0)))))
           self.linear_layer_static[i].assign(tf.divide(self.linear_layer_static[i],tf.math.sqrt(tf.math.multiply(scale_fact,tf.math.reduce_sum(tf.math.square(out_init),axis=0)))))
    
       c_mat = self.calculate_cmat(inputs) ## Here, we create the DPLS factors for orthogonalisation, based on weights estimated from the previous batch 
    
       self.c_mat_mean.assign(tf.expand_dims(tf.math.reduce_mean(c_mat, axis=0),axis=1))
       self.c_mat_var.assign(tf.expand_dims(tf.math.reduce_variance(c_mat, axis=0),axis=1))
    
       conv = tf.divide(tf.subtract(c_mat,tf.transpose(self.c_mat_mean)),tf.transpose(tf.math.sqrt(self.c_mat_var)))
       
       #self.moving_conv2.assign(scale_fact*tf.matmul(tf.transpose(conv),conv))
       self.moving_convX.assign(scale_fact*tf.matmul(tf.transpose(conv),X))  
          
       self.i.assign(1)
    
    def training_run(self, inputs):
        
        """ This function is called multiple times during training. The first operation carried out 
        here is to z-normalise the inputs using the batch-level mean and standard deviation of those
        inputs. The inputs are then orthogonalised with respect to previous Deep-PLS factors. The resulting
        features, which have then been orthognalised with respect to previous Deep-PLS factors, are then
        multiplied by projection weights, which are trained in this process.
        
        """
        
        X = tf.divide(tf.subtract(inputs[0], tf.math.reduce_mean(inputs[0], axis=0)),tf.math.reduce_std(inputs[0], axis=0)+self.epsilon) 
        conv = tf.divide(tf.subtract(inputs[1], tf.math.reduce_mean(inputs[1], axis=0)),tf.math.reduce_std(inputs[1], axis=0)+self.epsilon) ## Here, we z-normalise the input features to have mean of zero and standard deviation of one 
        
        #beta=tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(tf.transpose(conv),conv)),tf.transpose(conv)),X)
        #beta = tf.matmul(tf.transpose(conv),X)/tf.cast(tf.shape(inputs)[0],dtype=float)
        denom = tf.cast(tf.shape(inputs[0])[0],dtype=float)
        beta = tf.matmul(tf.transpose(conv),X)/denom
        
        lin_input = tf.subtract(X,tf.matmul(conv, beta))
        
        return lin_input 
        
   
    def update_moving_variables(self, inputs):
        
        """ This function is called for every batch the model sees during training. This function
        updates the moving variables using batch-level statistics.
        
        """
   
        
        scale_fact = tf.cast(self.tot_num/tf.shape(inputs[0])[0],dtype=float)
        
        self.moving_mean.assign(self.momentum*self.moving_mean + (tf.constant(1,dtype=float)-self.momentum)*tf.expand_dims(tf.math.reduce_mean(inputs[0], axis=0),axis=1))
        self.moving_var.assign(self.momentum*self.moving_var + (tf.constant(1,dtype=float)-self.momentum)*tf.expand_dims(tf.math.reduce_variance(inputs[0], axis=0),axis=1))
        
        self.c_mat_mean.assign(self.momentum*self.c_mat_mean + (tf.constant(1,dtype=float)-self.momentum)*tf.expand_dims(tf.math.reduce_mean(inputs[1], axis=0),axis=1))
        self.c_mat_var.assign(self.momentum*self.c_mat_var + (tf.constant(1,dtype=float)-self.momentum)*tf.expand_dims(tf.math.reduce_variance(inputs[1], axis=0),axis=1))
        
        X=tf.divide(tf.subtract(inputs[0],tf.transpose(self.moving_mean)),tf.transpose(tf.math.sqrt(self.moving_var)))
        conv = tf.divide(tf.subtract(inputs[1],tf.transpose(self.c_mat_mean)),tf.transpose(tf.math.sqrt(self.c_mat_var)))
    
        #self.moving_conv2.assign(self.momentum*self.moving_conv2 + scale_fact*(tf.constant(1,dtype=float)-self.momentum)*tf.matmul(tf.transpose(conv),conv))
        self.moving_convX.assign(self.momentum*self.moving_convX + scale_fact*(tf.constant(1,dtype=float)-self.momentum)*tf.matmul(tf.transpose(conv),X))  
        
        for i in range(self.tot_dims): # Here, we loop through weight projection vectors 
            self.linear_layer_static[i].assign(self.linear_layer_list[i])
        
        
          
    def calculate_cmat(self, inputs):
        
        """ This function is used to calculate Deep-PLS factors. These factors are then
        used to orthognalise the data inputs with respect to previous Deep-PLS factors.
        These Deep-PLS factors are calculated using the static weights, which were estimated
        in the previous batch iteration.
        """
        
        
        for i in range(self.tot_dims):
            if i == 0:
                X = tf.divide(tf.subtract(inputs, tf.math.reduce_mean(inputs, axis=0)),tf.math.reduce_std(inputs, axis=0)+self.epsilon) 
                c=tf.matmul(X,self.linear_layer_static[i])
                c_arr=c
            else:
                lin_input=self.training_run([inputs, c_arr])
                c=tf.matmul(lin_input,self.linear_layer_static[i])
                c_arr=tf.concat([c_arr,c],axis=1)
                
        return c_arr
        
    # At present, serialisation of layers is not required as this layer is only used 
    # internally by the measurement model layer
    
    # def get_config(self):
        
    #     config = super().get_config().copy()
        
    #     config.update({
    #         'kernel_regularizer': self.kernel_regularizer,
    #         'momentum': self.momentum,
    #         'epsilon': self.epsilon,
    #         'tot_num': self.tot_num,
    #         'tot_dims': self.tot_dims,
    #         })
        
    #     return config
    
    # @classmethod
    # def from_config(cls, config):
        
    #     cls(**config)
    #     cls.global_build(cls, config["tot_num"],config["tot_dims"])
        
    #     return cls
    
    
# F = FactorLayer()
# F.global_build(100,5)      
# conf = F.get_config()
# F.from_config(conf)
    
    