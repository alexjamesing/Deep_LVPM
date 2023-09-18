#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:06:47 2022

@author: ing
"""

import os
current_path=os.path.dirname(os.path.realpath(__file__))
print(current_path)


import tensorflow as tf
import numpy as np
# from Custom_Losses_and_Metrics import mse_loss
# from Custom_Losses_and_Metrics import corr_metric


loss_tracker_total = tf.keras.metrics.Mean(name="total_loss")
loss_tracker_mse = tf.keras.metrics.Mean(name="mean_squared_loss")
corr_tracker = tf.keras.metrics.Mean(name="corr_metric")

#tf.config.run_functions_eagerly(True)

## changes to git

class StructuralModel(tf.keras.Model):
    
    """ This is a custom model that can be used to establish associations 
    between different data-views. 
    
    Args:
    C_mv: This is a binary adjacency matrix, which defines the data-views that should 
    be linked via DLVPM. Here, ones represent connections and zeros represent
    un-connected data-views. In PLS-PM parlance, C_mv defines the structural model.
    model_list: This is a 1 x nview list of tf.keras models where nview is the 
    number of data-views under analysis
    ndims: This is the number of orthogonal latent variables to construct between
    different data-views.
    epochs: This is the number of epochs to run through for DNN models
    tot_num: This is the total number of features associated with the data input, 
    accross all batches. This number is required for normsalisation
    orthogonalisation: This is the orthogonalisation procedure, should be 'zca' or 
    'Moore-Penrose', defaults to 'Moore-Penrose'.
    
    
    
    
    """
    
    def __init__(self, C_mv, model_list, tot_num, ndims, epochs, batch_size, orthogonalisation='Moore-Penrose'):
        
        super().__init__()    
        
        self.C_mv = C_mv
        self.model_list = model_list
        self.epochs = epochs
        self.tot_num = tot_num
        self.ndims = ndims
        self.batch_size = batch_size
        self.orthogonalisation=orthogonalisation
        self.loss_tracker_total = tf.keras.metrics.Mean(name="total_loss")
        self.corr_tracker = tf.keras.metrics.Mean(name="cross_metric")
        self.loss_tracker_mse = tf.keras.metrics.Mean(name="mse_loss")
    
    
    def call(self,inputs):
        """ This is where the layer logic lives. Here, the global model runs 
        data through each of the measurement sub-models.
        """
        out=tf.stack([self.model_list[vie](inputs[vie]) for vie in range(len(self.model_list))],axis=2) 
        
        scale_fact = tf.cast(self.tot_num/tf.shape(out)[0],dtype=float) #
        out = tf.divide(out,tf.math.sqrt(tf.math.multiply(scale_fact,tf.math.reduce_sum(tf.math.square(out),axis=0))))  ## re-normlise latent factors, very important!
        
        
        return out
    
    
    def train_step(self, inputs):
        
        """ Here, we overwrite the model train step, which is called 
        during model-fit. We train the Deep-PLS model by alternately iterating 
        through data-views. For each data-view, the global model loops through
        the sub-models (measurement models) in each data-view. For each of the
        sub-models, training is carried out in order to minimise the sum of 
        squared losses between the latent variable in the view of interest,
        and other data-views.
        
        """
       
        ## inputs is passed by tensorflow as a list of lists, this must be unpacked
        inputs=inputs[0]
        
        # Here, we run the current data-iteration through the global model in a forward 
        # pass. We do this so that we can re-normalise the weights. 
        
        y = self(inputs, training=False)  ## forward pass
       # y = self(inputs, training=True)  ## forward pass
    
        scale_fact = tf.cast(self.tot_num/tf.shape(y)[0],dtype=float) #
        
        ## Here, we re-normalise targets. In the case of the zca approach, this normalisation also involves
        # an orthogonalisation step
        if self.orthogonalisation=='zca':
            ylist = [None]*len(inputs)
            for i in range(y.shape[2]):
                moving_conv2 = self.model_list[i].layers[-1].moving_conv2
                diag_offset = self.model_list[i].layers[-1].diag_offset
                sqrt_inv_y = tf.linalg.sqrtm(tf.linalg.inv(moving_conv2+diag_offset*tf.eye(moving_conv2.shape[0])))
                ylist[i]=tf.matmul(tf.squeeze(y[:,:,i]),sqrt_inv_y)
            y=tf.stack(ylist,axis=2)
        elif self.orthogonalisation=='Moore-Penrose':
            y = tf.divide(y,tf.math.sqrt(tf.math.multiply(scale_fact,tf.math.reduce_sum(tf.math.square(y),axis=0))))  ## re-normlise latent factors, very important!
     
        total_loss = [None]*(len(inputs))
        total_CC = [None]*(len(inputs))
        total_mse = [None]*(len(inputs))
        
        ## Iterate through training data-views
        for vie in range(len(inputs)):
           
        
            with tf.GradientTape() as tape:
                
                ## forward pass
                y_pred = self.model_list[vie](inputs[vie], training=True)
                
                mse_loss = self.mse_loss(y, y_pred, vie)
                
                internal_loss = self.model_list[vie].losses
                
                # # Compute the loss for the data-view in question
                loss = mse_loss + internal_loss
              
            
            # Compute gradients
            trainable_vars = self.model_list[vie].trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            
            # Update weights
            self.model_list[vie].optimizer.apply_gradients(zip(gradients, trainable_vars))
            
            corr_metric=self.corr_metric(y,y_pred,vie)
            
            ## add current losses and metrics to the global lists
            total_loss[vie]=tf.math.reduce_sum(loss)
            total_CC[vie]=corr_metric
            total_mse[vie]=mse_loss
            
        # Update losses and metrics
        self.loss_tracker_total.update_state(tf.stack(total_loss))
        self.corr_tracker.update_state(tf.stack(total_CC))
        self.loss_tracker_mse.update_state(tf.stack(total_mse))
        
        
        return {"total_loss": self.loss_tracker_total.result(), "cross_metric": self.corr_tracker.result(), "mse_loss":self.loss_tracker_mse.result()}

    def global_build(self):
        """ This funnction is called when the structural model is initialised.
        This function uses global parameters to initialise measurement models
        with the corect number of dimensions to extract, and correct number of 
        samples"""
        
        for vie in range(len(self.model_list)):
            
            # if vie==0:
            #     self.model_list[vie].global_build(self.tot_num, self.ndims, 'None')
            # else:
            self.model_list[vie].global_build(self.tot_num, self.ndims, self.orthogonalisation)
        

    def compile(self, optimizer):
        """ Here, we overwrite the model compilation step. This is necessary as
        normally, the model compilation step would normally take a loss. Using
        this method, the loss is built into the method itself. We can either 
        pass the optimizer a single optimizer object, or a list of objects, with a 
        different optimizer used for each data-view.
        """
        
        super().compile()
        
        self.global_build()
        
        if isinstance(optimizer, list):
            for vie in range(len(self.model_list)):
                self.model_list[vie].compile(optimizer[vie])
        elif isinstance(optimizer,tf.keras.optimizers.Optimizer):
            for vie in range(len(self.model_list)):
                self.model_list[vie].compile(optimizer)
        else:
            print('Error: optimizer must either be of the tf.keras.optimizer class, or a list of objects of this class')
        

    def test_step(self, inputs):
        
        """ This step is called by model.evaluate() on a batch-wise level. This function
        returns loss metrics for the test data.
        
        """
        
        ## inputs is passed by tensorflow as a list of lists, this must be unpacked
        inputs=inputs[0]
        
        y = self(inputs, training=False)  ## forward pass
    
        total_loss = [None]*(len(inputs))
        total_CC = [None]*(len(inputs))
        total_mse = [None]*(len(inputs))
        
        ## Iterate through training data-views
        for vie in range(len(inputs)):
          
                
            ## forward pass
            y_pred = self.model_list[vie](inputs[vie], training=False)
            
            mse_loss = self.mse_loss(y, y_pred, vie)
            internal_loss = self.model_list[vie].losses
            
            # Compute the loss for the data-view in question
            loss = mse_loss + internal_loss
    
            corr_metric=self.corr_metric(y,y_pred,vie)
            
            ## add current losses and metrics to the global lists
            total_loss[vie]=tf.math.reduce_sum(loss)
            total_CC[vie]=corr_metric
            total_mse[vie]=mse_loss
            
           
        # Update losses and metrics
        self.loss_tracker_total.update_state(total_loss)
        self.corr_tracker.update_state(total_CC)
        self.loss_tracker_mse.update_state(total_mse)
            
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.

        return [self.loss_tracker_total, self.corr_tracker, self.loss_tracker_mse]

        
    def mse_loss(self,y_true,y_pred,vie):
        
        """ This function returns the mean squared error loss between the latent
        factors in a particular data-view, and the latent factors to which that
        data-view is connected via the global PLS model.
        """
        
        y_true =  tf.squeeze(tf.gather(y_true,tf.where(self.C_mv[vie,:]),axis=2),axis=3) ## select the latent factors connected to the latent factor for view vie
        
        y_pred = tf.expand_dims(y_pred,axis=2) ## expand dimensions of the predicted latent factor so broadcasting is possible
        
        return tf.reduce_mean(tf.math.reduce_sum(tf.math.square(tf.subtract(y_true,y_pred)),axis=0))
    
    def corr_metric(self,y_true,y_pred,vie):
        
        """ This function returns the mean correlation between the latent factors
        in a data-view, and the latent factors to which that data-view is connected 
        via the global PLS model.
        
        """
      
        y_true =  tf.squeeze(tf.gather(y_true,tf.where(self.C_mv[vie,:]),axis=2),axis=3) ## select the latent factors connected to the latent factor for view vie
        
        ## Minus the mean
        y_true_mean = tf.subtract(y_true,tf.math.reduce_mean(y_true,axis=0))
        y_pred_mean = tf.subtract(y_pred,tf.math.reduce_mean(y_pred,axis=0))
        
        # # ## Normalise matrices
        y_true_norm = tf.divide(y_true_mean,tf.norm(y_true_mean,axis=0))
        y_pred_norm = tf.divide(y_pred_mean,tf.norm(y_pred_mean,axis=0))
        
        y_pred_norm = tf.expand_dims(y_pred_norm,axis=2) ## expand dimensions of the predicted latent factor so broadcasting is possible
        
        corr2=tf.math.reduce_sum(tf.math.multiply(y_true_norm, y_pred_norm),axis=0)

        return tf.math.reduce_mean(corr2)


        