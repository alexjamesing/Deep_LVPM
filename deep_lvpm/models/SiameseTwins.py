#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script creates a custom Keras/TensorFlow model for identifying correlated factors
(deep latent variables) between different data types. It is designed to work with different
data-views, and it establishes associations between these views using deep latent
variables. The data-views we wish to optimise associations between are defined using an 
adjacency matrix.
"""

import os
import tensorflow as tf
import numpy as np
# import deep_lvpm 
from deep_lvpm.layers.FactorLayer import FactorLayer
from deep_lvpm.layers.ZCALayer import ZCALayer
import pydot

# Set up metrics trackers
loss_tracker_total = tf.keras.metrics.Mean(name="total_loss")
loss_tracker_mse = tf.keras.metrics.Mean(name="mean_squared_loss")
corr_tracker = tf.keras.metrics.Mean(name="corr_metric")

@tf.keras.utils.register_keras_serializable(package="deep_lvpm",name="SiameseTwins")
class SiameseTwins(tf.keras.Model):
    
    """
    A custom Keras model to establish associations between different data-views.

    This model implements a deep learning approach to find deep latent variables (DLVs)
    that highlight the correlated factors between different types of data.
    The associations between data-views are defined using a binary adjacency matrix,
    where ones represent connections, and zeros represent un-connected data-views.

    Attributes:
      
        model: The keras/tensorflow model to be trained.
        tot_num: Total number of features across all batches.
        ndims: Number of orthogonal latent variables to construct.
        epochs: Number of training epochs.
        batch_size: Size of the batches used during training.
        orthogonalization: Orthogonalisation procedure ('zca' or 'Moore-Penrose').
        loss_tracker_total: Tracker for the total loss during training.
        corr_tracker: Tracker for the correlation metric during training.
        loss_tracker_mse: Tracker for the mean squared error loss during training.

    Methods:
        call: Runs data through each of the measurement sub-models.
        train_step: Performs a training step, updating the model weights.
        compile: Configures the model for training.
        test_step: Evaluates the model on a batch of test data.
        metrics: Returns the list of model's metrics.
        mse_loss: Calculates mean squared error loss for a data-view.
        corr_metric: Calculates the correlation metric for a data-view.
    """

    
    def __init__(self, model, regularizer, tot_num, ndims, orthogonalization='Moore-Penrose', momentum=0.95, epsilon=1e-4, diag_offset=1e-3, train_DLV=False, siamese_type = 'DLVPM', run_from_config=False, **kwargs):
        
        """
        Initializes the SiameseTwins instance.

        Args:
            model: A Keras/Tensorflow model to be trained
            regularizer: A regularizer to add to the projection layer of the network
            tot_num (int): Total number of features across all batches.
            ndims (int): Number of orthogonal latent variables to construct.
            orthogonalization (str, optional): Orthogonalisation procedure. Defaults to 'Moore-Penrose'.
            momentum (Float, optional): The momentum defines how quickly global parameters such as means and correlation matrices are updated
            epsilon (Float, optional): "epsilon" (often denoted as ε) is a small constant added for numerical stability in batch updates
            train_DLV (True/False): "train_DLV" defines whether target DLVs are calcualted in training or testing modes during model training
            siamese_type: This is the kind of siamese model to run. Default is DLVPM. The script can also run VicReg and BarlowTwins.
        """

        super().__init__(**kwargs)    
    
        self.model = model
        self.tot_num = tot_num
        self.ndims = ndims
        self.momentum = momentum
        self.epsilon = epsilon
        self.orthogonalization = orthogonalization
        self.regularizer = regularizer
        self.train_DLV = train_DLV
        self.siamese_type = siamese_type
        self.diag_offset = diag_offset

        if not run_from_config:
        # Add factor layer to each model in the list
            self.add_DLVPM_layer(regularizer)
    
        self.loss_tracker_total = tf.keras.metrics.Mean(name="total_loss")
        self.corr_tracker = tf.keras.metrics.Mean(name="cross_metric")
        self.loss_tracker_mse = tf.keras.metrics.Mean(name="mse_loss")

    
    def add_DLVPM_layer(self, regularizer):
        """
        Adds a FactorLayer on top of the given model.

        The method first checks whether the input model is sequential or functional,
        and then adds the FactorLayer in an appropriate way.

        :param model: A Keras/TensorFlow model (sequential or functional).
        :return: The model with an added FactorLayer on top.
        """
        if isinstance(self.model, tf.keras.Sequential):
            if self.orthogonalization == 'Moore-Penrose':
                self.model.add(FactorLayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon))
            elif self.orthogonalization == 'zca':
                self.model.add(ZCALayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon, diag_offset = self.diag_offset))
            else:
                print('Orthogonalization mode not recognised, must be "Moore-Penrose" or "zca"')
        elif isinstance(self.model, tf.keras.Model):
            if self.orthogonalization == 'Moore-Penrose':
                x = FactorLayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon)(self.model.output)
                self.model = tf.keras.Model(inputs=self.model.input, outputs=x)
            elif self.orthogonalization == 'zca':
                x = ZCALayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon, diag_offset = self.diag_offset)(self.model.output)
                self.model = tf.keras.Model(inputs=self.model.input, outputs=x)
            else:
                print('Orthogonalization mode not recognised, must be "Moore-Penrose" or "zca"')
        else:
            raise ValueError("The input model must be either a tf.keras.Sequential or a tf.keras.Model instance.")

    

    def organize_inputs_by_model(self, data_inputs):
    
        organized_inputs = []
        data_index = 0

        for vie in range(2):
            
            num_inputs = len(self.model.inputs) if hasattr(self.model, 'inputs') else 1

            if num_inputs == 1:
                # For a single input model, append the data directly.
                organized_inputs.append(data_inputs[data_index])
                data_index += 1
            else:
                # For models requiring multiple inputs, append a list of inputs.
                inputs_for_model = data_inputs[data_index:data_index + num_inputs]
                organized_inputs.append(inputs_for_model)
                data_index += num_inputs

        return organized_inputs

    
    def call(self, inputs, training=False):
        """
        Run data through each of the measurement sub-models.

        Args:
            inputs (list): A list of inputs for each data-view.
            training: Whether to call the model in training or inference mode. Can take values of True or False.

        Returns:
            tf.Tensor: The output of the model after processing the inputs.
        """

        out = self.model(inputs)

        return out


    def train_step(self, inputs):
        
        """
        Perform a training step, updating the model weights.

        Args:
            inputs (list or tuple): A list of inputs for each data-view.

        Returns:
            dict: A dictionary containing the total loss, cross metric, and mean squared error loss.
        """
       
        ## tensorflow packs inputs in another tuple, this should be unpacked
        inputs=inputs[0]
        
        total_loss = [None]*2
        total_CC = [None]*2
        total_mse = [None]*2

        inputs_nested = self.organize_inputs_by_model(inputs) 

        ## Iterate through training data-views
        for vie in range(2):

             # Here, we run the current data-iteration through the model in a forward pass
            y = self(inputs_nested[1 if vie == 0 else 0], training=self.train_DLV)  ## forward pass

            ## Here, we re-normalise the model weights
            scale_fact = tf.cast(self.tot_num/tf.shape(y)[0],dtype=float) # scale factor for re-scaling
            y = self.model.layers[-1].weight_normalizer([y, scale_fact]) ## Normalize weights and return normalized output (last layer of model)

            with tf.GradientTape() as tape:
                
                ## forward pass
                y_pred = self.model(inputs_nested[vie], training=True)

                y_pred = tf.divide(y_pred,tf.math.multiply(tf.math.sqrt(scale_fact),tf.norm(y_pred,axis=0))) ## Here, we re-normalize DLVs

                mse_loss = self.mse_loss(y, y_pred)
                
                internal_loss = self.model.losses
                
                # # Compute the loss for the data-view in question
                loss = mse_loss + internal_loss
            
            # Compute gradients
            trainable_vars = self.model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            
            corr_metric=self.corr_metric(y,y_pred)
            
            ## add current losses and metrics to the global lists
            total_loss[vie]=tf.math.reduce_sum(loss)
            total_CC[vie]=corr_metric
            total_mse[vie]=mse_loss

        # Update losses and metrics
        self.loss_tracker_total.update_state(tf.stack(total_loss))
        self.corr_tracker.update_state(tf.stack(total_CC))
        self.loss_tracker_mse.update_state(tf.stack(total_mse))
        
        return {"total_loss": self.loss_tracker_total.result(), "cross_metric": self.corr_tracker.result(), "mse_loss":self.loss_tracker_mse.result()}

    def compile(self, optimizer, **kwargs):
        # compile 
        super().compile(optimizer=optimizer, **kwargs)
   

    def test_step(self, inputs):
        
        """ This step is called by model.evaluate() on a batch-wise level. This function
        returns loss metrics for the test data.
        
        """
        
        ## tensorflow packs inputs in another tuple, this should be unpacked
        inputs=inputs[0]
        
        total_loss = [None]*2
        total_CC = [None]*2
        total_mse = [None]*2

        inputs_nested = self.organize_inputs_by_model(inputs) 

        ## Iterate through training data-views
        for vie in range(2):

             # Here, we run the current data-iteration through the model in a forward pass
            y = self(inputs_nested[1 if vie == 0 else 0], training=self.train_DLV)  ## forward pass

            ## Here, we re-normalise the model weights
            scale_fact = tf.cast(self.tot_num/tf.shape(y)[0],dtype=float) # scale factor for re-scaling
            y = self.model.layers[-1].weight_normalizer([y, scale_fact]) ## Normalize weights and return normalized output (last layer of model)
            
            
            ## forward pass
            y_pred = self.model(inputs_nested[vie], training=True)

            y_pred = tf.divide(y_pred,tf.math.multiply(tf.math.sqrt(scale_fact),tf.norm(y_pred,axis=0))) ## Here, we re-normalize DLVs

            mse_loss = self.mse_loss(y, y_pred)
            
            internal_loss = self.model.losses
            
            # # Compute the loss for the data-view in question
            loss = mse_loss + internal_loss
        
            corr_metric=self.corr_metric(y,y_pred)
            
            ## add current losses and metrics to the global lists
            total_loss[vie]=tf.math.reduce_sum(loss)
            total_CC[vie]=corr_metric
            total_mse[vie]=mse_loss

        # Update losses and metrics
        self.loss_tracker_total.update_state(tf.stack(total_loss))
        self.corr_tracker.update_state(tf.stack(total_CC))
        self.loss_tracker_mse.update_state(tf.stack(total_mse))

           
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.

        return [self.loss_tracker_total, self.corr_tracker, self.loss_tracker_mse]

        
    def mse_loss(self,y_true,y_pred):
        
        """ This function returns the mean squared error loss between the latent
        factors in a particular data-view, and the latent factors to which that
        data-view is connected via the global DLVPM model.
        """
        
        #y_true =  tf.squeeze(tf.gather(y_true,tf.where(self.Path[vie,:]),axis=2),axis=3) ## select the latent factors connected to the latent factor for view vie
        
        #y_pred = tf.expand_dims(y_pred,axis=2) ## expand dimensions of the predicted latent factor so broadcasting is possible
        
        mse_loss = tf.divide(tf.reduce_sum(tf.math.reduce_mean(tf.math.square(tf.subtract(y_true,y_pred)),axis=0)),2)

        return mse_loss
    
    def corr_metric(self,y_true,y_pred):
        
        """ This function returns the mean correlation between the latent factors
        in a data-view, and the latent factors to which that data-view is connected 
        via the global DLVPM model.
        
        """
      
        #y_true =  tf.squeeze(tf.gather(y_true,tf.where(self.Path[vie,:]),axis=2),axis=3) ## select the latent factors connected to the latent factor for view vie
        
        ## Minus the mean
        y_true_mean = tf.subtract(y_true,tf.math.reduce_mean(y_true,axis=0))
        y_pred_mean = tf.subtract(y_pred,tf.math.reduce_mean(y_pred,axis=0))
        
        # # ## Normalise matrices
        y_true_norm = tf.divide(y_true_mean,tf.norm(y_true_mean,axis=0))
        y_pred_norm = tf.divide(y_pred_mean,tf.norm(y_pred_mean,axis=0))
        
        #y_pred_norm = tf.expand_dims(y_pred_norm,axis=2) ## expand dimensions of the predicted latent factor so broadcasting is possible
        
        corr2=tf.math.reduce_sum(tf.math.multiply(y_true_norm, y_pred_norm),axis=0)

        return tf.math.reduce_mean(corr2)

    
    def get_config(self):

        """
        Gets configuration of the model for serialization.

        Returns:
            Dictionary containing the configuration of the model.
        """
        base_config = super().get_config()

        config = {
            "model": tf.keras.utils.serialize_keras_object(self.model),  # Include serialized model list in the configuration
            "regularizer": tf.keras.utils.serialize_keras_object(self.regularizer),
            "tot_num": self.tot_num,
            "ndims": self.ndims,  
            "orthogonalization": self.orthogonalization
        }
    
        return {**base_config, **config}
    
    @classmethod    
    def from_config(cls, config):
        """
        Creates an instance of the class from a config dictionary.

        Args:
            config (dict): A dictionary containing the configuration of the instance.

        Returns:
            An instance of the class.
        """
        # Deserialize Keras/TensorFlow objects
       
        # Deserialize each model in the model list using a list comprehension
        config['model'] = tf.keras.utils.deserialize_keras_object(config['model']) 
        config['run_from_config'] = True
        
        # If regularization is present in the config, deserialize it
        config['regularizer'] = tf.keras.utils.deserialize_keras_object(config['regularizer']) 
        
        return cls(**config)
    
#  def get_config(self):
#         base = super().get_config()
#         return {
#             **base,
#             "model": serialize_keras_object(self.model),
#             "regularizer": serialize_keras_object(self.regularizer),
#             "tot_num": self.tot_num,
#             "ndims": self.ndims,
#             "orthogonalization": self.orthogonalization,
#             "momentum": self.momentum,
#             "epsilon": self.epsilon,
#             "diag_offset": self.diag_offset,
#             "train_DLV": self.train_DLV,
#             "siamese_type": self.siamese_type,
#         }

#     @classmethod
#     def from_config(cls, config):
#         model = deserialize_keras_object(config.pop("model"))
#         regularizer = deserialize_keras_object(config.pop("regularizer"))
#         config["run_from_config"] = True
#         return cls(model=model, regularizer=regularizer, **config)



    def build_from_config(self, config):
        """ build is overwritten here as it is not needed. Individual measurement models
        are built seperately, this happens when tf.keras.saving.deserialize_keras_object is called
        on models in model_list"""

        return
        #self.build(config["input_shape"])
    








