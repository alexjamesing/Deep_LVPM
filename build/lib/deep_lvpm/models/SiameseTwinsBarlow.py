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
from deep_lvpm.layers.ConfoundLayer import ConfoundLayer
import pydot

# from Custom_Losses_and_Metrics import mse_loss
# from Custom_Losses_and_Metrics import corr_metric

# Set up metrics trackers
loss_tracker_total = tf.keras.metrics.Mean(name="total_loss")
loss_tracker_mse = tf.keras.metrics.Mean(name="mean_squared_loss")
corr_tracker = tf.keras.metrics.Mean(name="corr_metric")

def mean_absolute_correlation(y):
    # Calculate the correlation matrix between features
    y_centered = y - tf.reduce_mean(y, axis=0, keepdims=True)
    covariance_matrix = tf.matmul(y_centered, y_centered, transpose_a=True) / tf.cast(tf.shape(y)[0] - 1, tf.float32)
    
    # Standard deviations of each feature
    stddevs = tf.sqrt(tf.linalg.diag_part(covariance_matrix))
    stddev_matrix = tf.expand_dims(stddevs, axis=0) * tf.expand_dims(stddevs, axis=1)
    
    # Correlation matrix (avoid division by zero)
    correlation_matrix = tf.where(stddev_matrix != 0, covariance_matrix / stddev_matrix, tf.zeros_like(covariance_matrix))
    
    # Compute mean absolute correlation, excluding diagonal
    num_features = tf.shape(correlation_matrix)[0]
    mask = tf.ones_like(correlation_matrix, dtype=tf.bool)  # Boolean mask
    mask = tf.linalg.set_diag(mask, tf.zeros(num_features, dtype=tf.bool))  # Exclude diagonal
    mean_abs_corr = tf.reduce_mean(tf.abs(tf.boolean_mask(correlation_matrix, mask)))
    
    return mean_abs_corr

@tf.keras.utils.register_keras_serializable(package="deep_lvpm",name="SiameseTwins")
class SiameseTwinsBarlow(tf.keras.Model):
    
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

    
    def __init__(self, model, regularizer, tot_num, ndims, orthogonalization='zca', momentum=0.95, epsilon=1e-4, diag_offset=1e-3, train_DLV=False, siamese_type = 'DLVPM', run_from_config=False, **kwargs):
        
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
            #y = self.model.layers[-1].weight_normalizer([y, scale_fact]) ## Normalize weights and return normalized output (last layer of model)

            mean_corr = mean_absolute_correlation(y)

            tf.print('mean self-correlation is:' + str(mean_corr))

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
            self.model.optimizer.apply_gradients(zip(gradients, trainable_vars))
            
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

    def compile(self, optimizer):
        """ Here, we overwrite the model compilation step. This is necessary as
        normally, the model compilation step would normally take a loss. Using
        this method, the loss is built into the method itself. We can either 
        pass the optimizer a single optimizer object, or a list of objects, with a 
        different optimizer used for each data-view.
        """
        
        super().compile()

        self.model.compile(optimizer)
        
   

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

        
    def mse_loss(self, y_true, y_pred, lambda_param=1e-4):
        # y_true and y_pred are not used because the loss is computed
        # directly from the output embeddings of two augmented views.
        #tf.print(y_pred.shape)
        # Split the predictions into two sets of embeddings
        
        z_a = y_true
        z_b = y_pred

        # Normalize the embeddings along the batch dimension
        z_a_norm = (z_a - tf.reduce_mean(z_a, axis=0)) / tf.math.reduce_std(z_a, axis=0)
        z_b_norm = (z_b - tf.reduce_mean(z_b, axis=0)) / tf.math.reduce_std(z_b, axis=0)

        # Compute the cross-correlation matrix
        
        N = tf.cast(tf.shape(z_a)[0], tf.float32)

        c = tf.matmul(tf.squeeze(z_a_norm), z_b_norm, transpose_a=True) / N

        # Make the cross-correlation matrix as close to the identity matrix as possible
        c_diff = c - tf.eye(num_rows=tf.shape(c)[0], num_columns=tf.shape(c)[1])
        loss = tf.reduce_sum(c_diff**2) - lambda_param * tf.reduce_sum(tf.linalg.diag_part(c_diff)**2)
        return loss
    
    
    

    
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
    
    # def get_compile_config(self):
    #     """
    #     Serializes the optimizer configurations of the models.

    #     Returns:
    #         dict: A dictionary containing the serialized optimizer configurations of the models.
    #     """
    #     return {
    #         "model_optimizers": [tf.keras.utils.serialize_keras_object(model.optimizer) for model in self.model_list]
    #     }
    
    # def compile_from_config(self, config):
    #     """
    #     Compiles the models with the deserialized optimizer configurations.

    #     Args:
    #         config (dict): A dictionary containing the serialized optimizer configurations.
    #     """
    #     optimizer_list = [tf.keras.utils.deserialize_keras_object(optimizer_config) for optimizer_config in config["model_optimizers"]]
    #     self.compile(optimizer_list)

    def build_from_config(self, config):
        """ build is overwritten here as it is not needed. Individual measurement models
        are built seperately, this happens when tf.keras.saving.deserialize_keras_object is called
        on models in model_list"""

        return
        #self.build(config["input_shape"])
    








