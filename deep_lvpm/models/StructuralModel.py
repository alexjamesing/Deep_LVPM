#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script creates a custom Keras model for identifying correlated factors
(deep latent variables) between different data types. It is designed to work with different
data-views, and it establishes associations between these views using deep latent
variables. The data-views we wish to optimise associations between are defined using an 
adjacency matrix.
"""

import os
import numpy as np
import keras as keras
from deep_lvpm.layers.FactorLayer import FactorLayer
from deep_lvpm.layers.ZCALayer import ZCALayer
from deep_lvpm.layers.ConfoundLayer import ConfoundLayer
import pydot
from keras import ops
import tensorflow as tf

# Set up metrics trackers
loss_tracker_total = keras.metrics.Mean(name="total_loss")
loss_tracker_mse = keras.metrics.Mean(name="mean_squared_loss")
corr_tracker = keras.metrics.Mean(name="corr_metric")


@keras.utils.register_keras_serializable(package="deep_lvpm",name="StructuralModel")
class StructuralModel(keras.Model):
    
    """
    A custom Keras model to establish associations between different data-views.

    This model implements a deep learning approach to find deep latent variables (DLVs)
    that highlight the correlated factors between different types of data.
    The associations between data-views are defined using a binary adjacency matrix,
    where ones represent connections, and zeros represent un-connected data-views.

    Attributes:
        Path: A binary adjacency matrix defining the connections between data-views.
        model_list: A list of Keras models for each data-view.
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

    
    def __init__(self, Path, model_list, regularizer_list, tot_num, ndims, orthogonalization='Moore-Penrose', momentum=0.95, epsilon=1e-4, train_DLV=False, run_from_config=False, **kwargs):
        
        """
        Initializes the StructuralModel instance.

        Args:
            Path (tf.Tensor or np.array): A binary adjacency matrix defining connections between data-views.
            regularizer_list (list): A list of regularizers that are applied to projection layers for models
            in each data-view.
            model_list (list): A list of Keras models for each data-view.
            tot_num (int): Total number of features across all batches.
            ndims (int): Number of orthogonal latent variables to construct.
            orthogonalization (str, optional): Orthogonalisation procedure. Defaults to 'Moore-Penrose'.
            momentum (Float, optional): The momentum defines how quickly global parameters such as means and correlation matrices are updated
            epsilon (Float, optional): "epsilon" (often denoted as ε) is a small constant added for numerical stability in batch updates
            train_DLV (True/False): "train_DLV" defines whether target DLVs are calcualted in training or testing modes during model training
        """

        super().__init__(**kwargs)    
        
        self.Path = Path
        self.tot_num = tot_num
        self.ndims = ndims
        self.momentum = momentum
        self.epsilon = epsilon
        self.orthogonalization=orthogonalization
        self.regularizer_list = regularizer_list
        self.train_DLV = train_DLV

        if not run_from_config:
        # Add factor layer to each model in the list
            self.model_list = [self.add_DLVPM_layer(model, regularizer) for model, regularizer in zip(model_list, regularizer_list)]
        else:
            self.model_list = model_list

        self.loss_tracker_total = keras.metrics.Mean(name="total_loss")
        self.corr_tracker = keras.metrics.Mean(name="cross_metric")
        self.loss_tracker_mse = keras.metrics.Mean(name="mse_loss")

    
    def add_DLVPM_layer(self, model, regularizer):
        """
        Adds a FactorLayer on top of the given model.

        The method first checks whether the input model is sequential or functional,
        and then adds the FactorLayer in an appropriate way.

        :param model: A Keras/TensorFlow model (sequential or functional).
        :return: The model with an added FactorLayer on top.
        """
        if isinstance(model, keras.Sequential):
            if self.orthogonalization == 'Moore-Penrose':
                model.add(FactorLayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon))
            elif self.orthogonalization == 'zca':
                model.add(ZCALayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon))
            else:
                print('Orthogonalization mode not recognised, must be "Moore-Penrose" or "zca"')
        elif isinstance(model, keras.Model):
            if self.orthogonalization == 'Moore-Penrose':
                x = FactorLayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon)(model.output)
                model = keras.Model(inputs=model.input, outputs=x)
            elif self.orthogonalization == 'zca':
                x = ZCALayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon)(model.output)
                model = keras.Model(inputs=model.input, outputs=x)
            else:
                print('Orthogonalization mode not recognised, must be "Moore-Penrose" or "zca"')
        else:
            raise ValueError("The input model must be either a keras.Sequential or a keras.Model instance.")

        
        return model
    
    # def call(self, inputs, training=False):
    #     """
    #     Run data through each of the measurement sub-models.

    #     Args:
    #         inputs (list): A list of inputs for each data-view.
    #         training: Whether to call the model in training or inference mode. Can take values of True or False.

    #     Returns:
    #         tf.Tensor: The output of the model after processing the inputs.
    #     """

    #     inputs_nested = self.organize_inputs_by_model(inputs) ## this function organises flat inputs into a list of lists, which makes model training easier

    #     out=tf.stack([self.model_list[vie](inputs_nested[vie], training = training) for vie in range(len(self.model_list))],axis=2) ## Stack the outputs 
    
    #     return out

    def call(self, inputs, training=False):

        """
    #     Run data through each of the measurement sub-models.

    #     Args:
    #         inputs (list): A list of inputs for each data-view.
    #         training: Whether to call the model in training or inference mode. Can take values of True or False.

    #     Returns:
    #         The output of the model after processing the inputs.
    #     """


        inputs_nested = self.organize_inputs_by_model(inputs)
        out = ops.stack(
            [self.model_list[vie](inputs_nested[vie], training=training)
            for vie in range(len(self.model_list))],
            axis=2
            )
        return out


    def organize_inputs_by_model(self, data_inputs):
        organized_inputs = []
        data_index = 0

        for model in self.model_list:
            
            num_inputs = len(model.inputs) if hasattr(model, 'inputs') else 1

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


    # def train_step(self, inputs):
        
    #     """
    #     Perform a training step, updating the model weights.

    #     Args:
    #         inputs (list or tuple): A list of inputs for each data-view.

    #     Returns:
    #         dict: A dictionary containing the total loss, cross metric, and mean squared error loss.
    #     """
       
    #     ## tensorflow packs inputs in another tuple, this should be unpacked
    #     inputs=inputs[0]
        
    #     # Here, we run the current data-iteration through the global model in a forward 
    #     y = self(inputs, training=self.train_DLV)  ## forward pass

    #     ## Here, we re-normalise the model weights
    #     scale_fact = tf.cast(self.tot_num/tf.shape(y)[0],dtype=float) # scale factor for re-scaling

    #     y_list = []
    #     for vie in range(len(self.model_list)):
    #         y_view = y[:,:,vie] ## This is the current view under analysis
    #         y_view = self.model_list[vie].layers[-1].weight_normalizer([y_view, scale_fact]) ## Normalize weights and return normalized output (last layer of model)
    #         y_list.append(y_view) ## append normalized output to list
    #     y = tf.stack(y_list, axis=-1) ## normalized data output
            

    #     total_loss = [None]*(len(self.model_list))
    #     total_CC = [None]*(len(self.model_list))
    #     total_mse = [None]*(len(self.model_list))
        
    #     inputs_nested = self.organize_inputs_by_model(inputs) ## this function organises flat inputs into a list of lists, which makes model training easier

    #     ## Iterate through training data-views
    #     for vie in range(len(self.model_list)):

    #         with tf.GradientTape() as tape:
                
    #             ## forward pass
    #             y_pred = self.model_list[vie](inputs_nested[vie], training=True)

    #             y_pred = tf.divide(y_pred,tf.math.multiply(tf.math.sqrt(scale_fact),tf.norm(y_pred,axis=0))) ## Here, we re-normalize DLVs

    #             mse_loss = self.mse_loss(y, y_pred, vie)
                
    #             internal_loss = self.model_list[vie].losses
                
    #             # # Compute the loss for the data-view in question
    #             loss = mse_loss + internal_loss
            
            
    #         # Compute gradients
    #         trainable_vars = self.model_list[vie].trainable_variables
    #         gradients = tape.gradient(loss, trainable_vars)
            
    #         # Update weights
    #         self.model_list[vie].optimizer.apply_gradients(zip(gradients, trainable_vars))
            
    #         corr_metric=self.corr_metric(y,y_pred,vie)
            
    #         ## add current losses and metrics to the global lists
    #         total_loss[vie]=tf.math.reduce_sum(loss)
    #         total_CC[vie]=corr_metric
    #         total_mse[vie]=mse_loss
                
    #     # Update losses and metrics
    #     self.loss_tracker_total.update_state(tf.stack(total_loss))
    #     self.corr_tracker.update_state(tf.stack(total_CC))
    #     self.loss_tracker_mse.update_state(tf.stack(total_mse))
        
        
        # return {"total_loss": self.loss_tracker_total.result(), "cross_metric": self.corr_tracker.result(), "mse_loss":self.loss_tracker_mse.result()}

    def train_step(self, inputs):

        #     """
        #     Perform a training step, updating the model weights.

        #     Args:
        #         inputs (list or tuple): A list of inputs for each data-view.

        #     Returns:
        #         dict: A dictionary containing the total loss, cross metric, and mean squared error loss.
        #     """

        # Unpack (tf.data-like packs inputs in a tuple/list)
        inputs = inputs[0]

        # Forward pass through all views to construct global DLVs
        y = self(inputs, training=self.train_DLV)

        # scale_fact = tot_num / batch_size
        y_dtype = ops.dtype(y)
        scale_fact = ops.cast(self.tot_num, y_dtype) / ops.cast(ops.shape(y)[0], y_dtype)

        # Optional per-view normalization via last layer's weight_normalizer
        y_list = []
        for vie in range(len(self.model_list)):
            y_view = y[:, :, vie]
            y_view = self.model_list[vie].layers[-1].weight_normalizer([y_view, scale_fact])
            y_list.append(y_view)
        y = ops.stack(y_list, axis=-1)

        total_loss = [None] * len(self.model_list)
        total_CC = [None] * len(self.model_list)
        total_mse = [None] * len(self.model_list)

        inputs_nested = self.organize_inputs_by_model(inputs)

        for vie in range(len(self.model_list)):
            with tf.GradientTape() as tape:
                y_pred = self.model_list[vie](inputs_nested[vie], training=True)

                # y_pred / (sqrt(scale_fact) * ||y_pred||_2 over batch)
                # eps = ops.convert_to_tensor(self.epsilon, dtype=ops.dtype(y_pred))
                # denom = ops.sqrt(scale_fact) * ops.sqrt(ops.sum(ops.square(y_pred), axis=0) + eps)

                #y_pred = y_pred / denom

                y_pred = tf.divide(y_pred,tf.math.multiply(tf.math.sqrt(scale_fact),tf.norm(y_pred,axis=0))) ## Here, we re-normalize DLVs

                mse_loss = self.mse_loss(y, y_pred, vie)

                internal_loss = self.model_list[vie].losses
                # Keep your original addition semantics
                loss = mse_loss + internal_loss

            trainable_vars = self.model_list[vie].trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.model_list[vie].optimizer.apply_gradients(zip(gradients, trainable_vars))

            corr_metric = self.corr_metric(y, y_pred, vie)

            total_loss[vie] = ops.sum(loss)
            total_CC[vie] = corr_metric
            total_mse[vie] = mse_loss

        self.loss_tracker_total.update_state(ops.stack(total_loss))
        self.corr_tracker.update_state(ops.stack(total_CC))
        self.loss_tracker_mse.update_state(ops.stack(total_mse))

        return {
            "total_loss": self.loss_tracker_total.result(),
            "cross_metric": self.corr_tracker.result(),
            "mse_loss": self.loss_tracker_mse.result(),
        }
    

    def compile(self, optimizer):
        """ Here, we overwrite the model compilation step. This is necessary as
        normally, the model compilation step would normally take a loss. Using
        this method, the loss is built into the method itself. We can either 
        pass the optimizer a single optimizer object, or a list of objects, with a 
        different optimizer used for each data-view.
        """
        
        super().compile()
        
        #self.global_build()
        
        if isinstance(optimizer, list):
            for vie in range(len(self.model_list)):
                self.model_list[vie].compile(optimizer[vie])
        elif isinstance(optimizer,keras.optimizers.Optimizer):
            for vie in range(len(self.model_list)):
                self.model_list[vie].compile(optimizer)
        else:
            print('Error: optimizer must either be of the keras.optimizer class, or a list of objects of this class')
        

    # def test_step(self, inputs):
        
    #     """ This step is called by model.evaluate() on a batch-wise level. This function
    #     returns loss metrics for the test data.
        
    #     """
        
    #     ## tensorflow packs inputs in another tuple, this should be unpacked
    #     inputs=inputs[0]

    #     #inputs = self.organize_inputs_by_model(inputs)
        
    #     y = self(inputs, training=False)  ## forward pass
    
    #     total_loss = [None]*(len(self.model_list))
    #     total_CC = [None]*(len(self.model_list))
    #     total_mse = [None]*(len(self.model_list))
        
    #     inputs_nested = self.organize_inputs_by_model(inputs)
    #     ## Iterate through training data-views
    #     for vie in range(len(self.model_list)):
          
                
    #         ## forward pass
    #         y_pred = self.model_list[vie](inputs_nested[vie], training=False)
            
    #         mse_loss = self.mse_loss(y, y_pred, vie)
    #         internal_loss = self.model_list[vie].losses
            
    #         # Compute the loss for the data-view in question
    #         loss = mse_loss + internal_loss
    
    #         corr_metric=self.corr_metric(y,y_pred,vie)
            
    #         ## add current losses and metrics to the global lists
    #         total_loss[vie]=tf.math.reduce_sum(loss)
    #         total_CC[vie]=corr_metric
    #         total_mse[vie]=mse_loss
            
           
    #     # Update losses and metrics
    #     self.loss_tracker_total.update_state(total_loss)
    #     self.corr_tracker.update_state(total_CC)
    #     self.loss_tracker_mse.update_state(total_mse)
            
    #     return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        #     """ This step is called by model.evaluate() on a batch-wise level. This function
        #     returns loss metrics for the test data.
        #     """

        inputs = inputs[0]
        y = self(inputs, training=False)

        total_loss = [None] * len(self.model_list)
        total_CC = [None] * len(self.model_list)
        total_mse = [None] * len(self.model_list)

        inputs_nested = self.organize_inputs_by_model(inputs)

        for vie in range(len(self.model_list)):
            y_pred = self.model_list[vie](inputs_nested[vie], training=False)

            mse_loss = self.mse_loss(y, y_pred, vie)
            internal_loss = self.model_list[vie].losses
            loss = mse_loss + internal_loss

            corr_metric = self.corr_metric(y, y_pred, vie)

            total_loss[vie] = ops.sum(loss)
            total_CC[vie] = corr_metric
            total_mse[vie] = mse_loss

        # Keep your original update_state calls unchanged
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


    def mse_loss(self, y_true, y_pred, vie):
    
        """
        Mean squared error between y_pred (view vie) and the connected views in y_true.

        """
        # y_true: (batch, ndims, n_views)
        # y_pred: (batch, ndims)

        y_pred_exp = ops.expand_dims(y_pred, axis=2)  # (batch, ndims, 1)

        # Per-view squared error averaged over batch: (ndims, n_views)
        se_mean = ops.mean(ops.square(y_true - y_pred_exp), axis=0)

        # Mask to include only connected views
        mask = ops.cast(self.Path[vie, :], ops.dtype(se_mean))     # (n_views,)
        se_mean_masked = se_mean * ops.expand_dims(mask, axis=0)   # (ndims, n_views)

        mse_loss = ops.sum(se_mean_masked) / 2.0
        return mse_loss

        
    # def mse_loss(self,y_true,y_pred,vie):
        
    #     """ This function returns the mean squared error loss between the latent
    #     factors in a particular data-view, and the latent factors to which that
    #     data-view is connected via the global DLVPM model.
    #     """
        
    #     y_true =  tf.squeeze(tf.gather(y_true,tf.where(self.Path[vie,:]),axis=2),axis=3) ## select the latent factors connected to the latent factor for view vie
        
    #     y_pred = tf.expand_dims(y_pred,axis=2) ## expand dimensions of the predicted latent factor so broadcasting is possible
        
    #     mse_loss = tf.divide(tf.reduce_sum(tf.math.reduce_mean(tf.math.square(tf.subtract(y_true,y_pred)),axis=0)),2)

    #     return mse_loss



    def corr_metric(self, y_true, y_pred, vie):
        
        """
        Mean correlation between y_pred (view vie) and connected views in y_true.
        """

        eps = ops.convert_to_tensor(self.epsilon, dtype=ops.dtype(y_true))

        # Center over batch
        y_true_c = y_true - ops.mean(y_true, axis=0)   # (ndims, n_views) over batch
        y_pred_c = y_pred - ops.mean(y_pred, axis=0)   # (ndims) over batch

        denom_true = ops.sqrt(ops.sum(ops.square(y_true_c), axis=0) + eps)   # (ndims, n_views)
        denom_pred = ops.sqrt(ops.sum(ops.square(y_pred_c), axis=0) + eps)   # (ndims,)

        y_true_n = y_true_c / denom_true               # (batch, ndims, n_views)
        y_pred_n = y_pred_c / denom_pred               # (batch, ndims)

        y_pred_n = ops.expand_dims(y_pred_n, axis=2)   # (batch, ndims, 1)
        corr_mat = ops.sum(y_true_n * y_pred_n, axis=0)  # (ndims, n_views)

        # Mask only connected views
        mask = ops.cast(self.Path[vie, :], ops.dtype(corr_mat))    # (n_views,)
        corr_masked = corr_mat * ops.expand_dims(mask, axis=0)     # (ndims, n_views)

        # Average over dims and number of connected views
        n_conn = ops.sum(mask)
        n_conn_safe = ops.maximum(n_conn, ops.convert_to_tensor(1.0, dtype=ops.dtype(n_conn)))
        corr_mean = ops.sum(corr_masked) / (n_conn_safe * float(self.ndims))

        return corr_mean

    
    # def corr_metric(self,y_true,y_pred,vie):
        
    #     """ This function returns the mean correlation between the latent factors
    #     in a data-view, and the latent factors to which that data-view is connected 
    #     via the global DLVPM model.
        
    #     """
      
    #     y_true =  tf.squeeze(tf.gather(y_true,tf.where(self.Path[vie,:]),axis=2),axis=3) ## select the latent factors connected to the latent factor for view vie
        
    #     ## Minus the mean
    #     y_true_mean = tf.subtract(y_true,tf.math.reduce_mean(y_true,axis=0))
    #     y_pred_mean = tf.subtract(y_pred,tf.math.reduce_mean(y_pred,axis=0))
        
    #     # # ## Normalise matrices
    #     y_true_norm = tf.divide(y_true_mean,tf.norm(y_true_mean,axis=0))
    #     y_pred_norm = tf.divide(y_pred_mean,tf.norm(y_pred_mean,axis=0))
        
    #     y_pred_norm = tf.expand_dims(y_pred_norm,axis=2) ## expand dimensions of the predicted latent factor so broadcasting is possible
        
    #     corr2=tf.math.reduce_sum(tf.math.multiply(y_true_norm, y_pred_norm),axis=0)

    #     return tf.math.reduce_mean(corr2)

    # def calculate_corrmat(self, DLVs):
    #     """
    #     Compute Pearson correlation coefficient matrices for a 3D tensor.

    #     This function takes a 3D tensor of shape (n_samples, dimensions, DLVs) and computes
    #     the Pearson correlation coefficient between each pair of DLVs for each dimension. 
    #     The output is a list of symmetric matrices, one for each dimension, of shape (DLVs, DLVs).

    #     Args:
    #     DLVs (tf.Tensor): A 3D tensor of shape (n_samples, dimensions, DLVs).

    #     Returns:
    #     List[tf.Tensor]: A list of 2D tensors, each of shape (DLVs, DLVs), containing 
    #                     the Pearson correlation coefficients for each dimension.
    #     """
    #     # Ensure the input is a 3D tensor
    #     if len(DLVs.shape) != 3:
    #         raise ValueError("Input must be a 3D tensor")

    #     # List to store correlation matrices for each dimension
    #     correlation_matrices = []

    #     # Iterate through each dimension
    #     for dim in range(DLVs.shape[1]):
    #         # Select the data for the current dimension
    #         dim_DLVs = DLVs[:, dim, :]

    #         # Centering the DLVs by subtracting the mean
    #         mean_centered = dim_DLVs - tf.reduce_mean(dim_DLVs, axis=0)

    #         # Compute the standard deviation for each feature
    #         std_dev = tf.math.reduce_std(dim_DLVs, axis=0)

    #         # Normalize each feature
    #         normalized_DLVs = mean_centered / std_dev

    #         # Compute the correlation matrix for the current dimension
    #         correlation_matrix = tf.linalg.matmul(normalized_DLVs, normalized_DLVs, transpose_a=True) / tf.cast(tf.shape(dim_DLVs)[0], tf.float32)
    #         correlation_matrices.append(correlation_matrix)

    #     return correlation_matrices

    def calculate_corrmat(self, DLVs):
        
        """
        Compute Pearson correlation matrices for a 3D tensor using keras.ops.
        DLVs: (n_samples, dimensions, DLVs)
        Returns: list of (DLVs x DLVs) per dimension.
        """
        if len(DLVs.shape) != 3:
            raise ValueError("Input must be a 3D tensor")

        correlation_matrices = []
        n_samples = ops.cast(ops.shape(DLVs)[0], DLVs.dtype)
        eps = ops.convert_to_tensor(1e-7, dtype=DLVs.dtype)

        for dim in range(DLVs.shape[1]):
            dim_DLVs = DLVs[:, dim, :]  # (n_samples, n_feats)
            mean_centered = dim_DLVs - ops.mean(dim_DLVs, axis=0)
            std_dev = ops.std(dim_DLVs, axis=0) + eps
            normalized = mean_centered / std_dev
            correlation_matrix = ops.matmul(normalized, normalized, transpose_a=True) / n_samples
            correlation_matrices.append(correlation_matrix)

        return correlation_matrices

    

    def plot_structural_model(self, outputname):
        """
        This function plots the structural/path. model. Visualisation is quite simple. 
        Aesthetics are similar to those used in keras.utils.plot_model()
        outputname: This is the name of the output where we save the results. 

        """
        # Create a PyDot graph
        graph = pydot.Dot(graph_type='digraph', rankdir='TB')

        model_layer_list= [len(model.layers) for model in self.model_list]

        # Create nodes with labels
        for i in range(len(self.model_list)):

            label = "Measurement Model " + str(i) + "," + " " + str(model_layer_list[i]) + " layers"
            node = pydot.Node(str(i), label=label, shape="record") # create nodes to add to the pydot object
            graph.add_node(node) # add nodes to the pydot graph object

        adj_matrix = self.Path # this is the path. model we wish to plot

        # Create edges
        for i, row in enumerate(adj_matrix):
            for j, val in enumerate(row):
                if val == 1:
                    edge = pydot.Edge(str(i), str(j))
                    graph.add_edge(edge)

        graph.write_png(outputname)

            

    def get_config(self):

        """
        Gets configuration of the model for serialization.

        Returns:
            Dictionary containing the configuration of the model.
        """

        base_config = super().get_config()
        
        # Serialize each model in the model list using a list comprehension
        serialized_model_list = [keras.utils.serialize_keras_object(model) for model in self.model_list]
        regularized_model_list = [keras.utils.serialize_keras_object(regularizer) for regularizer in self.regularizer_list]
        
        config = {
            "Path": np.asarray(self.Path).tolist(),
            "model_list": serialized_model_list,  # Include serialized model list in the configuration
            "regularizer_list": regularized_model_list,
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
        config['Path'] = tf.constant(config['Path'])
        
        # Deserialize each model in the model list using a list comprehension
        config['model_list'] = [keras.utils.deserialize_keras_object(model_config) for model_config in config['model_list']]
        config['run_from_config'] = True
        
        # If regularization is present in the config, deserialize it
        if 'regularizer_list' in config:
            config['regularizer_list'] = [keras.utils.deserialize_keras_object(regularizer_config) for regularizer_config in config['regularizer_list']]
        
        return cls(**config)
    
    def get_compile_config(self):
        """
        Serializes the optimizer configurations of the models.

        Returns:
            dict: A dictionary containing the serialized optimizer configurations of the models.
        """
        return {
            "model_optimizers": [keras.utils.serialize_keras_object(model.optimizer) for model in self.model_list]
        }
    
    def compile_from_config(self, config):
        """
        Compiles the models with the deserialized optimizer configurations.

        Args:
            config (dict): A dictionary containing the serialized optimizer configurations.
        """
        optimizer_list = [keras.utils.deserialize_keras_object(optimizer_config) for optimizer_config in config["model_optimizers"]]
        self.compile(optimizer_list)

    def build_from_config(self, config):
        """ build is overwritten here as it is not needed. Individual measurement models
        are built seperately, this happens when keras.saving.deserialize_keras_object is called
        on models in model_list"""

        return
    


# import os
# import tensorflow as tf
# import numpy as np
# # import deep_lvpm 
# from deep_lvpm.layers.FactorLayer import FactorLayer
# from deep_lvpm.layers.ZCALayer import ZCALayer
# from deep_lvpm.layers.ConfoundLayer import ConfoundLayer
# import pydot

# # from Custom_Losses_and_Metrics import mse_loss
# # from Custom_Losses_and_Metrics import corr_metric

# # Set up metrics trackers
# loss_tracker_total = keras.metrics.Mean(name="total_loss")
# loss_tracker_mse = keras.metrics.Mean(name="mean_squared_loss")
# corr_tracker = keras.metrics.Mean(name="corr_metric")




# @keras.utils.register_keras_serializable(package="deep_lvpm",name="StructuralModel")
# class StructuralModel(keras.Model):
    
#     """
#     A custom Keras model to establish associations between different data-views.

#     This model implements a deep learning approach to find deep latent variables (DLVs)
#     that highlight the correlated factors between different types of data.
#     The associations between data-views are defined using a binary adjacency matrix,
#     where ones represent connections, and zeros represent un-connected data-views.

#     Attributes:
#         Path: A binary adjacency matrix defining the connections between data-views.
#         model_list: A list of Keras models for each data-view.
#         tot_num: Total number of features across all batches.
#         ndims: Number of orthogonal latent variables to construct.
#         epochs: Number of training epochs.
#         batch_size: Size of the batches used during training.
#         orthogonalization: Orthogonalisation procedure ('zca' or 'Moore-Penrose').
#         loss_tracker_total: Tracker for the total loss during training.
#         corr_tracker: Tracker for the correlation metric during training.
#         loss_tracker_mse: Tracker for the mean squared error loss during training.

#     Methods:
#         call: Runs data through each of the measurement sub-models.
#         train_step: Performs a training step, updating the model weights.
#         compile: Configures the model for training.
#         test_step: Evaluates the model on a batch of test data.
#         metrics: Returns the list of model's metrics.
#         mse_loss: Calculates mean squared error loss for a data-view.
#         corr_metric: Calculates the correlation metric for a data-view.
#     """

    
#     def __init__(self, Path, model_list, regularizer_list, tot_num, ndims, orthogonalization='Moore-Penrose', momentum=0.95, epsilon=1e-4, train_DLV=False, run_from_config=False, **kwargs):
        
#         """
#         Initializes the StructuralModel instance.

#         Args:
#             Path (tf.Tensor or np.array): A binary adjacency matrix defining connections between data-views.
#             regularizer_list (list): A list of regularizers that are applied to projection layers for models
#             in each data-view.
#             model_list (list): A list of Keras models for each data-view.
#             tot_num (int): Total number of features across all batches.
#             ndims (int): Number of orthogonal latent variables to construct.
#             orthogonalization (str, optional): Orthogonalisation procedure. Defaults to 'Moore-Penrose'.
#             momentum (Float, optional): The momentum defines how quickly global parameters such as means and correlation matrices are updated
#             epsilon (Float, optional): "epsilon" (often denoted as ε) is a small constant added for numerical stability in batch updates
#             train_DLV (True/False): "train_DLV" defines whether target DLVs are calcualted in training or testing modes during model training
#         """

#         super().__init__(**kwargs)    
        
#         self.Path = Path
#         self.tot_num = tot_num
#         self.ndims = ndims
#         self.momentum = momentum
#         self.epsilon = epsilon
#         self.orthogonalization=orthogonalization
#         self.regularizer_list = regularizer_list
#         self.train_DLV = train_DLV

#         if not run_from_config:
#         # Add factor layer to each model in the list
#             self.model_list = [self.add_DLVPM_layer(model, regularizer) for model, regularizer in zip(model_list, regularizer_list)]
#         else:
#             self.model_list = model_list

#         self.loss_tracker_total = keras.metrics.Mean(name="total_loss")
#         self.corr_tracker = keras.metrics.Mean(name="cross_metric")
#         self.loss_tracker_mse = keras.metrics.Mean(name="mse_loss")

    
#     def add_DLVPM_layer(self, model, regularizer):
#         """
#         Adds a FactorLayer on top of the given model.

#         The method first checks whether the input model is sequential or functional,
#         and then adds the FactorLayer in an appropriate way.

#         :param model: A Keras/TensorFlow model (sequential or functional).
#         :return: The model with an added FactorLayer on top.
#         """
#         if isinstance(model, keras.Sequential):
#             if self.orthogonalization == 'Moore-Penrose':
#                 model.add(FactorLayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon))
#             elif self.orthogonalization == 'zca':
#                 model.add(ZCALayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon))
#             else:
#                 print('Orthogonalization mode not recognised, must be "Moore-Penrose" or "zca"')
#         elif isinstance(model, keras.Model):
#             if self.orthogonalization == 'Moore-Penrose':
#                 x = FactorLayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon)(model.output)
#                 model = keras.Model(inputs=model.input, outputs=x)
#             elif self.orthogonalization == 'zca':
#                 x = ZCALayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon)(model.output)
#                 model = keras.Model(inputs=model.input, outputs=x)
#             else:
#                 print('Orthogonalization mode not recognised, must be "Moore-Penrose" or "zca"')
#         else:
#             raise ValueError("The input model must be either a keras.Sequential or a keras.Model instance.")

        
#         return model



    
#     def call(self, inputs, training=False):
#         """
#         Run data through each of the measurement sub-models.

#         Args:
#             inputs (list): A list of inputs for each data-view.
#             training: Whether to call the model in training or inference mode. Can take values of True or False.

#         Returns:
#             tf.Tensor: The output of the model after processing the inputs.
#         """

#         inputs_nested = self.organize_inputs_by_model(inputs) ## this function organises flat inputs into a list of lists, which makes model training easier

#         out=tf.stack([self.model_list[vie](inputs_nested[vie], training = training) for vie in range(len(self.model_list))],axis=2) ## Stack the outputs 
    
#         return out
    
#     def organize_inputs_by_model(self, data_inputs):
#         organized_inputs = []
#         data_index = 0

#         for model in self.model_list:
            
#             num_inputs = len(model.inputs) if hasattr(model, 'inputs') else 1

#             if num_inputs == 1:
#                 # For a single input model, append the data directly.
#                 organized_inputs.append(data_inputs[data_index])
#                 data_index += 1
#             else:
#                 # For models requiring multiple inputs, append a list of inputs.
#                 inputs_for_model = data_inputs[data_index:data_index + num_inputs]
#                 organized_inputs.append(inputs_for_model)
#                 data_index += num_inputs

#         return organized_inputs


#     def train_step(self, inputs):
        
#         """
#         Perform a training step, updating the model weights.

#         Args:
#             inputs (list or tuple): A list of inputs for each data-view.

#         Returns:
#             dict: A dictionary containing the total loss, cross metric, and mean squared error loss.
#         """
       
#         ## tensorflow packs inputs in another tuple, this should be unpacked
#         inputs=inputs[0]
        
#         # Here, we run the current data-iteration through the global model in a forward 
#         y = self(inputs, training=self.train_DLV)  ## forward pass

#         ## Here, we re-normalise the model weights
#         scale_fact = tf.cast(self.tot_num/tf.shape(y)[0],dtype=float) # scale factor for re-scaling

#         y_list = []
#         for vie in range(len(self.model_list)):
#             y_view = y[:,:,vie] ## This is the current view under analysis
#             y_view = self.model_list[vie].layers[-1].weight_normalizer([y_view, scale_fact]) ## Normalize weights and return normalized output (last layer of model)
#             y_list.append(y_view) ## append normalized output to list
#         y = tf.stack(y_list, axis=-1) ## normalized data output
            

#         total_loss = [None]*(len(self.model_list))
#         total_CC = [None]*(len(self.model_list))
#         total_mse = [None]*(len(self.model_list))
        
#         inputs_nested = self.organize_inputs_by_model(inputs) ## this function organises flat inputs into a list of lists, which makes model training easier

#         ## Iterate through training data-views
#         for vie in range(len(self.model_list)):

#             with tf.GradientTape() as tape:
                
#                 ## forward pass
#                 y_pred = self.model_list[vie](inputs_nested[vie], training=True)

#                 y_pred = tf.divide(y_pred,tf.math.multiply(tf.math.sqrt(scale_fact),tf.norm(y_pred,axis=0))) ## Here, we re-normalize DLVs

#                 mse_loss = self.mse_loss(y, y_pred, vie)
                
#                 internal_loss = self.model_list[vie].losses
                
#                 # # Compute the loss for the data-view in question
#                 loss = mse_loss + internal_loss
            
            
#             # Compute gradients
#             trainable_vars = self.model_list[vie].trainable_variables
#             gradients = tape.gradient(loss, trainable_vars)
            
#             # Update weights
#             self.model_list[vie].optimizer.apply_gradients(zip(gradients, trainable_vars))
            
#             corr_metric=self.corr_metric(y,y_pred,vie)
            
#             ## add current losses and metrics to the global lists
#             total_loss[vie]=tf.math.reduce_sum(loss)
#             total_CC[vie]=corr_metric
#             total_mse[vie]=mse_loss
                
#         # Update losses and metrics
#         self.loss_tracker_total.update_state(tf.stack(total_loss))
#         self.corr_tracker.update_state(tf.stack(total_CC))
#         self.loss_tracker_mse.update_state(tf.stack(total_mse))
        
        
#         return {"total_loss": self.loss_tracker_total.result(), "cross_metric": self.corr_tracker.result(), "mse_loss":self.loss_tracker_mse.result()}

#     def compile(self, optimizer):
#         """ Here, we overwrite the model compilation step. This is necessary as
#         normally, the model compilation step would normally take a loss. Using
#         this method, the loss is built into the method itself. We can either 
#         pass the optimizer a single optimizer object, or a list of objects, with a 
#         different optimizer used for each data-view.
#         """
        
#         super().compile()
        
#         #self.global_build()
        
#         if isinstance(optimizer, list):
#             for vie in range(len(self.model_list)):
#                 self.model_list[vie].compile(optimizer[vie])
#         elif isinstance(optimizer,keras.optimizers.Optimizer):
#             for vie in range(len(self.model_list)):
#                 self.model_list[vie].compile(optimizer)
#         else:
#             print('Error: optimizer must either be of the keras.optimizer class, or a list of objects of this class')
        

#     def test_step(self, inputs):
        
#         """ This step is called by model.evaluate() on a batch-wise level. This function
#         returns loss metrics for the test data.
        
#         """
        
#         ## tensorflow packs inputs in another tuple, this should be unpacked
#         inputs=inputs[0]

#         #inputs = self.organize_inputs_by_model(inputs)
        
#         y = self(inputs, training=False)  ## forward pass
    
#         total_loss = [None]*(len(self.model_list))
#         total_CC = [None]*(len(self.model_list))
#         total_mse = [None]*(len(self.model_list))
        
#         inputs_nested = self.organize_inputs_by_model(inputs)
#         ## Iterate through training data-views
#         for vie in range(len(self.model_list)):
          
                
#             ## forward pass
#             y_pred = self.model_list[vie](inputs_nested[vie], training=False)
            
#             mse_loss = self.mse_loss(y, y_pred, vie)
#             internal_loss = self.model_list[vie].losses
            
#             # Compute the loss for the data-view in question
#             loss = mse_loss + internal_loss
    
#             corr_metric=self.corr_metric(y,y_pred,vie)
            
#             ## add current losses and metrics to the global lists
#             total_loss[vie]=tf.math.reduce_sum(loss)
#             total_CC[vie]=corr_metric
#             total_mse[vie]=mse_loss
            
           
#         # Update losses and metrics
#         self.loss_tracker_total.update_state(total_loss)
#         self.corr_tracker.update_state(total_CC)
#         self.loss_tracker_mse.update_state(total_mse)
            
#         return {m.name: m.result() for m in self.metrics}

#     @property
#     def metrics(self):
#         # We list our `Metric` objects here so that `reset_states()` can be
#         # called automatically at the start of each epoch
#         # or at the start of `evaluate()`.

#         return [self.loss_tracker_total, self.corr_tracker, self.loss_tracker_mse]

        
#     def mse_loss(self,y_true,y_pred,vie):
        
#         """ This function returns the mean squared error loss between the latent
#         factors in a particular data-view, and the latent factors to which that
#         data-view is connected via the global DLVPM model.
#         """
        
#         y_true =  tf.squeeze(tf.gather(y_true,tf.where(self.Path[vie,:]),axis=2),axis=3) ## select the latent factors connected to the latent factor for view vie
        
#         y_pred = tf.expand_dims(y_pred,axis=2) ## expand dimensions of the predicted latent factor so broadcasting is possible
        
#         mse_loss = tf.divide(tf.reduce_sum(tf.math.reduce_mean(tf.math.square(tf.subtract(y_true,y_pred)),axis=0)),2)

#         return mse_loss
    
#     def corr_metric(self,y_true,y_pred,vie):
        
#         """ This function returns the mean correlation between the latent factors
#         in a data-view, and the latent factors to which that data-view is connected 
#         via the global DLVPM model.
        
#         """
      
#         y_true =  tf.squeeze(tf.gather(y_true,tf.where(self.Path[vie,:]),axis=2),axis=3) ## select the latent factors connected to the latent factor for view vie
        
#         ## Minus the mean
#         y_true_mean = tf.subtract(y_true,tf.math.reduce_mean(y_true,axis=0))
#         y_pred_mean = tf.subtract(y_pred,tf.math.reduce_mean(y_pred,axis=0))
        
#         # # ## Normalise matrices
#         y_true_norm = tf.divide(y_true_mean,tf.norm(y_true_mean,axis=0))
#         y_pred_norm = tf.divide(y_pred_mean,tf.norm(y_pred_mean,axis=0))
        
#         y_pred_norm = tf.expand_dims(y_pred_norm,axis=2) ## expand dimensions of the predicted latent factor so broadcasting is possible
        
#         corr2=tf.math.reduce_sum(tf.math.multiply(y_true_norm, y_pred_norm),axis=0)

#         return tf.math.reduce_mean(corr2)
    
#     import tensorflow as tf

#     def calculate_corrmat(self, DLVs):
#         """
#         Compute Pearson correlation coefficient matrices for a 3D tensor.

#         This function takes a 3D tensor of shape (n_samples, dimensions, DLVs) and computes
#         the Pearson correlation coefficient between each pair of DLVs for each dimension. 
#         The output is a list of symmetric matrices, one for each dimension, of shape (DLVs, DLVs).

#         Args:
#         DLVs (tf.Tensor): A 3D tensor of shape (n_samples, dimensions, DLVs).

#         Returns:
#         List[tf.Tensor]: A list of 2D tensors, each of shape (DLVs, DLVs), containing 
#                         the Pearson correlation coefficients for each dimension.
#         """
#         # Ensure the input is a 3D tensor
#         if len(DLVs.shape) != 3:
#             raise ValueError("Input must be a 3D tensor")

#         # List to store correlation matrices for each dimension
#         correlation_matrices = []

#         # Iterate through each dimension
#         for dim in range(DLVs.shape[1]):
#             # Select the data for the current dimension
#             dim_DLVs = DLVs[:, dim, :]

#             # Centering the DLVs by subtracting the mean
#             mean_centered = dim_DLVs - tf.reduce_mean(dim_DLVs, axis=0)

#             # Compute the standard deviation for each feature
#             std_dev = tf.math.reduce_std(dim_DLVs, axis=0)

#             # Normalize each feature
#             normalized_DLVs = mean_centered / std_dev

#             # Compute the correlation matrix for the current dimension
#             correlation_matrix = tf.linalg.matmul(normalized_DLVs, normalized_DLVs, transpose_a=True) / tf.cast(tf.shape(dim_DLVs)[0], tf.float32)
#             correlation_matrices.append(correlation_matrix)

#         return correlation_matrices
    

#     def plot_structural_model(self, outputname):
#         """
#         This function plots the structural/path. model. Visualisation is quite simple. 
#         Aesthetics are similar to those used in keras.utils.plot_model()
#         outputname: This is the name of the output where we save the results. 

#         """
#         # Create a PyDot graph
#         graph = pydot.Dot(graph_type='digraph', rankdir='TB')

#         model_layer_list= [len(model.layers) for model in self.model_list]

#         # Create nodes with labels
#         for i in range(len(self.model_list)):

#             label = "Measurement Model " + str(i) + "," + " " + str(model_layer_list[i]) + " layers"
#             node = pydot.Node(str(i), label=label, shape="record") # create nodes to add to the pydot object
#             graph.add_node(node) # add nodes to the pydot graph object

#         adj_matrix = self.Path # this is the path. model we wish to plot

#         # Create edges
#         for i, row in enumerate(adj_matrix):
#             for j, val in enumerate(row):
#                 if val == 1:
#                     edge = pydot.Edge(str(i), str(j))
#                     graph.add_edge(edge)

#         graph.write_png(outputname)

            

#     def get_config(self):

#         """
#         Gets configuration of the model for serialization.

#         Returns:
#             Dictionary containing the configuration of the model.
#         """
#         base_config = super().get_config()
        
#         # Serialize each model in the model list using a list comprehension
#         serialized_model_list = [keras.utils.serialize_keras_object(model) for model in self.model_list]
#         regularized_model_list = [keras.utils.serialize_keras_object(regularizer) for regularizer in self.regularizer_list]
        
#         config = {
#             "Path": np.asarray(self.Path).tolist(),
#             "model_list": serialized_model_list,  # Include serialized model list in the configuration
#             "regularizer_list": regularized_model_list,
#             "tot_num": self.tot_num,
#             "ndims": self.ndims,  
#             "orthogonalization": self.orthogonalization
#         }
    
#         return {**base_config, **config}
    
#     @classmethod    
#     def from_config(cls, config):
#         """
#         Creates an instance of the class from a config dictionary.

#         Args:
#             config (dict): A dictionary containing the configuration of the instance.

#         Returns:
#             An instance of the class.
#         """
#         # Deserialize Keras/TensorFlow objects
#         config['Path'] = tf.constant(config['Path'])
        
#         # Deserialize each model in the model list using a list comprehension
#         config['model_list'] = [keras.utils.deserialize_keras_object(model_config) for model_config in config['model_list']]
#         config['run_from_config'] = True
        
#         # If regularization is present in the config, deserialize it
#         if 'regularizer_list' in config:
#             config['regularizer_list'] = [keras.utils.deserialize_keras_object(regularizer_config) for regularizer_config in config['regularizer_list']]
        
#         return cls(**config)
    
#     def get_compile_config(self):
#         """
#         Serializes the optimizer configurations of the models.

#         Returns:
#             dict: A dictionary containing the serialized optimizer configurations of the models.
#         """
#         return {
#             "model_optimizers": [keras.utils.serialize_keras_object(model.optimizer) for model in self.model_list]
#         }
    
#     def compile_from_config(self, config):
#         """
#         Compiles the models with the deserialized optimizer configurations.

#         Args:
#             config (dict): A dictionary containing the serialized optimizer configurations.
#         """
#         optimizer_list = [keras.utils.deserialize_keras_object(optimizer_config) for optimizer_config in config["model_optimizers"]]
#         self.compile(optimizer_list)

#     def build_from_config(self, config):
#         """ build is overwritten here as it is not needed. Individual measurement models
#         are built seperately, this happens when keras.saving.deserialize_keras_object is called
#         on models in model_list"""

#         return
    




        #self.build(config["input_shape"])
    


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Keras 3 (multi-backend) version of StructuralModel

# This model establishes associations between multiple data-views via deep latent
# variables (DLVs), wired by a binary adjacency matrix (Path). Each view is a
# measurement model to which we append a DLV-producing layer (FactorLayer/ZCALayer).

# Compatible backends: TensorFlow, PyTorch, JAX (via keras.ops + keras.backend).
# """

# import os
# import numpy as np
# import pydot

# import keras
# from keras import Model
# from keras import ops
# from keras import backend as K
# from keras.metrics import Mean
# from keras.saving import (
#     register_keras_serializable,
#     serialize_keras_object,
#     deserialize_keras_object,
# )

# # deep_lvpm custom layers (must also be implemented with keras.ops)
# from deep_lvpm.layers.FactorLayer import FactorLayer
# from deep_lvpm.layers.ZCALayer import ZCALayer
# from deep_lvpm.layers.ConfoundLayer import ConfoundLayer  # imported for completeness


# @register_keras_serializable(package="deep_lvpm", name="StructuralModel")
# class StructuralModel(keras.Model):
#     """
#     A custom Keras 3 model to learn deep latent variables (DLVs) that align
#     across multiple data-views according to a binary adjacency matrix.

#     Attributes
#     ----------
#     Path : KerasTensor / backend tensor
#         Binary adjacency matrix (views x views): 1 = connect view i -> j.
#     model_list : list[keras.Model]
#         One measurement model per data-view; each ends with a DLV layer.
#     regularizer_list : list[keras.regularizers.Regularizer]
#         One regularizer per view for the projection layer (DLV layer).
#     tot_num : int
#         Total number of features across batches (used in weight renormalization).
#     ndims : int
#         Number of orthogonal latent variables to construct.
#     orthogonalization : str
#         'Moore-Penrose' or 'zca' – selects FactorLayer vs ZCALayer.
#     momentum : float
#     epsilon : float
#     train_DLV : bool
#         Whether to compute target DLVs in training or inference mode during training.
#     """

#     def __init__(
#         self,
#         Path,
#         model_list,
#         regularizer_list,
#         tot_num,
#         ndims,
#         orthogonalization="Moore-Penrose",
#         momentum=0.95,
#         epsilon=1e-4,
#         train_DLV=False,
#         run_from_config=False,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)

#         self.Path = ops.convert_to_tensor(Path)
#         self.tot_num = int(tot_num)
#         self.ndims = int(ndims)
#         self.momentum = float(momentum)
#         self.epsilon = float(epsilon)
#         self.orthogonalization = orthogonalization
#         self.regularizer_list = regularizer_list
#         self.train_DLV = bool(train_DLV)

#         if not run_from_config:
#             # Append a DLV layer to each measurement model
#             self.model_list = [
#                 self._add_dlvpm_layer(m, r) for m, r in zip(model_list, regularizer_list)
#             ]
#         else:
#             # Already deserialized measurement models contain their DLV layers
#             self.model_list = model_list

#         # Metric trackers
#         self.loss_tracker_total = Mean(name="total_loss")
#         self.corr_tracker = Mean(name="cross_metric")
#         self.loss_tracker_mse = Mean(name="mse_loss")

#     # ---------------------------
#     # Building blocks / utilities
#     # ---------------------------

#     def _add_dlvpm_layer(self, model, regularizer):
#         """Append FactorLayer or ZCALayer to a Sequential or Functional model."""
#         if isinstance(model, keras.Sequential):
#             layer_cls = FactorLayer if self.orthogonalization == "Moore-Penrose" else (
#                 ZCALayer if self.orthogonalization == "zca" else None
#             )
#             if layer_cls is None:
#                 raise ValueError('orthogonalization must be "Moore-Penrose" or "zca"')
#             model.add(
#                 layer_cls(
#                     kernel_regularizer=regularizer,
#                     tot_num=self.tot_num,
#                     ndims=self.ndims,
#                     momentum=self.momentum,
#                     epsilon=self.epsilon,
#                 )
#             )
#         elif isinstance(model, keras.Model):
#             if self.orthogonalization == "Moore-Penrose":
#                 x = FactorLayer(
#                     kernel_regularizer=regularizer,
#                     tot_num=self.tot_num,
#                     ndims=self.ndims,
#                     momentum=self.momentum,
#                     epsilon=self.epsilon,
#                 )(model.output)
#             elif self.orthogonalization == "zca":
#                 x = ZCALayer(
#                     kernel_regularizer=regularizer,
#                     tot_num=self.tot_num,
#                     ndims=self.ndims,
#                     momentum=self.momentum,
#                     epsilon=self.epsilon,
#                 )(model.output)
#             else:
#                 raise ValueError('orthogonalization must be "Moore-Penrose" or "zca"')
#             model = keras.Model(inputs=model.input, outputs=x)
#         else:
#             raise ValueError(
#                 "Each measurement model must be a keras.Sequential or keras.Model."
#             )
#         return model

#     def organize_inputs_by_model(self, data_inputs):
#         """Arrange a flat list of inputs into a per-model list (handles multi-input models)."""
#         organized, data_index = [], 0
#         for model in self.model_list:
#             num_inputs = len(model.inputs) if hasattr(model, "inputs") else 1
#             if num_inputs == 1:
#                 organized.append(data_inputs[data_index])
#                 data_index += 1
#             else:
#                 organized.append(data_inputs[data_index : data_index + num_inputs])
#                 data_index += num_inputs
#         return organized

#     # -------------
#     # Model methods
#     # -------------

#     def call(self, inputs, training=False):
#         """
#         Run inputs through each measurement sub-model.
#         Returns tensor of shape (batch, ndims, n_views).
#         """
#         inputs_nested = self.organize_inputs_by_model(inputs)
#         outs = [
#             self.model_list[v](inputs_nested[v], training=training)
#             for v in range(len(self.model_list))
#         ]  # each: (batch, ndims)
#         out = ops.stack(outs, axis=2)  # (batch, ndims, n_views)
#         return out

#     def train_step(self, data):
#         """
#         One training step updating *each* view's measurement model with its own optimizer.
#         """
#         # Unpack data the way tf.data/torch DataLoader/jax generator typically passes it
#         inputs = data[0] if isinstance(data, (tuple, list)) else data

#         # Forward pass through ALL views to construct "global" DLVs (optionally in train/infer mode)
#         y = self(inputs, training=self.train_DLV)  # (batch, ndims, n_views)

#         # scale_fact = tot_num / batch_size
#         batch_size = ops.cast(ops.shape(y)[0], y.dtype)
#         scale_fact = ops.cast(self.tot_num, y.dtype) / batch_size

#         # Optional weight/output renormalization per view
#         y_list = []
#         for vie in range(len(self.model_list)):
#             y_view = y[:, :, vie]  # (batch, ndims)
#             last_layer = self.model_list[vie].layers[-1]
#             if hasattr(last_layer, "weight_normalizer"):
#                 # Expecting signature: weight_normalizer([y_view, scale_fact])
#                 y_view = last_layer.weight_normalizer([y_view, scale_fact])
#             y_list.append(y_view)
#         y = ops.stack(y_list, axis=-1)  # (batch, ndims, n_views)

#         total_loss = [None] * len(self.model_list)
#         total_CC = [None] * len(self.model_list)
#         total_mse = [None] * len(self.model_list)

#         inputs_nested = self.organize_inputs_by_model(inputs)

#         # Optimize each view model separately (view-wise objective)
#         for vie in range(len(self.model_list)):
#             with K.GradientTape() as tape:
#                 y_pred = self.model_list[vie](
#                     inputs_nested[vie], training=True
#                 )  # (batch, ndims)

#                 # Re-normalize y_pred DLVs: y_pred / (sqrt(scale_fact) * ||y_pred||)
#                 denom = ops.sqrt(scale_fact) * (ops.norm(y_pred, axis=0) + self.epsilon)
#                 y_pred = y_pred / denom  # broadcasts over batch

#                 mse_loss_val = self.mse_loss(y, y_pred, vie)

#                 # Regularization (sum of layer losses)
#                 if self.model_list[vie].losses:
#                     reg_loss = ops.sum(ops.stack(self.model_list[vie].losses))
#                 else:
#                     reg_loss = ops.convert_to_tensor(0.0, dtype=ops.dtype(y))

#                 loss = mse_loss_val + reg_loss

#             # Apply gradients using the view's optimizer
#             trainable_vars = self.model_list[vie].trainable_variables
#             grads = tape.gradient(loss, trainable_vars)
#             self.model_list[vie].optimizer.apply_gradients(zip(grads, trainable_vars))

#             # Metrics
#             cc = self.corr_metric(y, y_pred, vie)
#             total_loss[vie] = ops.cast(loss, "float32")
#             total_CC[vie] = ops.cast(cc, "float32")
#             total_mse[vie] = ops.cast(mse_loss_val, "float32")

#         # Update trackers
#         self.loss_tracker_total.update_state(ops.stack(total_loss))
#         self.corr_tracker.update_state(ops.stack(total_CC))
#         self.loss_tracker_mse.update_state(ops.stack(total_mse))

#         return {
#             "total_loss": self.loss_tracker_total.result(),
#             "cross_metric": self.corr_tracker.result(),
#             "mse_loss": self.loss_tracker_mse.result(),
#         }

#     def test_step(self, data):
#         """Evaluation step computing the same metrics without weight updates."""
#         inputs = data[0] if isinstance(data, (tuple, list)) else data
#         y = self(inputs, training=False)  # (batch, ndims, n_views)

#         total_loss = [None] * len(self.model_list)
#         total_CC = [None] * len(self.model_list)
#         total_mse = [None] * len(self.model_list)

#         inputs_nested = self.organize_inputs_by_model(inputs)

#         for vie in range(len(self.model_list)):
#             y_pred = self.model_list[vie](inputs_nested[vie], training=False)

#             mse_loss_val = self.mse_loss(y, y_pred, vie)

#             if self.model_list[vie].losses:
#                 reg_loss = ops.sum(ops.stack(self.model_list[vie].losses))
#             else:
#                 reg_loss = ops.convert_to_tensor(0.0, dtype=ops.dtype(y))

#             loss = mse_loss_val + reg_loss
#             cc = self.corr_metric(y, y_pred, vie)

#             total_loss[vie] = ops.cast(loss, "float32")
#             total_CC[vie] = ops.cast(cc, "float32")
#             total_mse[vie] = ops.cast(mse_loss_val, "float32")

#         self.loss_tracker_total.update_state(ops.stack(total_loss))
#         self.corr_tracker.update_state(ops.stack(total_CC))
#         self.loss_tracker_mse.update_state(ops.stack(total_mse))

#         return {m.name: m.result() for m in self.metrics}

#     def compile(self, optimizer):
#         """
#         Compile wrapper. You can pass:
#           - a single keras optimizer (applied to all view models), or
#           - a list of optimizers (one per view).
#         The parent model itself does not use an optimizer because we override train_step
#         and update the sub-models directly.
#         """
#         super().compile()  # no loss/metrics—handled in train_step/test_step

#         if isinstance(optimizer, list):
#             if len(optimizer) != len(self.model_list):
#                 raise ValueError(
#                     "optimizer list length must match number of views (len(model_list))."
#                 )
#             for vie in range(len(self.model_list)):
#                 self.model_list[vie].compile(optimizer=optimizer[vie])
#         elif isinstance(optimizer, keras.optimizers.Optimizer):
#             for vie in range(len(self.model_list)):
#                 self.model_list[vie].compile(optimizer=optimizer)
#         else:
#             raise TypeError("optimizer must be a keras optimizer or list of optimizers.")

#     @property
#     def metrics(self):
#         # Ensures reset_states() is called automatically by Keras at epoch/evaluate start.
#         return [self.loss_tracker_total, self.corr_tracker, self.loss_tracker_mse]

#     # ---------------
#     # Loss & Metrics
#     # ---------------

#     def _path_mask(self, vie, dtype):
#         mask = ops.cast(self.Path[vie, :], "bool")       # (n_views,)
#         return ops.cast(mask, dtype)                     # numeric mask for weighting

#     def mse_loss(self, y_true_all, y_pred, vie):
#         """
#         Mean squared error between y_pred (view vie) and the *connected* DLVs
#         in the global stack y_true_all, as specified by Path[vie, :].

#         y_true_all : (batch, ndims, n_views)
#         y_pred     : (batch, ndims)
#         """
#         # Broadcast y_pred to (batch, ndims, 1) then to (batch, ndims, n_views)
#         y_pred_exp = ops.expand_dims(y_pred, axis=2)  # (batch, ndims, 1)

#         # Per-connection squared error averaged over batch -> (ndims, n_views)
#         se_mean = ops.mean(ops.square(y_true_all - y_pred_exp), axis=0)

#         # Mask to include only connected views; then sum over ndims and views
#         mask = self._path_mask(vie, dtype=ops.dtype(se_mean))  # (n_views,)
#         se_mean_masked = se_mean * ops.expand_dims(mask, axis=0)  # (ndims, n_views)

#         mse_loss_val = ops.sum(se_mean_masked) / 2.0
#         return mse_loss_val

#     def corr_metric(self, y_true_all, y_pred, vie):
#         """
#         Mean Pearson-style correlation between y_pred (view vie) and connected y_true DLVs.
#         Returns scalar averaged over connected views and dimensions.
#         """
#         eps = self.epsilon

#         # Center over batch
#         y_true_c = y_true_all - ops.mean(y_true_all, axis=0)  # (batch, ndims, n_views)
#         y_pred_c = y_pred - ops.mean(y_pred, axis=0)          # (batch, ndims)

#         # Normalize
#         denom_true = ops.norm(y_true_c, axis=0) + eps         # (ndims, n_views)
#         denom_pred = ops.norm(y_pred_c, axis=0) + eps         # (ndims,)
#         y_true_n = y_true_c / denom_true
#         y_pred_n = y_pred_c / denom_pred                      # (batch, ndims)
#         y_pred_n = ops.expand_dims(y_pred_n, axis=2)          # (batch, ndims, 1)

#         # Correlation per (dim, view)
#         corr_mat = ops.sum(y_true_n * y_pred_n, axis=0)       # (ndims, n_views)

#         # Mask to only connected targets
#         mask = self._path_mask(vie, dtype=ops.dtype(corr_mat))  # (n_views,)
#         corr_masked = corr_mat * ops.expand_dims(mask, axis=0)  # (ndims, n_views)

#         # Average over dims and number of connected views
#         n_conn = ops.sum(mask)  # scalar
#         n_conn_safe = ops.maximum(n_conn, ops.convert_to_tensor(1.0, dtype=ops.dtype(n_conn)))
#         corr_mean = ops.sum(corr_masked) / (n_conn_safe * float(self.ndims))
#         return corr_mean

#     # ---------------------------
#     # Analysis / visualization API
#     # ---------------------------

#     def calculate_corrmat(self, DLVs):
#         """
#         Compute Pearson correlation matrices for a 3D tensor.
#         DLVs: (n_samples, ndims, n_feats)  -> returns: list of (n_feats x n_feats) per dim.
#         """
#         if len(DLVs.shape) != 3:
#             raise ValueError("DLVs must be a 3D tensor")

#         correlation_matrices = []
#         n_samples = ops.cast(ops.shape(DLVs)[0], DLVs.dtype)
#         eps = ops.convert_to_tensor(1e-7, dtype=DLVs.dtype)

#         # Loop over dims (Python loop is fine; the body is vectorized)
#         for dim in range(DLVs.shape[1]):
#             dim_DLVs = DLVs[:, dim, :]                       # (n_samples, n_feats)
#             mean_centered = dim_DLVs - ops.mean(dim_DLVs, axis=0)
#             std_dev = ops.std(dim_DLVs, axis=0) + eps
#             normalized = mean_centered / std_dev
#             corr = ops.matmul(normalized, normalized, transpose_a=True) / n_samples
#             correlation_matrices.append(corr)
#         return correlation_matrices

#     def plot_structural_model(self, outputname):
#         """
#         Plot the path structure defined by self.Path using pydot.
#         """
#         graph = pydot.Dot(graph_type="digraph", rankdir="TB")

#         model_layer_list = [len(m.layers) for m in self.model_list]
#         for i, n_layers in enumerate(model_layer_list):
#             label = f"Measurement Model {i}, {n_layers} layers"
#             node = pydot.Node(str(i), label=label, shape="record")
#             graph.add_node(node)

#         # Backend-agnostic -> NumPy
#         adj = ops.convert_to_numpy(self.Path)
#         for i, row in enumerate(adj):
#             for j, val in enumerate(row):
#                 if int(val) == 1:
#                     graph.add_edge(pydot.Edge(str(i), str(j)))

#         graph.write_png(outputname)

#     # ---------------------------
#     # Saving / serialization hooks
#     # ---------------------------

#     def get_config(self):
#         base_config = super().get_config()
#         serialized_model_list = [serialize_keras_object(m) for m in self.model_list]
#         serialized_regularizers = [serialize_keras_object(r) for r in self.regularizer_list]

#         cfg = {
#             "Path": np.asarray(ops.convert_to_numpy(self.Path)).tolist(),
#             "model_list": serialized_model_list,
#             "regularizer_list": serialized_regularizers,
#             "tot_num": self.tot_num,
#             "ndims": self.ndims,
#             "orthogonalization": self.orthogonalization,
#             "momentum": self.momentum,
#             "epsilon": self.epsilon,
#             "train_DLV": self.train_DLV,
#         }
#         return {**base_config, **cfg}

#     @classmethod
#     def from_config(cls, config):
#         # Deserialize Keras objects and mark that models already contain their DLV layers
#         config["Path"] = ops.convert_to_tensor(config["Path"])
#         config["model_list"] = [deserialize_keras_object(mc) for mc in config["model_list"]]
#         config["regularizer_list"] = [
#             deserialize_keras_object(rc) for rc in config.get("regularizer_list", [])
#         ]
#         config["run_from_config"] = True
#         return cls(**config)

#     def get_compile_config(self):
#         """Serialize optimizers for each view model."""
#         return {"model_optimizers": [serialize_keras_object(m.optimizer) for m in self.model_list]}

#     def compile_from_config(self, config):
#         """Restore optimizers for each view model."""
#         optimizer_list = [deserialize_keras_object(o) for o in config["model_optimizers"]]
#         self.compile(optimizer_list)

#     def build_from_config(self, config):
#         """No-op: each measurement model is built separately when deserialized."""
#         return








