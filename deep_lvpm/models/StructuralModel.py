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

@tf.keras.saving.register_keras_serializable(package="deep_lvpm",name="StructuralModel")
class StructuralModel(tf.keras.Model):
    
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

    
    def __init__(self, Path, model_list, regularizer_list, tot_num, ndims, orthogonalization='Moore-Penrose', momentum=0.95, epsilon=1e-4, run_from_config=False, **kwargs):
        
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
        """

        super().__init__(**kwargs)    
        
        self.Path = Path
        self.model_list = model_list
        self.tot_num = tot_num
        self.ndims = ndims
        self.momentum = momentum
        self.epsilon = epsilon
        self.orthogonalization=orthogonalization
        self.loss_tracker_total = tf.keras.metrics.Mean(name="total_loss")
        self.corr_tracker = tf.keras.metrics.Mean(name="cross_metric")
        self.loss_tracker_mse = tf.keras.metrics.Mean(name="mse_loss")
        self.regularizer_list = regularizer_list

        if not run_from_config:
        # Add factor layer to each model in the list
            self.model_list = [self.add_DLVPM_layer(model, regularizer) for model, regularizer in zip(model_list, regularizer_list)]
        else:
            self.model_list = model_list
    
    def add_DLVPM_layer(self, model, regularizer):
        """
        Adds a FactorLayer on top of the given model.

        The method first checks whether the input model is sequential or functional,
        and then adds the FactorLayer in an appropriate way.

        :param model: A Keras/TensorFlow model (sequential or functional).
        :return: The model with an added FactorLayer on top.
        """
        if isinstance(model, tf.keras.Sequential):
            # For sequential models, we can just add a new layer on top
            if self.orthogonalization == 'Moore-Penrose':
                model.add(FactorLayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum,epsilon=self.epsilon))
            if self.orthogonalization == 'zca':
                model.add(ZCALayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum,epsilon=self.epsilon))
            else:
                print('Orthogonalization mode not recognised, must be "Moore-Penrose" or "zca"')
        elif isinstance(model, tf.keras.Model):
            # For functional models, we need to create a new model with the added layer
            if self.orthogonalization == 'Moore-Penrose':
                input = model.input
                x = FactorLayer(kernel_regularizer=regularizer,tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum,epsilon=self.epsilon)(model.output)
                model = tf.keras.Model(inputs=input, outputs=x)
            if self.orthogonalization == 'zca':
                input = model.input
                x = ZCALayer(kernel_regularizer=regularizer,tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum,epsilon=self.epsilon)(model.output)
                model = tf.keras.Model(inputs=input, outputs=x)
        else:
            raise ValueError("The input model must be either a tf.keras.Sequential or a tf.keras.Model instance.")
        return model

    
    def call(self,inputs):
        """
        Run data through each of the measurement sub-models.

        Args:
            inputs (list): A list of inputs for each data-view.

        Returns:
            tf.Tensor: The output of the model after processing the inputs.
        """

        inputs_nested = self.organize_inputs_by_model(inputs) ## this function organises flat inputs into a list of lists, which makes model training easier

        out=tf.stack([self.model_list[vie](inputs_nested[vie]) for vie in range(len(self.model_list))],axis=2) 
        
        # scale_fact = tf.cast(self.tot_num/tf.shape(out)[0],dtype=float) #
        # out = tf.divide(out,tf.math.sqrt(tf.math.multiply(scale_fact,tf.math.reduce_sum(tf.math.square(out),axis=0))))  ## re-normlise latent factors, very important!
        
        
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
        
        # Here, we run the current data-iteration through the global model in a forward 
        # pass. We do this so that we can re-normalise the weights. 
        is_training = False
        #
        y = self(inputs, training=is_training)  ## forward pass
        scale_fact = tf.cast(self.tot_num/tf.shape(y)[0],dtype=float) # scale factor for re-scaling
        y = tf.divide(y,tf.math.sqrt(tf.math.multiply(scale_fact,tf.math.reduce_sum(tf.math.square(y),axis=0)))) ## Here, we re-normalize DLVs

        total_loss = [None]*(len(self.model_list))
        total_CC = [None]*(len(self.model_list))
        total_mse = [None]*(len(self.model_list))
        
        inputs_nested = self.organize_inputs_by_model(inputs) ## this function organises flat inputs into a list of lists, which makes model training easier

        ## Iterate through training data-views
        for vie in range(len(self.model_list)):
           
        
            with tf.GradientTape() as tape:
                
                ## forward pass
                y_pred = self.model_list[vie](inputs_nested[vie], training=True)
                
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
        elif isinstance(optimizer,tf.keras.optimizers.Optimizer):
            for vie in range(len(self.model_list)):
                self.model_list[vie].compile(optimizer)
        else:
            print('Error: optimizer must either be of the tf.keras.optimizer class, or a list of objects of this class')
        

    def test_step(self, inputs):
        
        """ This step is called by model.evaluate() on a batch-wise level. This function
        returns loss metrics for the test data.
        
        """
        
        ## tensorflow packs inputs in another tuple, this should be unpacked
        inputs=inputs[0]

        #inputs = self.organize_inputs_by_model(inputs)
        
        y = self(inputs, training=False)  ## forward pass
    
        total_loss = [None]*(len(self.model_list))
        total_CC = [None]*(len(self.model_list))
        total_mse = [None]*(len(self.model_list))
        
        inputs_nested = self.organize_inputs_by_model(inputs)
        ## Iterate through training data-views
        for vie in range(len(self.model_list)):
          
                
            ## forward pass
            y_pred = self.model_list[vie](inputs_nested[vie], training=False)
            
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
        
        y_true =  tf.squeeze(tf.gather(y_true,tf.where(self.Path[vie,:]),axis=2),axis=3) ## select the latent factors connected to the latent factor for view vie
        
        y_pred = tf.expand_dims(y_pred,axis=2) ## expand dimensions of the predicted latent factor so broadcasting is possible
        
        return tf.reduce_sum(tf.math.reduce_sum(tf.math.square(tf.subtract(y_true,y_pred)),axis=0))
    
    def corr_metric(self,y_true,y_pred,vie):
        
        """ This function returns the mean correlation between the latent factors
        in a data-view, and the latent factors to which that data-view is connected 
        via the global PLS model.
        
        """
      
        y_true =  tf.squeeze(tf.gather(y_true,tf.where(self.Path[vie,:]),axis=2),axis=3) ## select the latent factors connected to the latent factor for view vie
        
        ## Minus the mean
        y_true_mean = tf.subtract(y_true,tf.math.reduce_mean(y_true,axis=0))
        y_pred_mean = tf.subtract(y_pred,tf.math.reduce_mean(y_pred,axis=0))
        
        # # ## Normalise matrices
        y_true_norm = tf.divide(y_true_mean,tf.norm(y_true_mean,axis=0))
        y_pred_norm = tf.divide(y_pred_mean,tf.norm(y_pred_mean,axis=0))
        
        y_pred_norm = tf.expand_dims(y_pred_norm,axis=2) ## expand dimensions of the predicted latent factor so broadcasting is possible
        
        corr2=tf.math.reduce_sum(tf.math.multiply(y_true_norm, y_pred_norm),axis=0)

        return tf.math.reduce_mean(corr2)
    
    import tensorflow as tf

    def calculate_corrmat(self, DLVs):
        """
        Compute Pearson correlation coefficient matrices for a 3D tensor.

        This function takes a 3D tensor of shape (n_samples, dimensions, DLVs) and computes
        the Pearson correlation coefficient between each pair of DLVs for each dimension. 
        The output is a list of symmetric matrices, one for each dimension, of shape (DLVs, DLVs).

        Args:
        DLVs (tf.Tensor): A 3D tensor of shape (n_samples, dimensions, DLVs).

        Returns:
        List[tf.Tensor]: A list of 2D tensors, each of shape (DLVs, DLVs), containing 
                        the Pearson correlation coefficients for each dimension.
        """
        # Ensure the input is a 3D tensor
        if len(DLVs.shape) != 3:
            raise ValueError("Input must be a 3D tensor")

        # List to store correlation matrices for each dimension
        correlation_matrices = []

        # Iterate through each dimension
        for dim in range(DLVs.shape[1]):
            # Select the data for the current dimension
            dim_DLVs = DLVs[:, dim, :]

            # Centering the DLVs by subtracting the mean
            mean_centered = dim_DLVs - tf.reduce_mean(dim_DLVs, axis=0)

            # Compute the standard deviation for each feature
            std_dev = tf.math.reduce_std(dim_DLVs, axis=0)

            # Normalize each feature
            normalized_DLVs = mean_centered / std_dev

            # Compute the correlation matrix for the current dimension
            correlation_matrix = tf.linalg.matmul(normalized_DLVs, normalized_DLVs, transpose_a=True) / tf.cast(tf.shape(dim_DLVs)[0], tf.float32)
            correlation_matrices.append(correlation_matrix)

        return correlation_matrices
    

    def plot_structural_model(self, outputname):
        """
        This function plots the structural/path. model. Visualisation is quite simple. 
        Aesthetics are similar to those used in tf.keras.utils.plot_model()
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
        serialized_model_list = [tf.keras.utils.serialize_keras_object(model) for model in self.model_list]
        regularized_model_list = [tf.keras.utils.serialize_keras_object(regularizer) for regularizer in self.regularizer_list]
        
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
        config['model_list'] = [tf.keras.saving.deserialize_keras_object(model_config) for model_config in config['model_list']]
        config['run_from_config'] = True
        
        # If regularization is present in the config, deserialize it
        if 'regularizer_list' in config:
            config['regularizer_list'] = [tf.keras.saving.deserialize_keras_object(regularizer_config) for regularizer_config in config['regularizer_list']]
        
        return cls(**config)
    
    def get_compile_config(self):
        """
        Serializes the optimizer configurations of the models.

        Returns:
            dict: A dictionary containing the serialized optimizer configurations of the models.
        """
        return {
            "model_optimizers": [tf.keras.saving.serialize_keras_object(model.optimizer) for model in self.model_list]
        }
    
    def compile_from_config(self, config):
        """
        Compiles the models with the deserialized optimizer configurations.

        Args:
            config (dict): A dictionary containing the serialized optimizer configurations.
        """
        optimizer_list = [tf.keras.saving.deserialize_keras_object(optimizer_config) for optimizer_config in config["model_optimizers"]]
        self.compile(optimizer_list)

    def build_from_config(self, config):
        """ build is overwritten here as it is not needed. Individual measurement models
        are built seperately, this happens when tf.keras.saving.deserialize_keras_object is called
        on models in model_list"""

        return
        #self.build(config["input_shape"])
    








