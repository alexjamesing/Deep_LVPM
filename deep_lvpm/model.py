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

be = keras.backend.backend()  # 'tensorflow' | 'torch' | 'jax' (we handle tf/torch)

if be == "tensorflow":
    try:
        import tensorflow as tf  # lazy import
    except ImportError as e:    
        raise RuntimeError(
            "Tensorflow backend requested but it is not installed. "
            "Install Tensorflow or switch Keras backend to Torch."
        ) from e

elif be == "torch":
    try:
        import torch  # lazy import
    except ImportError as e:
        raise RuntimeError(
            "Torch backend requested but it is not installed. "
            "Install Torch or switch Keras backend to TensorFlow."
        ) from e



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

    
    def __init__(self, Path, model_list, regularizer_list, tot_num, ndims, orthogonalization='Moore-Penrose', momentum=0.95, epsilon=1e-4, train_DLV=True, run_from_config=False, is_siamese=False, diag_offset=1e-3, sparse_l1_list=0.0, **kwargs):
        
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
        self.is_siamese = is_siamese
        self.diag_offset = diag_offset
        # Normalise sparse_l1_list to a per-view float list
        # Accept scalar (broadcast) or list-like length == n_views
        n_views = len(model_list)
        if sparse_l1_list is None:
            norm_sparse = [0.0] * n_views
        elif isinstance(sparse_l1_list, (list, tuple, np.ndarray)):
            norm_sparse = [float(x) for x in list(sparse_l1_list)]
            if len(norm_sparse) != n_views:
                raise ValueError(f"sparse_l1_list must have length {n_views}, got {len(norm_sparse)}")
        else:
            norm_sparse = [float(sparse_l1_list)] * n_views

        if self.is_siamese and any(abs(x - norm_sparse[0]) > 0.0 for x in norm_sparse):
            raise ValueError("In siamese mode, all entries of sparse_l1_list must be identical.")

        self.sparse_l1_list = norm_sparse

        if not run_from_config:
        # Add factor layer to each model in the list
            if self.is_siamese == True:
                new_model = self.add_DLVPM_layer(model_list[0], regularizer_list[0], self.sparse_l1_list[0])
                self.model_list = [new_model] * len(model_list)   # duplicates the *reference*
            else:
                self.model_list = [
                    self.add_DLVPM_layer(model, regularizer, sparse_l1)
                    for model, regularizer, sparse_l1 in zip(model_list, regularizer_list, self.sparse_l1_list)
                ]
        else:
            self.model_list = model_list

        self.loss_tracker_total = keras.metrics.Mean(name="total_loss")
        self.corr_tracker = keras.metrics.Mean(name="cross_metric")
        self.loss_tracker_mse = keras.metrics.Mean(name="mse_loss")
        self.loss_tracker_redundancy = keras.metrics.Mean(name="redundancy")


    
    def add_DLVPM_layer(self, model, regularizer, sparse_l1):
        """
        Adds a FactorLayer on top of the given model.

        The method first checks whether the input model is sequential or functional,
        and then adds the FactorLayer in an appropriate way.

        :param model: A Keras/TensorFlow model (sequential or functional).
        :return: The model with an added FactorLayer on top.
        """
        if isinstance(model, keras.Sequential):
            if self.orthogonalization == 'Moore-Penrose':
                model.add(FactorLayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon, sparse_l1=sparse_l1))
            elif self.orthogonalization == 'zca':
                model.add(ZCALayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon, diag_offset = self.diag_offset, sparse_l1=sparse_l1))
            else:
                print('Orthogonalization mode not recognised, must be "Moore-Penrose" or "zca"')
        elif isinstance(model, keras.Model):
            if self.orthogonalization == 'Moore-Penrose':
                x = FactorLayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon, sparse_l1=sparse_l1)(model.output)
                model = keras.Model(inputs=model.input, outputs=x)
            elif self.orthogonalization == 'zca':
                x = ZCALayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon, diag_offset = self.diag_offset, sparse_l1=sparse_l1)(model.output)
                model = keras.Model(inputs=model.input, outputs=x)
            else:
                print('Orthogonalization mode not recognised, must be "Moore-Penrose" or "zca"')
        else:
            raise ValueError("The input model must be either a keras.Sequential or a keras.Model instance.")

        
        return model
    

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


    def _normalize_pred(self, y_pred, scale_fact):
        eps = getattr(self, "epsilon", 1e-8)
        eps = ops.convert_to_tensor(eps, dtype=ops.dtype(y_pred))
        # y_pred / (sqrt(scale_fact) * ||y_pred||_2 over batch)
        denom = ops.sqrt(scale_fact) * ops.sqrt(ops.sum(ops.square(y_pred), axis=0) + eps)
        return y_pred / denom

    def _step_tf(self, vie, inputs_v, y, scale_fact):
        """This is the training step for the tensorflow backend"""

        model = self.model_list[vie]
        with tf.GradientTape() as tape:
            y_pred = model(inputs_v, training=True)
            y_pred = self._normalize_pred(y_pred, scale_fact)
            mse_loss = self.mse_loss(y, y_pred, vie)
            internal_loss = tf.add_n(model.losses) if model.losses else tf.cast(0.0, mse_loss.dtype)
            loss = mse_loss + internal_loss

        trainable_vars = model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        model.optimizer.apply_gradients(zip(grads, trainable_vars))
        corr = self.corr_metric(y, y_pred, vie)

        return loss, mse_loss, corr

    def _step_torch(self, vie, inputs_v, y, scale_fact):
        """This is the training step for the Torch backend"""

        model = self.model_list[vie]

        # Forward pass (PyTorch autograd records ops by default)
        y_pred = model(inputs_v, training=True)
        y_pred = self._normalize_pred(y_pred, scale_fact)
        mse_loss = self.mse_loss(y, y_pred, vie)

        if model.losses:
            internal_loss = torch.stack(
                [
                    l
                    if torch.is_tensor(l)
                    else torch.tensor(l, dtype=mse_loss.dtype, device=mse_loss.device)
                    for l in model.losses
                ]
            ).sum()
        else:
            internal_loss = torch.zeros((), dtype=mse_loss.dtype, device=mse_loss.device)
                              
        loss = mse_loss + internal_loss

        # Compute grads w.r.t. model variables and apply
        trainable_vars = model.trainable_variables
        # Some backends wrap the underlying torch tensor on .value; handle both.
        vars_for_grad = [getattr(v, "value", v) for v in trainable_vars]
        grads = torch.autograd.grad(
            loss,
            vars_for_grad,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        # Replace None grads with zeros (can happen for detached vars / unused params)
        fixed_grads = [
            g if g is not None else torch.zeros_like(getattr(v, "value", v))
            for g, v in zip(grads, trainable_vars)
        ]
        model.optimizer.apply_gradients(zip(fixed_grads, trainable_vars))

        corr = self.corr_metric(y, y_pred, vie)
        return loss, mse_loss, corr

    def _weight_normaliser(self,inputs):
        """This is an internal function designed to normalise weights
        after each batch"""
        
         # Forward pass through all views to construct global DLVs
        y = self(inputs, training=self.train_DLV)

        # scale_fact = tot_num / batch_size
        y_dtype = ops.dtype(y)
        scale_fact = ops.cast(self.tot_num, y_dtype) / ops.cast(self._shape_fn(y)[0], y_dtype)

        # per-view normalization via last layer's weight_normalizer
        y_list = []
        for vie in range(len(self.model_list)):
            y_view = y[:, :, vie]
            y_view = self.model_list[vie].layers[-1].weight_normalizer([y_view, scale_fact, self.train_DLV])
            y_list.append(y_view)

        y = ops.stack(y_list, axis=-1)

        return y, scale_fact


    def train_step(self, inputs):
        """This is the main training set, it runs differently in tensorflow and torch"""



        # Unpack (tf.data-like packs inputs in a tuple/list)
        inputs = inputs[0]

        be = keras.backend.backend()  # 'tensorflow' | 'torch' | 'jax' (we handle tf/torch)

        if be == "tensorflow":
            y, scale_fact = self._weight_normaliser(inputs)
        elif be == "torch":
            with torch.no_grad():
                y, scale_fact = self._weight_normaliser(inputs)
            

        total_loss = [None] * len(self.model_list)
        total_CC   = [None] * len(self.model_list)
        total_mse  = [None] * len(self.model_list)
        total_redundancy = [None] * len(self.model_list)

        inputs_nested = self.organize_inputs_by_model(inputs)
        
        be = keras.backend.backend()  # 'tensorflow' | 'torch' | 'jax' (we handle tf/torch)

        for vie in range(len(self.model_list)):
            if be == "tensorflow":
                loss, mse_loss, corr = self._step_tf(vie, inputs_nested[vie], y, scale_fact)
            elif be == "torch":
                loss, mse_loss, corr = self._step_torch(vie, inputs_nested[vie], y, scale_fact)
            else:
                raise NotImplementedError(f"Backend '{be}' not supported in custom train_step.")

            total_loss[vie] = ops.sum(loss)
            total_CC[vie]   = corr
            total_mse[vie]  = mse_loss
            total_redundancy[vie] = self.calculate_redundancy(y[:,:,vie])

        self.loss_tracker_total.update_state(ops.stack(total_loss))
        self.corr_tracker.update_state(ops.stack(total_CC))
        self.loss_tracker_mse.update_state(ops.stack(total_mse))
        self.loss_tracker_redundancy.update_state(ops.stack(total_redundancy))

        return {
            "total_loss": self.loss_tracker_total.result(),
            "cross_metric": self.corr_tracker.result(),
            "mse_loss": self.loss_tracker_mse.result(),
            "redundancy": self.loss_tracker_redundancy.result()
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
        elif isinstance(optimizer,keras.optimizers.Optimizer): ## This case is important when running a siamese network on one data-view
            for vie in range(len(self.model_list)):
                self.model_list[0].compile(optimizer)
        else:
            print('Error: optimizer must either be of the keras.optimizer class, or a list of objects of this class')
        


    def test_step(self, inputs):
        """ This step is called by model.evaluate() on a batch-wise level. This function
        returns loss metrics for the test data.
        """

        inputs = inputs[0]
        y = self(inputs, training=False)

        total_loss = [None] * len(self.model_list)
        total_CC = [None] * len(self.model_list)
        total_mse = [None] * len(self.model_list)
        total_redundancy = [None] * len(self.model_list)

        inputs_nested = self.organize_inputs_by_model(inputs)

        for vie in range(len(self.model_list)):
            y_pred = self.model_list[vie](inputs_nested[vie], training=False)

            mse_loss = self.mse_loss(y, y_pred, vie)
            
            internal_losses = self.model_list[vie].losses
            if internal_losses:
                internal_loss = ops.sum(
                    ops.stack(
                        [ops.convert_to_tensor(loss, dtype=ops.dtype(mse_loss)) for loss in internal_losses],
                        axis=0,
                    ),
                    axis=0,
                )
            else:
                internal_loss = ops.zeros_like(mse_loss)

            loss = mse_loss + internal_loss

            corr = self.corr_metric(y, y_pred, vie)

            total_loss[vie] = ops.sum(loss)
            total_CC[vie]   = corr
            total_mse[vie]  = mse_loss
            total_redundancy[vie] = self.calculate_redundancy(y[:,:,vie])

        self.loss_tracker_total.update_state(ops.stack(total_loss))
        self.corr_tracker.update_state(ops.stack(total_CC))
        self.loss_tracker_mse.update_state(ops.stack(total_mse))
        self.loss_tracker_redundancy.update_state(ops.stack(total_redundancy))

        return {
            "total_loss": self.loss_tracker_total.result(),
            "cross_metric": self.corr_tracker.result(),
            "mse_loss": self.loss_tracker_mse.result(),
            "redundancy": self.loss_tracker_redundancy.result()
        }


    @property
    def metrics(self):
        """We list our `Metric` objects here so that `reset_states()` can be
        called automatically at the start of each epoch
        or at the start of `evaluate()`."""

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
    
     # This function avoids problems with passing symbolic 
     # tensors to ops.shape in tensorflow

    def _shape_fn(self,X):
        backend = keras.backend.backend()
        if backend == "tensorflow":
            shape = tf.shape(X)  # handles unknown ranks
        else:
            shape = ops.shape(X)
        return shape
    

    def calculate_redundancy(self, Y, epsilon=1e-8):
        """
        Args:
            X: Tensor / KerasTensor, shape (N, D). Each column is a variable.
            epsilon: Small constant for numerical stability.

        Returns:
            Scalar tensor: mean(|corr(i, j)|) over all i != j.
        """
        Y = ops.convert_to_tensor(Y)
        Y = ops.cast(Y, "float32")

        # Center columns
        col_mean = ops.mean(Y, axis=0, keepdims=True)
        Yc = Y - col_mean

        backend = keras.backend.backend()

        # Sample-size for covariance
        n = self._shape_fn(Yc)[0]
        n_f = ops.cast(n, Y.dtype)
        denom_n = ops.maximum(n_f - 1.0, 1.0)  # guard when N == 1

        # Covariance between columns: (D x D)
        cov = ops.matmul(ops.transpose(Yc), Yc) / denom_n

        # Column std devs (D,)
        var = ops.sum(Yc * Yc, axis=0) / denom_n
        std = ops.sqrt(ops.maximum(var, epsilon))

        # Correlation matrix: cov / (std_i * std_j)
        std_col = ops.reshape(std, (-1, 1))              # (D,1)
        denom = std_col * ops.transpose(std_col)         # (D,D)
        corr = cov / ops.maximum(denom, epsilon)         # (D,D)

        # Mean absolute correlation over off-diagonal entries
        corr_abs = ops.abs(corr)
        D = self._shape_fn(corr_abs)[0]
        mask = ops.ones_like(corr_abs) - ops.cast(ops.eye(D), corr_abs.dtype)  # zero diagonal
        total = ops.sum(corr_abs * mask)

        D_f = ops.cast(D, corr_abs.dtype)
        num_pairs = ops.maximum(D_f * (D_f - 1.0), 1.0)  # count of off-diagonal entries

        return total / num_pairs



    from keras import ops

    def calculate_corrmat(self, DLVs):
        """
        Compute Pearson correlation matrices for a 3D tensor using keras.ops.
        DLVs: (n_samples, dimensions, DLVs)
        Returns: list of (DLVs x DLVs) per dimension.
        """
        if len(DLVs.shape) != 3:
            raise ValueError("Input must be a 3D tensor")

        # ✅ Ensure we’re working with a backend tensor, even if DLVs was numpy
        DLVs = ops.convert_to_tensor(DLVs)

        correlation_matrices = []
        n_samples = ops.cast(self._shape_fn(DLVs)[0], DLVs.dtype)
        eps = ops.convert_to_tensor(1e-7, dtype=DLVs.dtype)

        # Use shape function to be backend-friendly
        n_dims = int(self._shape_fn(DLVs)[1])

        for dim in range(n_dims):
            dim_DLVs = DLVs[:, dim, :]  # (n_samples, n_feats)
            mean_centered = dim_DLVs - ops.mean(dim_DLVs, axis=0)
            std_dev = ops.std(dim_DLVs, axis=0) + eps
            normalized = mean_centered / std_dev
            correlation_matrix = ops.matmul(
                ops.transpose(normalized),
                normalized
            ) / n_samples
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
            "orthogonalization": self.orthogonalization,
            "sparse_l1_list": self.sparse_l1_list,
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
        config['Path'] = ops.convert_to_tensor(config["Path"], dtype=ops.floatx())
        
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
    
