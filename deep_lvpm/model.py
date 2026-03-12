# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# This script creates a custom Keras model for identifying correlated factors
# (deep latent variables) between different data types. It is designed to work with different
# data-views, and it establishes associations between these views using deep latent
# variables. The data-views we wish to optimise associations between are defined using an 
# adjacency matrix.
# """

# import os
# import numpy as np
# import keras as keras
# from deep_lvpm.layers.FactorLayer import FactorLayer
# from deep_lvpm.layers.ZCALayer import ZCALayer
# from deep_lvpm.layers.ConfoundLayer import ConfoundLayer
# import pydot
# from keras import ops

# be = keras.backend.backend()  # 'tensorflow' | 'torch' | 'jax' (we handle tf/torch)

# if be == "tensorflow":
#     try:
#         import tensorflow as tf  # lazy import
#     except ImportError as e:    
#         raise RuntimeError(
#             "Tensorflow backend requested but it is not installed. "
#             "Install Tensorflow or switch Keras backend to Torch."
#         ) from e

# elif be == "torch":
#     try:
#         import torch  # lazy import
#     except ImportError as e:
#         raise RuntimeError(
#             "Torch backend requested but it is not installed. "
#             "Install Torch or switch Keras backend to TensorFlow."
#         ) from e



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

    
#     def __init__(self, Path, model_list, regularizer_list, tot_num, ndims, orthogonalization='Moore-Penrose', momentum=0.95, epsilon=1e-4, train_DLV=True, run_from_config=False, is_siamese=False, diag_offset=1e-3, sparse_l1_list=0.0, order=False, order_every=100, **kwargs):
        
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
#             order (bool): If True and orthogonalization == 'zca', enable ordering.
#             order_every (int|str): Apply rotation every N batches (default 100) using
#                 moving consensus; pass 'end' to rotate only after the very last batch;
#                 ignored if order is False or orthogonalization != 'zca'.
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
#         self.is_siamese = is_siamese
#         self.diag_offset = diag_offset
#         # Whether to run order_variates (only valid for 'zca')
#         self.order = bool(order)
#         if self.order and self.orthogonalization != 'zca':
#             raise ValueError("'order' is only available when orthogonalization='zca'.")
#         # Rotation interval (in batches)
#         if isinstance(order_every, str) and order_every.lower() == 'end':
#             self.order_every = 'end'
#         else:
#             try:
#                 self.order_every = int(order_every)
#             except Exception:
#                 self.order_every = 100
#             if self.order_every <= 0:
#                 self.order_every = 100
#         self._moving_cov = None
#         # Accumulators for consensus covariance stats (for per-batch final-order logic)
#         self._acc_AtA = None
#         self._acc_sumA = None
#         self._acc_count = None

#         # Normalise sparse_l1_list to a per-view float list
#         # Accept scalar (broadcast) or list-like length == n_views
#         n_views = len(model_list)
#         if sparse_l1_list is None:
#             norm_sparse = [0.0] * n_views
#         elif isinstance(sparse_l1_list, (list, tuple, np.ndarray)):
#             norm_sparse = [float(x) for x in list(sparse_l1_list)]
#             if len(norm_sparse) != n_views:
#                 raise ValueError(f"sparse_l1_list must have length {n_views}, got {len(norm_sparse)}")
#         else:
#             norm_sparse = [float(sparse_l1_list)] * n_views

#         if self.is_siamese and any(abs(x - norm_sparse[0]) > 0.0 for x in norm_sparse):
#             raise ValueError("In siamese mode, all entries of sparse_l1_list must be identical.")

#         self.sparse_l1_list = norm_sparse

#         if not run_from_config:
#         # Add factor layer to each model in the list
#             if self.is_siamese == True:
#                 new_model = self.add_DLVPM_layer(model_list[0], regularizer_list[0], self.sparse_l1_list[0])
#                 self.model_list = [new_model] * len(model_list)   # duplicates the *reference*
#             else:
#                 self.model_list = [
#                     self.add_DLVPM_layer(model, regularizer, sparse_l1)
#                     for model, regularizer, sparse_l1 in zip(model_list, regularizer_list, self.sparse_l1_list)
#                 ]
#         else:
#             self.model_list = model_list

#         self.loss_tracker_total = keras.metrics.Mean(name="total_loss")
#         self.corr_tracker = keras.metrics.Mean(name="cross_metric")
#         self.loss_tracker_mse = keras.metrics.Mean(name="mse_loss")
#         self.loss_tracker_redundancy = keras.metrics.Mean(name="redundancy")
#         # Tracks Spearman rank alignment of mean per-dimension correlations across views
#         self.order_rank_tracker = keras.metrics.Mean(name="order_rank")

#     def build(self, input_shape):

        
#         super().build(input_shape)
#         initializer = keras.initializers.Identity()
#         self._moving_cov = self.add_weight(
#             name="structural_moving_cov",
#             shape=(self.ndims, self.ndims),
#             initializer=initializer,
#             trainable=False,
#         )
#         # Track standardized consensus correlation (moving), akin to ZCALayer.moving_conv2
#         self.moving_consensus = self.add_weight(
#             name="moving_consensus",
#             shape=(self.ndims, self.ndims),
#             initializer="zeros",
#             trainable=False,
#         )
#         self.moving_consensus.assign(ops.eye(self.ndims, self.ndims))
#         # Accumulators for consensus covariance across batches (non-trainable)



#         self._acc_AtA = self.add_weight(
#             name="acc_AtA",
#             shape=(self.ndims, self.ndims),
#             initializer="zeros",
#             trainable=False,
#         )
#         self._acc_sumA = self.add_weight(
#             name="acc_sumA",
#             shape=(self.ndims,),
#             initializer="zeros",
#             trainable=False,
#         )
#         self._acc_count = self.add_weight(
#             name="acc_count",
#             shape=(),
#             initializer="zeros",
#             trainable=False,
#         )


    
#     def add_DLVPM_layer(self, model, regularizer, sparse_l1):
#         """
#         Adds a FactorLayer on top of the given model.

#         The method first checks whether the input model is sequential or functional,
#         and then adds the FactorLayer in an appropriate way.

#         :param model: A Keras/TensorFlow model (sequential or functional).
#         :return: The model with an added FactorLayer on top.
#         """
#         if isinstance(model, keras.Sequential):
#             if self.orthogonalization == 'Moore-Penrose':
#                 model.add(FactorLayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon, sparse_l1=sparse_l1))
#             elif self.orthogonalization == 'zca':
#                 model.add(ZCALayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon, diag_offset = self.diag_offset, sparse_l1=sparse_l1))
#             else:
#                 print('Orthogonalization mode not recognised, must be "Moore-Penrose" or "zca"')
#         elif isinstance(model, keras.Model):
#             if self.orthogonalization == 'Moore-Penrose':
#                 x = FactorLayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon, sparse_l1=sparse_l1)(model.output)
#                 model = keras.Model(inputs=model.input, outputs=x)
#             elif self.orthogonalization == 'zca':
#                 x = ZCALayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon, diag_offset = self.diag_offset, sparse_l1=sparse_l1)(model.output)
#                 model = keras.Model(inputs=model.input, outputs=x)
#             else:
#                 print('Orthogonalization mode not recognised, must be "Moore-Penrose" or "zca"')
#         else:
#             raise ValueError("The input model must be either a keras.Sequential or a keras.Model instance.")

        
#         return model
    

#     def call(self, inputs, training=False):

#         """
#     #     Run data through each of the measurement sub-models.

#     #     Args:
#     #         inputs (list): A list of inputs for each data-view.
#     #         training: Whether to call the model in training or inference mode. Can take values of True or False.

#     #     Returns:
#     #         The output of the model after processing the inputs.
#     #     """


#         inputs_nested = self.organize_inputs_by_model(inputs)
#         out = ops.stack(
#             [self.model_list[vie](inputs_nested[vie], training=training)
#             for vie in range(len(self.model_list))],
#             axis=2
#             )
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


#     def _normalize_pred(self, y_pred, scale_fact):
#         eps = getattr(self, "epsilon", 1e-8)
#         eps = ops.convert_to_tensor(eps, dtype=ops.dtype(y_pred))
#         # y_pred / (sqrt(scale_fact) * ||y_pred||_2 over batch)
#         denom = ops.sqrt(scale_fact) * ops.sqrt(ops.sum(ops.square(y_pred), axis=0) + eps)
#         return y_pred / denom

#     def _step_tf(self, vie, inputs_v, y, scale_fact):
#         """This is the training step for the tensorflow backend"""

#         model = self.model_list[vie]
#         with tf.GradientTape() as tape:
#             y_pred = model(inputs_v, training=True)
#             y_pred = self._normalize_pred(y_pred, scale_fact)
#             mse_loss = self.mse_loss(y, y_pred, vie)
#             internal_loss = tf.add_n(model.losses) if model.losses else tf.cast(0.0, mse_loss.dtype)
#             loss = mse_loss + internal_loss

#         trainable_vars = model.trainable_variables
#         grads = tape.gradient(loss, trainable_vars)
#         model.optimizer.apply_gradients(zip(grads, trainable_vars))
#         corr = self.corr_metric(y, y_pred, vie)

#         return loss, mse_loss, corr

#     def _step_torch(self, vie, inputs_v, y, scale_fact):
#         """This is the training step for the Torch backend"""

#         model = self.model_list[vie]

#         # Forward pass (PyTorch autograd records ops by default)
#         y_pred = model(inputs_v, training=True)
#         y_pred = self._normalize_pred(y_pred, scale_fact)
#         mse_loss = self.mse_loss(y, y_pred, vie)

#         if model.losses:
#             internal_loss = torch.stack(
#                 [
#                     l
#                     if torch.is_tensor(l)
#                     else torch.tensor(l, dtype=mse_loss.dtype, device=mse_loss.device)
#                     for l in model.losses
#                 ]
#             ).sum()
#         else:
#             internal_loss = torch.zeros((), dtype=mse_loss.dtype, device=mse_loss.device)
                              
#         loss = mse_loss + internal_loss

#         # Compute grads w.r.t. model variables and apply
#         trainable_vars = model.trainable_variables
#         # Some backends wrap the underlying torch tensor on .value; handle both.
#         vars_for_grad = [getattr(v, "value", v) for v in trainable_vars]
#         grads = torch.autograd.grad(
#             loss,
#             vars_for_grad,
#             retain_graph=False,
#             create_graph=False,
#             allow_unused=True,
#         )
#         # Replace None grads with zeros (can happen for detached vars / unused params)
#         fixed_grads = [
#             g if g is not None else torch.zeros_like(getattr(v, "value", v))
#             for g, v in zip(grads, trainable_vars)
#         ]
#         model.optimizer.apply_gradients(zip(fixed_grads, trainable_vars))

#         corr = self.corr_metric(y, y_pred, vie)
#         return loss, mse_loss, corr

#     def _weight_normaliser(self,inputs):
#         """This is an internal function designed to normalise weights
#         after each batch"""
        
#          # Forward pass through all views to construct global DLVs
#         y = self(inputs, training=self.train_DLV)

#         # scale_fact = tot_num / batch_size
#         y_dtype = ops.dtype(y)
#         scale_fact = ops.cast(self.tot_num, y_dtype) / ops.cast(self._shape_fn(y)[0], y_dtype)

#         # per-view normalization via last layer's weight_normalizer
#         y_list = []
#         for vie in range(len(self.model_list)):
#             y_view = y[:, :, vie]
#             y_view = self.model_list[vie].layers[-1].weight_normalizer([y_view, scale_fact, self.train_DLV])
#             y_list.append(y_view)

#         y = ops.stack(y_list, axis=-1)

#         return y, scale_fact

#     def _update_moving_consensus(self, Y):
#         """Update moving consensus correlation from inference-mode Y.

#         Args:
#             Y: Tensor of shape (N, D, K) (or (N, D) if single-view)
#         """
#         # scale_factor = tot_num / batch_size
#         y_dtype = ops.dtype(Y)
#         n = ops.cast(self._shape_fn(Y)[0], y_dtype)
#         scale_fact = ops.cast(self.tot_num, y_dtype) / ops.maximum(n, ops.cast(1.0, y_dtype))

#         # Consensus A = sum over views; center across samples
#         A = Y if ops.ndim(Y) == 2 else ops.sum(Y, axis=-1)  # (N, D)
#         mu = ops.mean(A, axis=0, keepdims=True)
#         A_c = A - mu

#         # Covariance proxy with scale
#         cov = scale_fact * ops.matmul(ops.transpose(A_c), A_c)  # (D, D)

#         # Standardize to correlation-like matrix
#         D_dim = self._shape_fn(cov)[0]
#         I = ops.eye(D_dim, dtype=ops.dtype(cov))
#         eps = ops.convert_to_tensor(getattr(self, "epsilon", 1e-8), dtype=ops.dtype(cov))
#         var = ops.sum(cov * I, axis=1)
#         inv_std = 1.0 / ops.sqrt(ops.maximum(var, eps))
#         inv_std_col = ops.expand_dims(inv_std, axis=1)
#         corr = inv_std_col * cov * ops.transpose(inv_std_col)

#         # Momentum update
#         momentum = ops.convert_to_tensor(self.momentum, dtype=self.moving_consensus.dtype)
#         one = ops.convert_to_tensor(1.0, dtype=self.moving_consensus.dtype)
#         self.moving_consensus.assign(
#             momentum * self.moving_consensus + (one - momentum) * ops.cast(corr, self.moving_consensus.dtype)
#         )

#     # --- Small helpers to avoid duplicating SVD + rotation boilerplate ---
#     def _compute_rotation_from_moving(self):
#         """Return rotation matrix V from current moving_consensus via SVD.

#         Returns V (ndims x ndims) or raises if SVD fails.
#         """
#         M = self.moving_consensus
#         _, _, Vt = ops.linalg.svd(M, full_matrices=False)
#         return ops.transpose(Vt)

#     def _apply_rotation_to_models(self, V):
#         """Right-multiply the projection of each unique measurement model by V."""
#         seen = set()
#         for mdl in self.model_list:
#             mid = id(mdl)
#             if mid in seen:
#                 continue
#             seen.add(mid)
#             last = mdl.layers[-1]
#             last.project.assign(ops.matmul(last.project, V))

#     def _reset_moving_consensus(self):
#         """Reset moving_consensus to identity."""
#         ident = ops.eye(self.ndims, self.ndims, dtype=self.moving_consensus.dtype)
#         self.moving_consensus.assign(ident)

#     def _order_pairwise_tau(self, vec):
#         """Pairwise order agreement (Kendall-tau-like) vs descending index order.

#         Computes, over all pairs i<j, the fraction of concordant minus
#         discordant pairs where desired order is vec[i] >= vec[j]. Ties contribute 0.
#         Returns value in [-1, 1].
#         """
#         vec = ops.convert_to_tensor(vec)
#         D_dim = self._shape_fn(vec)[0]
#         # Pairwise differences
#         vdiff = ops.expand_dims(vec, 1) - ops.expand_dims(vec, 0)  # (D,D)
#         # Upper-triangular mask (i<j)
#         ones = ops.ones_like(vec)
#         idx = ops.cumsum(ones, axis=0) - 1.0  # [0..D-1]
#         idiff = ops.expand_dims(idx, 1) - ops.expand_dims(idx, 0)
#         upper = ops.cast(idiff < 0, ops.dtype(vdiff))
#         # Sign of differences: +1 concordant, -1 discordant, 0 ties
#         signs = ops.sign(vdiff) * upper
#         num = ops.sum(signs)
#         n_pairs = ops.sum(upper)
#         n_pairs = ops.maximum(n_pairs, ops.convert_to_tensor(1.0, dtype=ops.dtype(vdiff)))
#         # Normalize and clip to [-1, 1]
#         tau = num / n_pairs
#         tau = ops.clip(tau, -1.0, 1.0)
#         return tau

#     def _order_monotonic_pearson(self, vec):
#         """Pearson correlation between vec and a strictly descending index 1..D.

#         Values near 1 indicate vec is strongly decreasing with index (best-first),
#         -1 indicates increasing with index (worst-first), and 0 indicates no
#         linear trend. Graph-safe and tie-robust.
#         """
#         v = ops.convert_to_tensor(vec)
#         D_dim = self._shape_fn(v)[0]
#         # Build index 1..D in graph-friendly way
#         idx = ops.cumsum(ops.ones_like(v), axis=0)  # 1..D
#         # Center both
#         v_c = v - ops.mean(v)
#         i_c = idx - ops.mean(idx)
#         num = ops.sum(v_c * i_c)
#         den = ops.sqrt(ops.sum(v_c * v_c)) * ops.sqrt(ops.sum(i_c * i_c))
#         den = ops.maximum(den, ops.convert_to_tensor(1e-12, dtype=ops.dtype(v)))
#         rho = num / den
#         # We want descending match to map to +1; idx ascending gives positive when v decreases.
#         # If you prefer descending index explicitly, flip sign: rho_desc = -rho
#         return -rho


#     def train_step(self, inputs):
#         """This is the main training set, it runs differently in tensorflow and torch"""



#         # Unpack (tf.data-like packs inputs in a tuple/list)
#         inputs = inputs[0]

#         # Update moving consensus correlation using inference-mode outputs
#         try:
#             y_cons = self(inputs, training=False)
#             self._update_moving_consensus(y_cons)
#         except Exception:
#             y_cons = None

#         # Metrics derived directly from inference-mode outputs (y_cons)
#         cons_CC = None
#         cons_red = None
#         cons_per_view_dim_means = None
#         if y_cons is not None:
#             inputs_nested_cons = self.organize_inputs_by_model(inputs)
#             eps_cons = ops.convert_to_tensor(self.epsilon, dtype=ops.dtype(y_cons))
#             y_true_c_cons = y_cons - ops.mean(y_cons, axis=0)
#             denom_true_cons = ops.sqrt(ops.sum(ops.square(y_true_c_cons), axis=0) + eps_cons)
#             y_true_n_cons = y_true_c_cons / denom_true_cons

#             cons_CC = []
#             cons_red = []
#             cons_per_view_dim_means = []
#             for vie in range(len(self.model_list)):
#                 y_pred_eval_cons = self.model_list[vie](inputs_nested_cons[vie], training=False)
#                 cons_CC.append(self.corr_metric(y_cons, y_pred_eval_cons, vie))
#                 cons_red.append(self.calculate_redundancy(y_cons[:, :, vie]))

#                 y_pred_c_cons = y_pred_eval_cons - ops.mean(y_pred_eval_cons, axis=0)
#                 denom_pred_cons = ops.sqrt(ops.sum(ops.square(y_pred_c_cons), axis=0) + eps_cons)
#                 y_pred_n_cons = y_pred_c_cons / denom_pred_cons
#                 y_pred_n_cons = ops.expand_dims(y_pred_n_cons, axis=2)
#                 corr_mat_cons = ops.sum(y_true_n_cons * y_pred_n_cons, axis=0)
#                 mask_cons = ops.cast(self.Path[vie, :], ops.dtype(corr_mat_cons))
#                 corr_masked_cons = corr_mat_cons * ops.expand_dims(mask_cons, axis=0)
#                 n_conn_cons = ops.sum(mask_cons)
#                 n_conn_safe_cons = ops.maximum(n_conn_cons, ops.convert_to_tensor(1.0, dtype=ops.dtype(n_conn_cons)))
#                 per_dim_mean_cons = ops.sum(corr_masked_cons, axis=1) / n_conn_safe_cons
#                 cons_per_view_dim_means.append(per_dim_mean_cons)

#         be = keras.backend.backend()  # 'tensorflow' | 'torch' | 'jax' (we handle tf/torch)

#         if be == "tensorflow":
#             y, scale_fact = self._weight_normaliser(inputs)
#             # Also cache inference-mode DLVs for ordering stats (pre-update, as before)
#         elif be == "torch":
#             with torch.no_grad():
#                 y, scale_fact = self._weight_normaliser(inputs)
           

#         total_loss = [None] * len(self.model_list)
#         total_CC   = [None] * len(self.model_list)
#         total_mse  = [None] * len(self.model_list)
#         total_redundancy = [None] * len(self.model_list)

#         inputs_nested = self.organize_inputs_by_model(inputs)
        
#         be = keras.backend.backend()  # 'tensorflow' | 'torch' | 'jax' (we handle tf/torch)

#         per_view_dim_means = []
#         for vie in range(len(self.model_list)):
#             if be == "tensorflow":
#                 loss, mse_loss, corr = self._step_tf(vie, inputs_nested[vie], y, scale_fact)
#             elif be == "torch":
#                 loss, mse_loss, corr = self._step_torch(vie, inputs_nested[vie], y, scale_fact)
#             else:
#                 raise NotImplementedError(f"Backend '{be}' not supported in custom train_step.")

#             total_loss[vie] = ops.sum(loss)
#             total_CC[vie]   = corr
#             total_mse[vie]  = mse_loss
#             total_redundancy[vie] = self.calculate_redundancy(y[:,:,vie])

#         # Rotation is handled at end of fit via callbacks; do not rotate per batch here

#         self.loss_tracker_total.update_state(ops.stack(total_loss))
#         if cons_CC is not None:
#             self.corr_tracker.update_state(ops.stack(cons_CC))
#         else:
#             self.corr_tracker.update_state(ops.stack(total_CC))
#         self.loss_tracker_mse.update_state(ops.stack(total_mse))
#         if cons_red is not None:
#             self.loss_tracker_redundancy.update_state(ops.stack(cons_red))
#         else:
#             self.loss_tracker_redundancy.update_state(ops.stack(total_redundancy))

#         # After loop: compute Spearman rank metric across vies if requested
#         order_rank_val = None
#         source_dim_means = cons_per_view_dim_means if cons_per_view_dim_means is not None else per_view_dim_means
#         if source_dim_means is not None and len(source_dim_means) > 0:
#             try:
#                 per_view_stack = ops.stack(source_dim_means, axis=1)  # (ndims, n_views)
#                 mean_over_vies = ops.mean(per_view_stack, axis=1)       # (ndims,)
#                 # Use monotonic Pearson trend as an ordering score (graph-safe)
#                 rho = self._order_monotonic_pearson(mean_over_vies)
#                 self.order_rank_tracker.update_state(rho)
#                 order_rank_val = rho
#             except Exception:
#                 pass

#         # Cache the latest batch inputs/outputs for callback use on training end.
#         self._last_batch_inputs = inputs
#         self._last_batch_y = y
#         self._last_batch_scale_factor = scale_fact

#         return {
#             "total_loss": self.loss_tracker_total.result(),
#             "cross_metric": self.corr_tracker.result(),
#             "mse_loss": self.loss_tracker_mse.result(),
#             "redundancy": self.loss_tracker_redundancy.result(),
#             "order_rank": self.order_rank_tracker.result(),
#         }


#     # Removed FinalOrderWithDataCallback (no longer needed)

#     # Removed per-batch _OrderEveryXCallback; only end-of-fit ordering remains


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
#         elif isinstance(optimizer,keras.optimizers.Optimizer): ## This case is important when running a siamese network on one data-view
#             for vie in range(len(self.model_list)):
#                 self.model_list[0].compile(optimizer)
#         else:
#             print('Error: optimizer must either be of the keras.optimizer class, or a list of objects of this class')
        

#     def fit(self, *args, **kwargs):
#         """Attach internal scheduling callback for order_variates, then delegate to Keras.

#         If order=True and orthogonalization='zca', we try to mirror post-hoc ordering by
#         running a final rotation using the same data passed to fit (unless an explicit
#         order_inputs is provided). If we can't derive inputs (e.g., Dataset), we fall
#         back to accumulating over training batches.
#         """
#         callbacks = list(kwargs.get('callbacks', []) or [])

#         # Attach moving-consensus-based periodic order
#         if self.orthogonalization == 'zca':
#             callbacks.append(self._OrderCallback(self))

#         if callbacks:
#             kwargs['callbacks'] = callbacks
#         return super().fit(*args, **kwargs)


#     def test_step(self, inputs):
#         """ This step is called by model.evaluate() on a batch-wise level. This function
#         returns loss metrics for the test data.
#         """

#         inputs = inputs[0]
#         y = self(inputs, training=False)

#         # Shared normalization for corr computation across views
#         eps = ops.convert_to_tensor(self.epsilon, dtype=ops.dtype(y))
#         y_true_c = y - ops.mean(y, axis=0)   # (batch, ndims, n_views)
#         denom_true = ops.sqrt(ops.sum(ops.square(y_true_c), axis=0) + eps)  # (ndims, n_views)
#         y_true_n = y_true_c / denom_true

#         total_loss = [None] * len(self.model_list)
#         total_CC = [None] * len(self.model_list)
#         total_mse = [None] * len(self.model_list)
#         total_redundancy = [None] * len(self.model_list)
#         per_view_dim_means = []

#         inputs_nested = self.organize_inputs_by_model(inputs)

#         for vie in range(len(self.model_list)):
#             y_pred = self.model_list[vie](inputs_nested[vie], training=False)

#             mse_loss = self.mse_loss(y, y_pred, vie)
            
#             internal_losses = self.model_list[vie].losses
#             if internal_losses:
#                 internal_loss = ops.sum(
#                     ops.stack(
#                         [ops.convert_to_tensor(loss, dtype=ops.dtype(mse_loss)) for loss in internal_losses],
#                         axis=0,
#                     ),
#                     axis=0,
#                 )
#             else:
#                 internal_loss = ops.zeros_like(mse_loss)

#             loss = mse_loss + internal_loss

#             corr = self.corr_metric(y, y_pred, vie)

#             total_loss[vie] = ops.sum(loss)
#             total_CC[vie]   = corr
#             total_mse[vie]  = mse_loss
#             total_redundancy[vie] = self.calculate_redundancy(y[:,:,vie])

#             # Build per-dimension mean correlations across connected views (for ranking metric)
#             try:
#                 y_pred_c = y_pred - ops.mean(y_pred, axis=0)
#                 denom_pred = ops.sqrt(ops.sum(ops.square(y_pred_c), axis=0) + eps)
#                 y_pred_n = y_pred_c / denom_pred
#                 y_pred_n = ops.expand_dims(y_pred_n, axis=2)  # (batch, ndims, 1)
#                 corr_mat = ops.sum(y_true_n * y_pred_n, axis=0)  # (ndims, n_views)
#                 mask = ops.cast(self.Path[vie, :], ops.dtype(corr_mat))
#                 corr_masked = corr_mat * ops.expand_dims(mask, axis=0)
#                 n_conn = ops.sum(mask)
#                 n_conn_safe = ops.maximum(n_conn, ops.convert_to_tensor(1.0, dtype=ops.dtype(n_conn)))
#                 per_dim_mean = ops.sum(corr_masked, axis=1) / n_conn_safe  # (ndims,)
#                 per_view_dim_means.append(per_dim_mean)
#             except Exception:
#                 pass

#         self.loss_tracker_total.update_state(ops.stack(total_loss))
#         self.corr_tracker.update_state(ops.stack(total_CC))
#         self.loss_tracker_mse.update_state(ops.stack(total_mse))
#         self.loss_tracker_redundancy.update_state(ops.stack(total_redundancy))

#         # Compute order_rank during evaluation
#         if len(per_view_dim_means) > 0:
#             try:
#                 per_view_stack = ops.stack(per_view_dim_means, axis=1)  # (ndims, n_views)
#                 mean_over_vies = ops.mean(per_view_stack, axis=1)       # (ndims,)
#                 rho_eval = self._order_monotonic_pearson(mean_over_vies)
#                 self.order_rank_tracker.update_state(rho_eval)
#             except Exception:
#                 pass
#         return {
#             "total_loss": self.loss_tracker_total.result(),
#             "cross_metric": self.corr_tracker.result(),
#             "mse_loss": self.loss_tracker_mse.result(),
#             "redundancy": self.loss_tracker_redundancy.result(),
#             "order_rank": self.order_rank_tracker.result(),
#         }

#     class _OrderCallback(keras.callbacks.Callback):
#         """Rotate projection weights every N batches using moving_consensus.

#         - Triggers on each train batch end; applies SVD-based rotation every
#           `model_ref.order_every` batches.
#         - After each rotation, re-initializes the model's moving_consensus to identity.
#         """
#         def __init__(self, model_ref):
#             super().__init__()
#             self._m = model_ref
#             self._batch_count = 0

#         def on_train_batch_end(self, batch, logs=None):
#             m = self._m
#             if not (getattr(m, 'order', False) and getattr(m, 'orthogonalization', '') == 'zca'):
#                 return
#             self._batch_count += 1
#             interval = getattr(m, 'order_every', 100)
#             # If user requested only end-of-training rotation, skip periodic
#             if isinstance(interval, str) and interval.lower() == 'end':
#                 return
#             if interval is None or (isinstance(interval, (int, float)) and interval <= 0):
#                 interval = 100
#             if (self._batch_count % int(interval)) != 0:
#                 return
#             try:
#                 V = m._compute_rotation_from_moving()
#                 m._apply_rotation_to_models(V)
#                 m._reset_moving_consensus()
#             except Exception:
#                 # Do not interrupt training if rotation fails
#                 return

#         def on_train_end(self, logs=None):
#             """Always perform a final rotation at the end of training.

#             Ensures we rotate even if the last batch index is not aligned with
#             the order_every interval. If moving_consensus was reset on the last
#             batch, this is effectively a no-op (rotation by identity).
#             """
#             m = self._m
#             if not (getattr(m, 'order', False) and getattr(m, 'orthogonalization', '') == 'zca'):
#                 return
#             try:
#                 V = m._compute_rotation_from_moving()
#                 m._apply_rotation_to_models(V)
#                 m._reset_moving_consensus()
#             except Exception:
#                 return


#     @property
#     def metrics(self):
#         """We list our `Metric` objects here so that `reset_states()` can be
#         called automatically at the start of each epoch
#         or at the start of `evaluate()`."""

#         return [self.loss_tracker_total, self.corr_tracker, self.loss_tracker_mse, self.order_rank_tracker]


#     def mse_loss(self, y_true, y_pred, vie):
    
#         """
#         Mean squared error between y_pred (view vie) and the connected views in y_true.

#         """
#         # y_true: (batch, ndims, n_views)
#         # y_pred: (batch, ndims)

#         y_pred_exp = ops.expand_dims(y_pred, axis=2)  # (batch, ndims, 1)

#         # Per-view squared error averaged over batch: (ndims, n_views)
#         se_mean = ops.mean(ops.square(y_true - y_pred_exp), axis=0)

#         # Mask to include only connected views
#         mask = ops.cast(self.Path[vie, :], ops.dtype(se_mean))     # (n_views,)
#         se_mean_masked = se_mean * ops.expand_dims(mask, axis=0)   # (ndims, n_views)

#         mse_loss = ops.sum(se_mean_masked) / 2.0
        
#         return mse_loss


#     def corr_metric(self, y_true, y_pred, vie):
        
#         """
#         Mean correlation between y_pred (view vie) and connected views in y_true.
#         """

#         eps = ops.convert_to_tensor(self.epsilon, dtype=ops.dtype(y_true))

#         # Center over batch
#         y_true_c = y_true - ops.mean(y_true, axis=0)   # (ndims, n_views) over batch
#         y_pred_c = y_pred - ops.mean(y_pred, axis=0)   # (ndims) over batch

#         denom_true = ops.sqrt(ops.sum(ops.square(y_true_c), axis=0) + eps)   # (ndims, n_views)
#         denom_pred = ops.sqrt(ops.sum(ops.square(y_pred_c), axis=0) + eps)   # (ndims,)

#         y_true_n = y_true_c / denom_true               # (batch, ndims, n_views)
#         y_pred_n = y_pred_c / denom_pred               # (batch, ndims)

#         y_pred_n = ops.expand_dims(y_pred_n, axis=2)   # (batch, ndims, 1)
#         corr_mat = ops.sum(y_true_n * y_pred_n, axis=0)  # (ndims, n_views)

#         # Mask only connected views
#         mask = ops.cast(self.Path[vie, :], ops.dtype(corr_mat))    # (n_views,)
#         corr_masked = corr_mat * ops.expand_dims(mask, axis=0)     # (ndims, n_views)

#         # Average over dims and number of connected views
#         n_conn = ops.sum(mask)
#         n_conn_safe = ops.maximum(n_conn, ops.convert_to_tensor(1.0, dtype=ops.dtype(n_conn)))
#         corr_mean = ops.sum(corr_masked) / (n_conn_safe * float(self.ndims))

#         return corr_mean
    
#      # This function avoids problems with passing symbolic 
#      # tensors to ops.shape in tensorflow

#     def _shape_fn(self,X):
#         backend = keras.backend.backend()
#         if backend == "tensorflow":
#             shape = tf.shape(X)  # handles unknown ranks
#         else:
#             shape = ops.shape(X)
#         return shape
    

#     def calculate_redundancy(self, Y, epsilon=1e-8):
#         """
#         Args:
#             X: Tensor / KerasTensor, shape (N, D). Each column is a variable.
#             epsilon: Small constant for numerical stability.

#         Returns:
#             Scalar tensor: mean(|corr(i, j)|) over all i != j.
#         """
#         Y = ops.convert_to_tensor(Y)
#         Y = ops.cast(Y, "float32")

#         # Center columns
#         col_mean = ops.mean(Y, axis=0, keepdims=True)
#         Yc = Y - col_mean

#         backend = keras.backend.backend()

#         # Sample-size for covariance
#         n = self._shape_fn(Yc)[0]
#         n_f = ops.cast(n, Y.dtype)
#         denom_n = ops.maximum(n_f - 1.0, 1.0)  # guard when N == 1

#         # Covariance between columns: (D x D)
#         cov = ops.matmul(ops.transpose(Yc), Yc) / denom_n

#         # Column std devs (D,)
#         var = ops.sum(Yc * Yc, axis=0) / denom_n
#         std = ops.sqrt(ops.maximum(var, epsilon))

#         # Correlation matrix: cov / (std_i * std_j)
#         std_col = ops.reshape(std, (-1, 1))              # (D,1)
#         denom = std_col * ops.transpose(std_col)         # (D,D)
#         corr = cov / ops.maximum(denom, epsilon)         # (D,D)

#         # Mean absolute correlation over off-diagonal entries
#         corr_abs = ops.abs(corr)
#         D = self._shape_fn(corr_abs)[0]
#         mask = ops.ones_like(corr_abs) - ops.cast(ops.eye(D), corr_abs.dtype)  # zero diagonal
#         total = ops.sum(corr_abs * mask)

#         D_f = ops.cast(D, corr_abs.dtype)
#         num_pairs = ops.maximum(D_f * (D_f - 1.0), 1.0)  # count of off-diagonal entries

#         return total / num_pairs



#     from keras import ops

#     def calculate_corrmat(self, DLVs):
#         """
#         Compute Pearson correlation matrices for a 3D tensor using keras.ops.
#         DLVs: (n_samples, dimensions, DLVs)
#         Returns: list of (DLVs x DLVs) per dimension.
#         """
#         if len(DLVs.shape) != 3:
#             raise ValueError("Input must be a 3D tensor")

#         # ✅ Ensure we’re working with a backend tensor, even if DLVs was numpy
#         DLVs = ops.convert_to_tensor(DLVs)

#         correlation_matrices = []
#         n_samples = ops.cast(self._shape_fn(DLVs)[0], DLVs.dtype)
#         eps = ops.convert_to_tensor(1e-7, dtype=DLVs.dtype)

#         # Use shape function to be backend-friendly
#         n_dims = int(self._shape_fn(DLVs)[1])

#         for dim in range(n_dims):
#             dim_DLVs = DLVs[:, dim, :]  # (n_samples, n_feats)
#             mean_centered = dim_DLVs - ops.mean(dim_DLVs, axis=0)
#             std_dev = ops.std(dim_DLVs, axis=0) + eps
#             normalized = mean_centered / std_dev
#             correlation_matrix = ops.matmul(
#                 ops.transpose(normalized),
#                 normalized
#             ) / n_samples
#             correlation_matrices.append(correlation_matrix)

#         return correlation_matrices

#     from keras import ops



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
#             "orthogonalization": self.orthogonalization,
#             "sparse_l1_list": self.sparse_l1_list,
#             "order": self.order,
#             "order_every": getattr(self, 'order_every', 100),
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
#         config['Path'] = ops.convert_to_tensor(config["Path"], dtype=ops.floatx())
        
#         # Deserialize each model in the model list using a list comprehension
#         config['model_list'] = [keras.utils.deserialize_keras_object(model_config) for model_config in config['model_list']]
#         config['run_from_config'] = True
        
#         # If regularization is present in the config, deserialize it
#         if 'regularizer_list' in config:
#             config['regularizer_list'] = [keras.utils.deserialize_keras_object(regularizer_config) for regularizer_config in config['regularizer_list']]

#         # Ensure 'order' defaults to False if not provided
#         if 'order' not in config:
#             config['order'] = False
#         # Handle order_every (interval in batches). Allow 'end' sentinel.
#         oe = config.get('order_every', 100)
#         if isinstance(oe, str) and oe.lower() == 'end':
#             config['order_every'] = 'end'
#         else:
#             try:
#                 config['order_every'] = int(oe)
#             except Exception:
#                 config['order_every'] = 100
        
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

    
    def __init__(self, Path, model_list, regularizer_list, tot_num, ndims, orthogonalization='Moore-Penrose', momentum=0.95, epsilon=1e-4, train_DLV=True, run_from_config=False, is_siamese=False, diag_offset=1e-3, **kwargs):
        
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

        if not run_from_config:
        # Add factor layer to each model in the list
            if self.is_siamese == True:
                new_model = self.add_DLVPM_layer(model_list[0], regularizer_list[0])
                self.model_list = [new_model] * len(model_list)   # duplicates the *reference*
            else:
                self.model_list = [self.add_DLVPM_layer(model, regularizer) for model, regularizer in zip(model_list, regularizer_list)]
        else:
            self.model_list = model_list

        self.loss_tracker_total = keras.metrics.Mean(name="total_loss")
        self.corr_tracker = keras.metrics.Mean(name="cross_metric")
        self.loss_tracker_mse = keras.metrics.Mean(name="mse_loss")
        self.loss_tracker_redundancy = keras.metrics.Mean(name="redundancy")


    
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
                model.add(ZCALayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon, diag_offset = self.diag_offset))
            else:
                print('Orthogonalization mode not recognised, must be "Moore-Penrose" or "zca"')
        elif isinstance(model, keras.Model):
            if self.orthogonalization == 'Moore-Penrose':
                x = FactorLayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon)(model.output)
                model = keras.Model(inputs=model.input, outputs=x)
            elif self.orthogonalization == 'zca':
                x = ZCALayer(kernel_regularizer=regularizer, tot_num=self.tot_num, ndims=self.ndims, momentum=self.momentum, epsilon=self.epsilon, diag_offset = self.diag_offset)(model.output)
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
       
        return loss, mse_loss

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

    def _weight_normaliser(self, y):
        """This is an internal function designed to normalise weights and DLVs
        after each batch"""
        
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



        total_loss = [None] * len(self.model_list)
        total_CC   = [None] * len(self.model_list)
        total_mse  = [None] * len(self.model_list)
        total_redundancy = [None] * len(self.model_list)


        # Unpack (tf.data-like packs inputs in a tuple/list)
        inputs = inputs[0]
        be = keras.backend.backend()  # 'tensorflow' | 'torch' | 'jax' (we handle tf/torch)

        if be == "tensorflow":
            y_raw = self(inputs, training=self.train_DLV)  # pre-orthogonalization
            y_ortho, scale_fact = self._weight_normaliser(y_raw)
        elif be == "torch":
            with torch.no_grad():
                y_raw = self(inputs, training=self.train_DLV)  # pre-orthogonalization
                y_ortho, scale_fact = self._weight_normaliser(y_raw)
        else:
            raise NotImplementedError(f"Backend '{be}' not supported in custom train_step.")

        inputs_nested = self.organize_inputs_by_model(inputs)

        for vie in range(len(self.model_list)):
            if be == "tensorflow":
                loss, mse_loss = self._step_tf(vie, inputs_nested[vie], y_ortho, scale_fact)
            elif be == "torch":
                loss, mse_loss = self._step_torch(vie, inputs_nested[vie], y_ortho, scale_fact)

            total_loss[vie] = ops.sum(loss)
            total_CC[vie] = self.corr_metric(y_raw, y_raw[:,:,vie], vie)
            total_redundancy[vie] = self.calculate_redundancy(y_raw[:,:,vie])
            total_mse[vie]  = mse_loss

        self.loss_tracker_total.update_state(ops.stack(total_loss))
        self.corr_tracker.update_state(ops.stack(total_CC))
        self.loss_tracker_mse.update_state(ops.stack(total_mse))
        self.loss_tracker_redundancy.update_state(ops.stack(total_redundancy))

        return {
            "total_loss": self.loss_tracker_total.result(),
            "cross_metric": self.corr_tracker.result(),
            "mse_loss": self.loss_tracker_mse.result(),
            "redundancy": self.loss_tracker_redundancy.result(),
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
            "redundancy": self.loss_tracker_redundancy.result(),
        }


    @property
    def metrics(self):
        """We list our `Metric` objects here so that `reset_states()` can be
        called automatically at the start of each epoch
        or at the start of `evaluate()`."""

        return [
            self.loss_tracker_total,
            self.corr_tracker,
            self.loss_tracker_mse,
            self.loss_tracker_redundancy,
        ]


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
    
