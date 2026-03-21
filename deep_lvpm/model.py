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

import inspect
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

    
    def __init__(
        self,
        Path,
        model_list,
        regularizer_list,
        tot_num,
        ndims,
        orthogonalization='Moore-Penrose',
        momentum=0.95,
        epsilon=1e-4,
        train_DLV=True,
        run_from_config=False,
        is_siamese=False,
        diag_offset=1e-3,
        sparse_l1_list=0.0,
        orthog_weight=0.0,
        order=False,
        order_loss_weight=1.0,
        **kwargs,
    ):
        
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
        
        self._path_array = self._path_to_numpy(Path)
        self.Path = ops.convert_to_tensor(self._path_array, dtype=keras.backend.floatx())
        self.tot_num = tot_num
        self.ndims = ndims
        self.momentum = momentum
        self.epsilon = epsilon
        self.orthogonalization=orthogonalization
        self.regularizer_list = regularizer_list
        self.train_DLV = train_DLV
        self.is_siamese = is_siamese
        self.diag_offset = diag_offset
        n_views = len(model_list)
        self.sparse_l1_list = self._normalize_sparse_l1_list(sparse_l1_list, n_views)
        self.orthog_weight = float(orthog_weight)
        if self.orthog_weight != 0.0 and self.orthogonalization != 'zca':
            raise ValueError("'orthog_weight' is only available when orthogonalization='zca'.")
        self.order = bool(order)
        if self.order and self.orthogonalization != 'zca':
            raise ValueError("'order' is only available when orthogonalization='zca'.")
        self.order_loss_weight = float(order_loss_weight)
        self.order_optimizer = None

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
        self.order_loss_tracker = keras.metrics.Mean(name="order_loss")
        n_views = len(self.model_list)
        float_dtype = keras.backend.floatx()
        self.order_rotation_raw = self.add_weight(
            name="order_rotation_raw",
            shape=(self.ndims, self.ndims),
            initializer=keras.initializers.Identity(),
            dtype=float_dtype,
            trainable=self.order,
        )
        self._order_moving_omega = self.add_weight(
            name="order_moving_omega",
            shape=(self.ndims, self.ndims),
            initializer="zeros",
            dtype=float_dtype,
            trainable=False,
        )
        self._order_basis = self.add_weight(
            name="order_basis",
            shape=(self.ndims, self.ndims),
            initializer=keras.initializers.Identity(),
            dtype=float_dtype,
            trainable=False,
        )
        self._order_dim_scaling = self.add_weight(
            name="order_dim_scaling",
            shape=(self.ndims,),
            initializer="ones",
            dtype=float_dtype,
            trainable=False,
        )
        
        self._order_cross_sum = self.add_weight(
            name="structural_order_cross_sum",
            shape=(n_views, n_views, self.ndims, self.ndims),
            initializer="zeros",
            dtype=float_dtype,
            trainable=False,
        )
        self._order_sum_left = self.add_weight(
            name="structural_order_sum_left",
            shape=(n_views, n_views, self.ndims),
            initializer="zeros",
            dtype=float_dtype,
            trainable=False,
        )
        self._order_sum_right = self.add_weight(
            name="structural_order_sum_right",
            shape=(n_views, n_views, self.ndims),
            initializer="zeros",
            dtype=float_dtype,
            trainable=False,
        )
        self._order_pair_count = self.add_weight(
            name="structural_order_pair_count",
            shape=(n_views, n_views),
            initializer="zeros",
            dtype=float_dtype,
            trainable=False,
        )


    
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
                model.add(
                    FactorLayer(
                        kernel_regularizer=regularizer,
                        tot_num=self.tot_num,
                        ndims=self.ndims,
                        momentum=self.momentum,
                        epsilon=self.epsilon,
                        sparse_l1=sparse_l1,
                    )
                )
            elif self.orthogonalization == 'zca':
                model.add(
                    ZCALayer(
                        kernel_regularizer=regularizer,
                        tot_num=self.tot_num,
                        ndims=self.ndims,
                        momentum=self.momentum,
                        epsilon=self.epsilon,
                        diag_offset=self.diag_offset,
                        sparse_l1=sparse_l1,
                        orthog_weight=self.orthog_weight,
                    )
                )
            else:
                print('Orthogonalization mode not recognised, must be "Moore-Penrose" or "zca"')
        elif isinstance(model, keras.Model):
            if self.orthogonalization == 'Moore-Penrose':
                x = FactorLayer(
                    kernel_regularizer=regularizer,
                    tot_num=self.tot_num,
                    ndims=self.ndims,
                    momentum=self.momentum,
                    epsilon=self.epsilon,
                    sparse_l1=sparse_l1,
                )(model.output)
                model = keras.Model(inputs=model.input, outputs=x)
            elif self.orthogonalization == 'zca':
                x = ZCALayer(
                    kernel_regularizer=regularizer,
                    tot_num=self.tot_num,
                    ndims=self.ndims,
                    momentum=self.momentum,
                    epsilon=self.epsilon,
                    diag_offset=self.diag_offset,
                    sparse_l1=sparse_l1,
                    orthog_weight=self.orthog_weight,
                )(model.output)
                model = keras.Model(inputs=model.input, outputs=x)
            else:
                print('Orthogonalization mode not recognised, must be "Moore-Penrose" or "zca"')
        else:
            raise ValueError("The input model must be either a keras.Sequential or a keras.Model instance.")

        
        return model


    def _path_to_numpy(self, path):
        if isinstance(path, np.ndarray):
            return path.astype(keras.backend.floatx())
        try:
            return np.asarray(keras.ops.convert_to_numpy(path), dtype=keras.backend.floatx())
        except Exception:
            return np.asarray(path, dtype=keras.backend.floatx())


    def _normalize_sparse_l1_list(self, sparse_l1_list, n_views):
        if sparse_l1_list is None:
            values = [0.0] * n_views
        elif isinstance(sparse_l1_list, (list, tuple, np.ndarray)):
            values = [float(x) for x in list(sparse_l1_list)]
            if len(values) != n_views:
                raise ValueError(f"sparse_l1_list must have length {n_views}, got {len(values)}")
        else:
            values = [float(sparse_l1_list)] * n_views

        if self.is_siamese and any(abs(x - values[0]) > 1e-12 for x in values):
            raise ValueError("In siamese mode, all entries of sparse_l1_list must be identical.")

        return values


    def _plain_optimizer_value(self, value):
        if isinstance(value, (bool, int, float, str, type(None))):
            return value
        try:
            array_value = np.asarray(keras.ops.convert_to_numpy(value))
            if array_value.shape == ():
                return array_value.item()
            return array_value
        except Exception:
            return value


    def _clone_optimizer(self, optimizer):
        optimizer_cls = optimizer.__class__
        signature = inspect.signature(optimizer_cls.__init__)
        kwargs = {}

        for name, parameter in signature.parameters.items():
            if name in {"self", "args", "kwargs"}:
                continue
            if not hasattr(optimizer, name):
                continue

            value = self._plain_optimizer_value(getattr(optimizer, name))
            if parameter.kind in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }:
                kwargs[name] = value

        try:
            return optimizer_cls(**kwargs)
        except Exception:
            learning_rate = self._plain_optimizer_value(getattr(optimizer, "learning_rate", 1e-3))
            return optimizer_cls(learning_rate=learning_rate)


    def _reset_structural_order_stats(self):
        self._order_cross_sum.assign(ops.zeros_like(self._order_cross_sum))
        self._order_sum_left.assign(ops.zeros_like(self._order_sum_left))
        self._order_sum_right.assign(ops.zeros_like(self._order_sum_right))
        self._order_pair_count.assign(ops.zeros_like(self._order_pair_count))


    def _has_structural_order_stats(self):
        count_value = float(np.sum(np.asarray(keras.ops.convert_to_numpy(self._order_pair_count))))
        return count_value > 0.0


    def _update_structural_order_stats(self, y, view_present):
        if not self.order:
            return

        dtype = self._order_cross_sum.dtype
        y = ops.cast(y, dtype)
        view_present = ops.cast(view_present, dtype)
        n_views = len(self.model_list)

        cross_rows = []
        sum_left_rows = []
        sum_right_rows = []
        count_rows = []

        zero_cross = ops.zeros((self.ndims, self.ndims), dtype=dtype)
        zero_sum = ops.zeros((self.ndims,), dtype=dtype)
        zero_count = ops.convert_to_tensor(0.0, dtype=dtype)

        for left_vie in range(n_views):
            y_left = y[:, :, left_vie]
            left_mask = view_present[:, left_vie]

            cross_row = []
            sum_left_row = []
            sum_right_row = []
            count_row = []

            for right_vie in range(n_views):
                edge_weight = float(self._path_array[left_vie, right_vie])
                if left_vie == right_vie or edge_weight == 0.0:
                    cross_row.append(zero_cross)
                    sum_left_row.append(zero_sum)
                    sum_right_row.append(zero_sum)
                    count_row.append(zero_count)
                    continue

                pair_mask = left_mask * view_present[:, right_vie]
                pair_mask_exp = ops.expand_dims(pair_mask, axis=1)
                y_left_masked = y_left * pair_mask_exp
                y_right_masked = y[:, :, right_vie] * pair_mask_exp

                cross_row.append(ops.matmul(ops.transpose(y_left_masked), y_right_masked))
                sum_left_row.append(ops.sum(y_left_masked, axis=0))
                sum_right_row.append(ops.sum(y_right_masked, axis=0))
                count_row.append(ops.sum(pair_mask))

            cross_rows.append(ops.stack(cross_row, axis=0))
            sum_left_rows.append(ops.stack(sum_left_row, axis=0))
            sum_right_rows.append(ops.stack(sum_right_row, axis=0))
            count_rows.append(ops.stack(count_row, axis=0))

        self._order_cross_sum.assign(self._order_cross_sum + ops.stack(cross_rows, axis=0))
        self._order_sum_left.assign(self._order_sum_left + ops.stack(sum_left_rows, axis=0))
        self._order_sum_right.assign(self._order_sum_right + ops.stack(sum_right_rows, axis=0))
        self._order_pair_count.assign(self._order_pair_count + ops.stack(count_rows, axis=0))


    def _compute_structural_rotation(self):
        if not self._has_structural_order_stats():
            return ops.eye(self.ndims, dtype=self._order_cross_sum.dtype)

        dtype = self._order_cross_sum.dtype
        omega = ops.zeros((self.ndims, self.ndims), dtype=dtype)
        one = ops.convert_to_tensor(1.0, dtype=dtype)

        for left_vie in range(len(self.model_list)):
            for right_vie in range(len(self.model_list)):
                edge_weight = float(self._path_array[left_vie, right_vie])
                if left_vie == right_vie or edge_weight == 0.0:
                    continue

                count = self._order_pair_count[left_vie, right_vie]
                count_safe = ops.maximum(count, one)
                sum_left = self._order_sum_left[left_vie, right_vie]
                sum_right = self._order_sum_right[left_vie, right_vie]
                mean_outer = ops.matmul(
                    ops.expand_dims(sum_left, axis=1),
                    ops.expand_dims(sum_right, axis=0),
                ) / count_safe
                centered_cross = self._order_cross_sum[left_vie, right_vie] - mean_outer
                pair_cov = centered_cross / count_safe
                valid = ops.cast(count > one, dtype)
                omega = omega + ops.cast(edge_weight, dtype) * pair_cov * valid

        omega = 0.5 * (omega + ops.transpose(omega))
        omega = omega + ops.cast(self.epsilon, dtype) * ops.eye(self.ndims, dtype=dtype)
        _, eigvecs = ops.linalg.eigh(omega)
        return ops.flip(eigvecs, axis=1)


    def _apply_structural_rotation(self, rotation):
        seen = set()

        for mdl in self.model_list:
            model_id = id(mdl)
            if model_id in seen:
                continue
            seen.add(model_id)

            last_layer = mdl.layers[-1]
            if not isinstance(last_layer, ZCALayer):
                continue

            rotation_project = ops.cast(rotation, last_layer.project.dtype)
            last_layer.project.assign(ops.matmul(last_layer.project, rotation_project))

            rotation_cov = ops.cast(rotation, last_layer.moving_conv2.dtype)
            rotated_cov = ops.matmul(
                ops.transpose(rotation_cov),
                ops.matmul(last_layer.moving_conv2, rotation_cov),
            )
            rotated_cov = 0.5 * (rotated_cov + ops.transpose(rotated_cov))
            last_layer.moving_conv2.assign(rotated_cov)


    def _apply_structural_ordering(self):
        if not self.order or not self._has_structural_order_stats():
            return False

        rotation = self._compute_structural_rotation()
        self._apply_structural_rotation(rotation)
        self._reset_structural_order_stats()
        return True


    def _current_order_rotation(self, dtype=None):
        if not self.order:
            return ops.eye(self.ndims, dtype=dtype or keras.backend.floatx())

        rotation_raw = self.order_rotation_raw
        if dtype is not None:
            rotation_raw = ops.cast(rotation_raw, dtype)

        q, r = ops.linalg.qr(rotation_raw, mode="reduced")
        diag = ops.diagonal(r)
        sign = ops.sign(diag)
        sign = ops.where(sign == 0, ops.ones_like(sign), sign)
        q = q * ops.expand_dims(sign, axis=0)
        return q


    def _apply_order_rotation(self, y, rotation):
        if not self.order:
            return y

        rotated = []
        for vie in range(len(self.model_list)):
            rotated.append(ops.matmul(y[:, :, vie], rotation))
        return ops.stack(rotated, axis=2)


    def _batch_structural_matrix(self, y, view_present):
        dtype = ops.dtype(y)
        y = ops.cast(y, dtype)
        view_present = ops.cast(view_present, dtype)
        omega = ops.zeros((self.ndims, self.ndims), dtype=dtype)
        one = ops.convert_to_tensor(1.0, dtype=dtype)

        for left_vie in range(len(self.model_list)):
            y_left = y[:, :, left_vie]
            left_mask = view_present[:, left_vie]

            for right_vie in range(len(self.model_list)):
                edge_weight = float(self._path_array[left_vie, right_vie])
                if left_vie == right_vie or edge_weight == 0.0:
                    continue

                pair_mask = left_mask * view_present[:, right_vie]
                pair_mask_exp = ops.expand_dims(pair_mask, axis=1)
                pair_count = ops.sum(pair_mask)
                pair_count_safe = ops.maximum(pair_count, one)

                y_left_masked = y_left * pair_mask_exp
                y_right_masked = y[:, :, right_vie] * pair_mask_exp
                sum_left = ops.sum(y_left_masked, axis=0)
                sum_right = ops.sum(y_right_masked, axis=0)
                cross = ops.matmul(ops.transpose(y_left_masked), y_right_masked)
                mean_outer = ops.matmul(
                    ops.expand_dims(sum_left, axis=1),
                    ops.expand_dims(sum_right, axis=0),
                ) / pair_count_safe
                centered_cross = cross - mean_outer
                pair_cov = centered_cross / pair_count_safe
                valid = ops.cast(pair_count > one, dtype)
                omega = omega + ops.cast(edge_weight, dtype) * pair_cov * valid

        omega = 0.5 * (omega + ops.transpose(omega))
        return omega


    def _update_order_moving_omega(self, omega_batch):
        if not self.order:
            return

        momentum = ops.cast(self.momentum, dtype=self._order_moving_omega.dtype)
        one = ops.cast(1.0, dtype=self._order_moving_omega.dtype)
        omega_cast = ops.cast(omega_batch, dtype=self._order_moving_omega.dtype)
        self._order_moving_omega.assign(
            momentum * self._order_moving_omega + (one - momentum) * omega_cast
        )


    def _refresh_order_basis_and_scaling(self, dtype=None):
        if not self.order:
            return

        work_dtype = dtype or self._order_moving_omega.dtype
        omega = ops.cast(self._order_moving_omega, work_dtype)
        omega = 0.5 * (omega + ops.transpose(omega))
        omega = omega + ops.cast(self.epsilon, work_dtype) * ops.eye(self.ndims, dtype=work_dtype)

        eigvals, eigvecs = ops.linalg.eigh(omega)
        eigvals = ops.flip(eigvals, axis=0)
        eigvecs = ops.flip(eigvecs, axis=1)
        scaling = ops.cast(self.ndims, work_dtype) * ops.softmax(eigvals)
        scaling = ops.sqrt(ops.maximum(scaling, ops.cast(self.epsilon, work_dtype)))

        self._order_basis.assign(ops.cast(eigvecs, self._order_basis.dtype))
        self._order_dim_scaling.assign(ops.cast(scaling, self._order_dim_scaling.dtype))


    def _apply_order_basis_and_scaling(self, values):
        if not self.order:
            return values

        dtype = ops.dtype(values)
        basis = ops.cast(self._order_basis, dtype)
        scaling = ops.cast(self._order_dim_scaling, dtype)

        if len(values.shape) == 2:
            transformed = ops.matmul(values, basis)
            return transformed * ops.expand_dims(scaling, axis=0)

        if len(values.shape) == 3:
            transformed_views = []
            scale_expanded = ops.expand_dims(scaling, axis=0)
            for vie in range(len(self.model_list)):
                transformed = ops.matmul(values[:, :, vie], basis)
                transformed_views.append(transformed * scale_expanded)
            return ops.stack(transformed_views, axis=2)

        raise ValueError("Ordering expects a 2D or 3D tensor.")


    def _encoder_order_regularizer(self, y_pred):
        if not self.order:
            return self._zero_scalar(dtype=ops.dtype(y_pred))

        dtype = ops.dtype(y_pred)
        ordered_target = ops.stop_gradient(self._apply_order_basis_and_scaling(y_pred))
        sq_error = ops.square(y_pred - ordered_target)
        order_loss = ops.mean(sq_error)
        return ops.cast(self.order_loss_weight, dtype) * order_loss


    def _order_loss_from_rotation(self, rotation, omega):
        dtype = ops.dtype(rotation)
        omega = ops.cast(omega, dtype)
        rotated_omega = ops.matmul(ops.transpose(rotation), ops.matmul(omega, rotation))
        rotated_omega = 0.5 * (rotated_omega + ops.transpose(rotated_omega))
        diag_weights = ops.convert_to_tensor(
            np.arange(self.ndims, 0, -1, dtype=np.float32),
            dtype=dtype,
        )
        diag_values = ops.diagonal(rotated_omega)
        weighted_diag = ops.sum(diag_weights * diag_values) / ops.sum(diag_weights)
        return -ops.cast(self.order_loss_weight, dtype) * weighted_diag


    def _step_order_tf(self, y_ortho, view_present):
        omega_batch = self._batch_structural_matrix(y_ortho, view_present)
        self._update_order_moving_omega(omega_batch)
        omega_target = ops.stop_gradient(ops.cast(self._order_moving_omega, ops.dtype(y_ortho)))

        with tf.GradientTape() as tape:
            rotation = self._current_order_rotation(dtype=ops.dtype(y_ortho))
            order_loss = self._order_loss_from_rotation(rotation, omega_target)
            scaled_order_loss = self.order_optimizer.scale_loss(order_loss)

        grads = tape.gradient(scaled_order_loss, [self.order_rotation_raw])
        if grads[0] is not None:
            self.order_optimizer.apply_gradients([(grads[0], self.order_rotation_raw)])

        rotation = self._current_order_rotation(dtype=ops.dtype(y_ortho))
        return rotation, order_loss


    def _step_order_torch(self, y_ortho, view_present):
        omega_batch = self._batch_structural_matrix(y_ortho, view_present)
        self._update_order_moving_omega(omega_batch)
        omega_target = ops.stop_gradient(ops.cast(self._order_moving_omega, ops.dtype(y_ortho)))

        rotation = self._current_order_rotation(dtype=ops.dtype(y_ortho))
        order_loss = self._order_loss_from_rotation(rotation, omega_target)
        scaled_order_loss = self.order_optimizer.scale_loss(order_loss)
        self.zero_grad()
        scaled_order_loss.backward()

        rotation_var = getattr(self.order_rotation_raw, "value", self.order_rotation_raw)
        grad = rotation_var.grad
        if grad is None:
            grad = ops.zeros_like(self.order_rotation_raw)
        self.order_optimizer.apply([grad], [self.order_rotation_raw])
        rotation_var.grad = None

        rotation = self._current_order_rotation(dtype=ops.dtype(y_ortho))
        return rotation, order_loss


    def _order_rotation_step(self, y_ortho, view_present):
        if not self.order:
            return None, self._zero_scalar(dtype=ops.dtype(y_ortho))

        if self.order_optimizer is None:
            raise RuntimeError("order=True requires the StructuralModel to be compiled before training.")

        if keras.backend.backend() == "tensorflow":
            return self._step_order_tf(y_ortho, view_present)
        return self._step_order_torch(y_ortho, view_present)
    

    def call(self, inputs, training=False):
        """Run each view encoder on observed rows only and scatter latents back."""

        inputs_nested = self.organize_inputs_by_model(inputs)
        out, _ = self._forward_views_with_missing(inputs_nested, training=training)
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


    def _reference_input(self, inputs_v):
        """Return a representative tensor for a view, including multi-input views."""
        if isinstance(inputs_v, (list, tuple)):
            return inputs_v[0]
        return inputs_v


    def _is_float_tensor(self, tensor):
        """True when the tensor dtype can legitimately contain NaNs."""
        dtype_name = keras.backend.standardize_dtype(ops.dtype(tensor))
        return dtype_name.startswith("float") or dtype_name.startswith("bfloat")


    def _bool_fill(self, reference_tensor, value):
        """Create a backend-native boolean vector with one entry per batch row."""
        batch_size = self._shape_fn(reference_tensor)[0]
        if keras.backend.backend() == "tensorflow":
            return tf.fill([batch_size], value)
        return torch.full((reference_tensor.shape[0],), value, dtype=torch.bool, device=reference_tensor.device)


    def _zero_scalar(self, dtype=None):
        """Return a scalar zero tensor in the requested dtype."""
        return ops.convert_to_tensor(0.0, dtype=dtype or keras.backend.floatx())


    def _zero_latents(self, batch_size, dtype=None, reference_tensor=None):
        """Allocate an all-zero latent batch for scatter-style reconstruction."""
        dtype = dtype or keras.backend.floatx()
        if keras.backend.backend() == "tensorflow":
            return tf.zeros((batch_size, self.ndims), dtype=dtype)

        if reference_tensor is not None:
            return torch.zeros(
                (reference_tensor.shape[0], self.ndims),
                dtype=reference_tensor.dtype if isinstance(dtype, str) else dtype,
                device=reference_tensor.device,
            )

        torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
        return torch.zeros((batch_size, self.ndims), dtype=torch_dtype)


    def _row_indices(self, row_mask):
        """Return row indices where a boolean mask is true."""
        if keras.backend.backend() == "tensorflow":
            return tf.reshape(tf.where(row_mask), (-1,))
        return torch.nonzero(row_mask, as_tuple=False).reshape(-1)


    def _gather_rows_by_index(self, inputs_v, row_indices):
        """Gather observed rows for a single- or multi-input view."""
        if isinstance(inputs_v, (list, tuple)):
            return [self._gather_rows_by_index(tensor, row_indices) for tensor in inputs_v]

        if keras.backend.backend() == "tensorflow":
            return tf.gather(inputs_v, row_indices, axis=0)
        if row_indices.device != inputs_v.device:
            row_indices = row_indices.to(inputs_v.device)
        return inputs_v.index_select(0, row_indices)


    def _gather_rows(self, inputs_v, row_mask):
        """Gather rows using a boolean mask."""
        return self._gather_rows_by_index(inputs_v, self._row_indices(row_mask))


    def _scatter_rows_by_index(self, values, row_indices, batch_size, dtype=None, reference_tensor=None):
        """Scatter observed latent rows back into a zero-filled batch tensor."""
        if keras.backend.backend() == "tensorflow":
            zeros = self._zero_latents(batch_size, dtype=dtype)
            return tf.tensor_scatter_nd_update(zeros, tf.expand_dims(row_indices, axis=1), values)

        return self._scatter_rows_torch(values, row_indices, dtype=dtype, reference_tensor=reference_tensor)


    def _scatter_rows_torch(self, values, row_indices, dtype=None, reference_tensor=None):
        """Torch-only scatter helper used by the backend-agnostic wrapper."""
        target_device = reference_tensor.device if reference_tensor is not None else values.device
        if values.device != target_device:
            values = values.to(target_device)
        if row_indices.device != target_device:
            row_indices = row_indices.to(target_device)

        if reference_tensor is not None:
            zeros = torch.zeros(
                (reference_tensor.shape[0], self.ndims),
                dtype=values.dtype if isinstance(dtype, str) else values.dtype,
                device=target_device,
            )
        else:
            zeros = torch.zeros(
                (int(row_indices.max().item()) + 1 if row_indices.numel() else 0, self.ndims),
                dtype=values.dtype if isinstance(dtype, str) else (dtype or values.dtype),
                device=target_device,
            )
        zeros[row_indices] = values
        return zeros


    def _tensor_row_missing_mask(self, tensor, view_index):
        """Detect rows that are entirely NaN and reject partial-NaN rows."""
        reference_tensor = self._reference_input(tensor)
        if not self._is_float_tensor(tensor):
            return self._bool_fill(reference_tensor, False)

        if keras.backend.backend() == "tensorflow":
            nan_mask = tf.math.is_nan(tensor)
            flat_nan_mask = tf.reshape(nan_mask, (tf.shape(tensor)[0], -1))
            row_any_nan = tf.reduce_any(flat_nan_mask, axis=1)
            row_all_nan = tf.reduce_all(flat_nan_mask, axis=1)
            assertion = tf.debugging.assert_equal(
                row_any_nan,
                row_all_nan,
                message=(
                    f"View {view_index} contains partially missing rows. "
                    "Only all-NaN rows are supported for missing-view handling."
                ),
            )
            with tf.control_dependencies([assertion]):
                return tf.identity(row_all_nan)

        nan_mask = torch.isnan(tensor)
        flat_nan_mask = nan_mask.reshape(nan_mask.shape[0], -1)
        row_any_nan = torch.any(flat_nan_mask, dim=1)
        row_all_nan = torch.all(flat_nan_mask, dim=1)
        if bool(torch.any(torch.logical_xor(row_any_nan, row_all_nan)).item()):
            raise ValueError(
                f"View {view_index} contains partially missing rows. "
                "Only all-NaN rows are supported for missing-view handling."
            )
        return row_all_nan


    def _view_row_present_mask(self, inputs_v, view_index):
        """Return the per-row observed mask for a view, including multi-input views."""
        tensors = inputs_v if isinstance(inputs_v, (list, tuple)) else [inputs_v]
        reference_tensor = self._reference_input(inputs_v)
        missing_mask = None

        for tensor in tensors:
            if not self._is_float_tensor(tensor):
                continue

            current_missing = self._tensor_row_missing_mask(tensor, view_index)
            if missing_mask is None:
                missing_mask = current_missing
                continue

            if keras.backend.backend() == "tensorflow":
                assertion = tf.debugging.assert_equal(
                    missing_mask,
                    current_missing,
                    message=(
                        f"View {view_index} has inconsistent missing-row masks across inputs. "
                        "All tensors for a multi-input view must mark the same rows as missing."
                    ),
                )
                with tf.control_dependencies([assertion]):
                    missing_mask = tf.identity(missing_mask)
            elif not torch.equal(missing_mask, current_missing):
                raise ValueError(
                    f"View {view_index} has inconsistent missing-row masks across inputs. "
                    "All tensors for a multi-input view must mark the same rows as missing."
                )

        if missing_mask is None:
            return self._bool_fill(reference_tensor, True)

        if keras.backend.backend() == "tensorflow":
            return tf.logical_not(missing_mask)
        return torch.logical_not(missing_mask)


    def _encode_view_on_present_rows(self, vie, inputs_v, row_mask, training):
        """Run one encoder on observed rows only, then scatter back to batch size."""
        reference_tensor = self._reference_input(inputs_v)
        batch_size = self._shape_fn(reference_tensor)[0]
        row_indices = self._row_indices(row_mask)

        if keras.backend.backend() == "tensorflow":
            dtype = self.model_list[vie].compute_dtype or keras.backend.floatx()

            def encode_present_rows():
                observed_inputs = self._gather_rows_by_index(inputs_v, row_indices)
                y_obs = self.model_list[vie](observed_inputs, training=training)
                return self._scatter_rows_by_index(
                    y_obs,
                    row_indices,
                    batch_size,
                    dtype=ops.dtype(y_obs),
                )

            def encode_missing_rows():
                return self._zero_latents(batch_size, dtype=dtype)

            return tf.cond(tf.shape(row_indices)[0] > 0, encode_present_rows, encode_missing_rows)

        if row_indices.numel() == 0:
            return self._zero_latents(batch_size, reference_tensor=reference_tensor)

        observed_inputs = self._gather_rows_by_index(inputs_v, row_indices)
        y_obs = self.model_list[vie](observed_inputs, training=training)
        return self._scatter_rows_by_index(
            y_obs,
            row_indices,
            batch_size,
            dtype=y_obs.dtype,
            reference_tensor=reference_tensor,
        )


    def _forward_views_with_missing(self, inputs_nested, training=False):
        """Encode all views with missing-row filtering and return stacked latents plus masks."""
        y_list = []
        view_present_list = []

        for vie in range(len(self.model_list)):
            row_present = self._view_row_present_mask(inputs_nested[vie], vie)
            view_present_list.append(row_present)
            y_list.append(
                self._encode_view_on_present_rows(vie, inputs_nested[vie], row_present, training=training)
            )

        y = ops.stack(y_list, axis=2)
        view_present = ops.stack(view_present_list, axis=1)
        return y, view_present


    def _normalize_pred(self, y_pred, scale_fact):
        eps = getattr(self, "epsilon", 1e-8)
        eps = ops.convert_to_tensor(eps, dtype=ops.dtype(y_pred))
        # y_pred / (sqrt(scale_fact) * ||y_pred||_2 over batch)
        denom = ops.sqrt(scale_fact) * ops.sqrt(ops.sum(ops.square(y_pred), axis=0) + eps)
        return y_pred / denom

    def _step_tf(self, vie, inputs_v, y, view_present, row_mask, scale_fact):
        """TensorFlow training step using only observed rows for the source view."""

        model = self.model_list[vie]
        batch_size = self._shape_fn(y)[0]
        with tf.GradientTape() as tape:
            y_pred_obs = model(inputs_v, training=True)
            y_pred_obs = self._normalize_pred(y_pred_obs, scale_fact)
            y_pred = self._scatter_rows_by_index(
                y_pred_obs,
                self._row_indices(row_mask),
                batch_size,
                dtype=ops.dtype(y_pred_obs),
            )
            mse_loss = self.mse_loss(y, y_pred, vie, view_present)
            order_loss = self._encoder_order_regularizer(y_pred_obs)
            internal_loss = tf.add_n(model.losses) if model.losses else tf.cast(0.0, mse_loss.dtype)
            loss = mse_loss + order_loss + internal_loss

        trainable_vars = model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        grads_and_vars = [(grad, var) for grad, var in zip(grads, trainable_vars) if grad is not None]
        if grads_and_vars:
            model.optimizer.apply_gradients(grads_and_vars)
       
        return loss, mse_loss, order_loss

    def _step_torch(self, vie, inputs_v, y, view_present, row_mask, scale_fact):
        """Torch training step using only observed rows for the source view."""

        model = self.model_list[vie]
        reference_tensor = self._reference_input(inputs_v)

        # Forward pass (PyTorch autograd records ops by default)
        y_pred_obs = model(inputs_v, training=True)
        y_pred_obs = self._normalize_pred(y_pred_obs, scale_fact)
        y_pred = self._scatter_rows_by_index(
            y_pred_obs,
            self._row_indices(row_mask),
            self._shape_fn(y)[0],
            dtype=y_pred_obs.dtype,
            reference_tensor=reference_tensor,
        )
        mse_loss = self.mse_loss(y, y_pred, vie, view_present)
        order_loss = self._encoder_order_regularizer(y_pred_obs)

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
                              
        loss = mse_loss + order_loss + internal_loss

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

        return loss, mse_loss, order_loss


    def _scale_factor_for_rows(self, row_mask, dtype):
        """Compute the batch scaling factor from the number of observed rows."""
        row_count = ops.sum(ops.cast(row_mask, dtype))
        row_count_safe = ops.maximum(row_count, ops.convert_to_tensor(1.0, dtype=dtype))
        return ops.cast(self.tot_num, dtype) / row_count_safe


    def _normalize_view_latents(self, vie, y_view, row_mask, scale_fact):
        """Apply the final layer's latent normalization to observed rows only."""
        batch_size = self._shape_fn(y_view)[0]
        row_indices = self._row_indices(row_mask)

        if keras.backend.backend() == "tensorflow":
            def normalize_present_rows():
                y_obs = self._gather_rows_by_index(y_view, row_indices)
                y_obs = self.model_list[vie].layers[-1].weight_normalizer([y_obs, scale_fact, self.train_DLV])
                return self._scatter_rows_by_index(
                    y_obs,
                    row_indices,
                    batch_size,
                    dtype=ops.dtype(y_obs),
                )

            def normalize_missing_rows():
                return self._zero_latents(batch_size, dtype=ops.dtype(y_view))

            return tf.cond(tf.shape(row_indices)[0] > 0, normalize_present_rows, normalize_missing_rows)

        if row_indices.numel() == 0:
            return self._zero_latents(batch_size, reference_tensor=y_view)

        y_obs = self._gather_rows_by_index(y_view, row_indices)
        y_obs = self.model_list[vie].layers[-1].weight_normalizer([y_obs, scale_fact, self.train_DLV])
        return self._scatter_rows_by_index(
            y_obs,
            row_indices,
            batch_size,
            dtype=y_obs.dtype,
            reference_tensor=y_view,
        )

    def _weight_normaliser(self, y, view_present):
        """This is an internal function designed to normalise weights and DLVs
        after each batch"""

        y_dtype = ops.dtype(y)

        # per-view normalization via last layer's weight_normalizer
        y_list = []
        scale_fact_list = []
        for vie in range(len(self.model_list)):
            y_view = y[:, :, vie]
            scale_fact = self._scale_factor_for_rows(view_present[:, vie], y_dtype)
            scale_fact_list.append(scale_fact)
            y_view = self._normalize_view_latents(vie, y_view, view_present[:, vie], scale_fact)
            y_list.append(y_view)

        y = ops.stack(y_list, axis=-1)

        return y, ops.stack(scale_fact_list)


    def train_step(self, inputs):
        """This is the main training set, it runs differently in tensorflow and torch"""



        total_loss = [None] * len(self.model_list)
        total_CC   = [None] * len(self.model_list)
        total_mse  = [None] * len(self.model_list)
        total_order = [None] * len(self.model_list)
        total_redundancy = [None] * len(self.model_list)


        # Unpack (tf.data-like packs inputs in a tuple/list)
        inputs = inputs[0]
        be = keras.backend.backend()  # 'tensorflow' | 'torch' | 'jax' (we handle tf/torch)
        inputs_nested = self.organize_inputs_by_model(inputs)

        if be == "tensorflow":
            y_raw, view_present = self._forward_views_with_missing(inputs_nested, training=self.train_DLV)
            y_ortho, scale_fact = self._weight_normaliser(y_raw, view_present)
        elif be == "torch":
            with torch.no_grad():
                y_raw, view_present = self._forward_views_with_missing(inputs_nested, training=self.train_DLV)
                y_ortho, scale_fact = self._weight_normaliser(y_raw, view_present)
        else:
            raise NotImplementedError(f"Backend '{be}' not supported in custom train_step.")

        if self.order:
            omega_batch = self._batch_structural_matrix(y_ortho, view_present)
            self._update_order_moving_omega(omega_batch)
            self._refresh_order_basis_and_scaling(dtype=ops.dtype(y_ortho))

        for vie in range(len(self.model_list)):
            source_mask = view_present[:, vie]

            if be == "tensorflow":
                zero = self._zero_scalar(dtype=ops.dtype(y_ortho))

                def run_step():
                    observed_inputs = self._gather_rows(inputs_nested[vie], source_mask)
                    return self._step_tf(
                        vie,
                        observed_inputs,
                        y_ortho,
                        view_present,
                        source_mask,
                        scale_fact[vie],
                    )

                def skip_step():
                    return zero, zero, zero

                loss, mse_loss, order_loss = tf.cond(
                    tf.reduce_any(source_mask),
                    run_step,
                    skip_step,
                )
            elif be == "torch":
                if bool(torch.any(source_mask).item()):
                    observed_inputs = self._gather_rows(inputs_nested[vie], source_mask)
                    loss, mse_loss, order_loss = self._step_torch(
                        vie,
                        observed_inputs,
                        y_ortho,
                        view_present,
                        source_mask,
                        scale_fact[vie],
                    )
                else:
                    loss = self._zero_scalar(dtype=ops.dtype(y_ortho))
                    mse_loss = self._zero_scalar(dtype=ops.dtype(y_ortho))
                    order_loss = self._zero_scalar(dtype=ops.dtype(y_ortho))

            total_loss[vie] = ops.sum(loss)
            total_CC[vie] = self.corr_metric(y_raw, y_raw[:,:,vie], vie, view_present)
            total_redundancy[vie] = self.calculate_redundancy(y_raw[:,:,vie], row_mask=view_present[:, vie])
            total_mse[vie]  = mse_loss
            total_order[vie] = order_loss

        mean_total_loss = ops.mean(ops.stack(total_loss))
        mean_order_loss = ops.mean(ops.stack(total_order))
        combined_total_loss = mean_total_loss

        self.loss_tracker_total.update_state(combined_total_loss)
        self.corr_tracker.update_state(ops.stack(total_CC))
        self.loss_tracker_mse.update_state(ops.stack(total_mse))
        self.loss_tracker_redundancy.update_state(ops.stack(total_redundancy))
        self.order_loss_tracker.update_state(mean_order_loss)

        return {
            "total_loss": self.loss_tracker_total.result(),
            "cross_metric": self.corr_tracker.result(),
            "mse_loss": self.loss_tracker_mse.result(),
            "redundancy": self.loss_tracker_redundancy.result(),
            "order_loss": self.order_loss_tracker.result(),
        }

    # class _StructuralOrderCallback(keras.callbacks.Callback):
    #     def __init__(self, model_ref):
    #         super().__init__()
    #         self._m = model_ref
    #         self._batch_count = 0
    #
    #     def on_train_begin(self, logs=None):
    #         self._batch_count = 0
    #         self._m._reset_structural_order_stats()
    #
    #     def on_train_batch_end(self, batch, logs=None):
    #         model_ref = self._m
    #         if not model_ref.order:
    #             return
    #
    #         self._batch_count += 1
    #         model_ref._apply_structural_ordering()
    #
    #     def on_train_end(self, logs=None):
    #         model_ref = self._m
    #         if not model_ref.order:
    #             return
    #
    #         model_ref._apply_structural_ordering()


    def compile(self, optimizer):
        """ Here, we overwrite the model compilation step. This is necessary as
        normally, the model compilation step would normally take a loss. Using
        this method, the loss is built into the method itself. We can either 
        pass the optimizer a single optimizer object, or a list of objects, with a 
        different optimizer used for each data-view.
        """
        
        super().compile()
        self.order_optimizer = None
        
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
        be = keras.backend.backend()
        inputs_nested = self.organize_inputs_by_model(inputs)
        y, view_present = self._forward_views_with_missing(inputs_nested, training=False)

        if self.order:
            self._refresh_order_basis_and_scaling(dtype=ops.dtype(y))

        total_loss = [None] * len(self.model_list)
        total_CC = [None] * len(self.model_list)
        total_mse = [None] * len(self.model_list)
        total_order = [None] * len(self.model_list)
        total_redundancy = [None] * len(self.model_list)

        for vie in range(len(self.model_list)):

            source_mask = view_present[:, vie]
            zero = self._zero_scalar(dtype=ops.dtype(y))

            if be == "tensorflow":
                def run_eval():
                    observed_inputs = self._gather_rows(inputs_nested[vie], source_mask)
                    y_pred_obs = self.model_list[vie](observed_inputs, training=False)
                    y_pred = self._scatter_rows_by_index(
                        y_pred_obs,
                        self._row_indices(source_mask),
                        self._shape_fn(y)[0],
                        dtype=ops.dtype(y_pred_obs),
                    )
                    mse_loss = self.mse_loss(y, y_pred, vie, view_present)
                    order_loss = self._encoder_order_regularizer(y_pred_obs)

                    internal_losses = self.model_list[vie].losses
                    if internal_losses:
                        internal_loss = ops.sum(
                            ops.stack(
                                [
                                    ops.convert_to_tensor(loss, dtype=ops.dtype(mse_loss))
                                    for loss in internal_losses
                                ],
                                axis=0,
                            ),
                            axis=0,
                        )
                    else:
                        internal_loss = ops.zeros_like(mse_loss)

                    loss = mse_loss + order_loss + internal_loss
                    corr = self.corr_metric(y, y_pred, vie, view_present)
                    return loss, mse_loss, order_loss, corr

                def skip_eval():
                    return zero, zero, zero, zero

                loss, mse_loss, order_loss, corr = tf.cond(
                    tf.reduce_any(source_mask),
                    run_eval,
                    skip_eval,
                )
            else:
                if bool(torch.any(source_mask).item()):
                    observed_inputs = self._gather_rows(inputs_nested[vie], source_mask)
                    y_pred_obs = self.model_list[vie](observed_inputs, training=False)
                    y_pred = self._scatter_rows_by_index(
                        y_pred_obs,
                        self._row_indices(source_mask),
                        self._shape_fn(y)[0],
                        dtype=y_pred_obs.dtype,
                        reference_tensor=self._reference_input(inputs_nested[vie]),
                    )
                    mse_loss = self.mse_loss(y, y_pred, vie, view_present)
                    order_loss = self._encoder_order_regularizer(y_pred_obs)

                    internal_losses = self.model_list[vie].losses
                    if internal_losses:
                        internal_loss = ops.sum(
                            ops.stack(
                                [
                                    ops.convert_to_tensor(loss, dtype=ops.dtype(mse_loss))
                                    for loss in internal_losses
                                ],
                                axis=0,
                            ),
                            axis=0,
                        )
                    else:
                        internal_loss = ops.zeros_like(mse_loss)

                    loss = mse_loss + order_loss + internal_loss
                    corr = self.corr_metric(y, y_pred, vie, view_present)
                else:
                    loss = zero
                    mse_loss = zero
                    order_loss = zero
                    corr = zero

            total_loss[vie] = ops.sum(loss)
            total_CC[vie]   = corr
            total_mse[vie]  = mse_loss
            total_order[vie] = order_loss
            total_redundancy[vie] = self.calculate_redundancy(y[:,:,vie], row_mask=view_present[:, vie])

        mean_total_loss = ops.mean(ops.stack(total_loss))
        mean_order_loss = ops.mean(ops.stack(total_order))
        combined_total_loss = mean_total_loss

        self.loss_tracker_total.update_state(combined_total_loss)
        self.corr_tracker.update_state(ops.stack(total_CC))
        self.loss_tracker_mse.update_state(ops.stack(total_mse))
        self.loss_tracker_redundancy.update_state(ops.stack(total_redundancy))
        self.order_loss_tracker.update_state(mean_order_loss)

        return {
            "total_loss": self.loss_tracker_total.result(),
            "cross_metric": self.corr_tracker.result(),
            "mse_loss": self.loss_tracker_mse.result(),
            "redundancy": self.loss_tracker_redundancy.result(),
            "order_loss": self.order_loss_tracker.result(),
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
            self.order_loss_tracker,
        ]


    def mse_loss(self, y_true, y_pred, vie, view_present):
    
        """
        Mean squared error between y_pred (view vie) and the connected views in y_true.

        """
        # y_true: (batch, ndims, n_views)
        # y_pred: (batch, ndims)

        dtype = ops.dtype(y_true)

        source_mask = ops.cast(view_present[:, vie], dtype)
        target_mask = ops.cast(view_present, dtype)
        pair_mask = ops.expand_dims(source_mask, axis=1) * target_mask

        y_pred_exp = ops.expand_dims(y_pred, axis=2)  # (batch, ndims, 1)
        sq_error = ops.square(y_true - y_pred_exp)
        sq_error = sq_error * ops.expand_dims(pair_mask, axis=1)

        counts = ops.sum(pair_mask, axis=0)
        counts_safe = ops.maximum(counts, ops.convert_to_tensor(1.0, dtype=dtype))
        se_mean = ops.sum(sq_error, axis=0) / ops.expand_dims(counts_safe, axis=0)

        connected_mask = ops.cast(self.Path[vie, :], dtype)
        valid_connected = connected_mask * ops.cast(counts > 0, dtype)
        se_mean_masked = se_mean * ops.expand_dims(valid_connected, axis=0)

        mse_loss = ops.sum(se_mean_masked) / 2.0
        return mse_loss


    def corr_metric(self, y_true, y_pred, vie, view_present):
        
        """
        Mean correlation between y_pred (view vie) and connected views in y_true.
        """

        if keras.backend.backend() == "torch":
            target_device = y_pred.device
            if y_true.device != target_device:
                y_true = y_true.to(target_device)
            if view_present.device != target_device:
                view_present = view_present.to(target_device)
            dtype = y_true.dtype
            eps = torch.as_tensor(self.epsilon, dtype=dtype, device=target_device)
            source_mask = view_present[:, vie].to(dtype=dtype)
            total_corr = torch.zeros((), dtype=dtype, device=target_device)
            total_weight = torch.zeros((), dtype=dtype, device=target_device)

            for target_vie in range(len(self.model_list)):
                connected = torch.as_tensor(self.Path[vie, target_vie], dtype=dtype, device=target_device)
                target_mask = view_present[:, target_vie].to(dtype=dtype)
                pair_mask = source_mask * target_mask
                pair_mask_exp = torch.unsqueeze(pair_mask, dim=1)

                pair_count = torch.sum(pair_mask)
                pair_count_safe = torch.maximum(
                    pair_count,
                    torch.tensor(1.0, dtype=dtype, device=target_device),
                )

                y_true_target = y_true[:, :, target_vie]
                y_true_mean = torch.sum(y_true_target * pair_mask_exp, dim=0) / pair_count_safe
                y_pred_mean = torch.sum(y_pred * pair_mask_exp, dim=0) / pair_count_safe

                y_true_c = (y_true_target - y_true_mean) * pair_mask_exp
                y_pred_c = (y_pred - y_pred_mean) * pair_mask_exp

                denom_true = torch.sqrt(torch.sum(torch.square(y_true_c), dim=0) + eps)
                denom_pred = torch.sqrt(torch.sum(torch.square(y_pred_c), dim=0) + eps)
                corr_dim = torch.sum((y_true_c / denom_true) * (y_pred_c / denom_pred), dim=0)
                pair_corr = torch.sum(corr_dim) / torch.tensor(float(self.ndims), dtype=dtype, device=target_device)

                valid_pair = connected * (pair_count > 1.0).to(dtype)
                total_corr = total_corr + pair_corr * valid_pair
                total_weight = total_weight + valid_pair

            total_weight_safe = torch.maximum(
                total_weight,
                torch.tensor(1.0, dtype=dtype, device=target_device),
            )
            return total_corr / total_weight_safe

        eps = ops.convert_to_tensor(self.epsilon, dtype=ops.dtype(y_true))
        dtype = ops.dtype(y_true)
        source_mask = ops.cast(view_present[:, vie], dtype)
        total_corr = ops.convert_to_tensor(0.0, dtype=dtype)
        total_weight = ops.convert_to_tensor(0.0, dtype=dtype)

        for target_vie in range(len(self.model_list)):
            connected = ops.cast(self.Path[vie, target_vie], dtype)
            target_mask = ops.cast(view_present[:, target_vie], dtype)
            pair_mask = source_mask * target_mask
            pair_mask_exp = ops.expand_dims(pair_mask, axis=1)

            pair_count = ops.sum(pair_mask)
            pair_count_safe = ops.maximum(pair_count, ops.convert_to_tensor(1.0, dtype=dtype))

            y_true_target = y_true[:, :, target_vie]
            y_true_mean = ops.sum(y_true_target * pair_mask_exp, axis=0) / pair_count_safe
            y_pred_mean = ops.sum(y_pred * pair_mask_exp, axis=0) / pair_count_safe

            y_true_c = (y_true_target - y_true_mean) * pair_mask_exp
            y_pred_c = (y_pred - y_pred_mean) * pair_mask_exp

            denom_true = ops.sqrt(ops.sum(ops.square(y_true_c), axis=0) + eps)
            denom_pred = ops.sqrt(ops.sum(ops.square(y_pred_c), axis=0) + eps)
            corr_dim = ops.sum((y_true_c / denom_true) * (y_pred_c / denom_pred), axis=0)
            pair_corr = ops.sum(corr_dim) / ops.convert_to_tensor(float(self.ndims), dtype=dtype)

            valid_pair = connected * ops.cast(pair_count > 1.0, dtype)
            total_corr = total_corr + pair_corr * valid_pair
            total_weight = total_weight + valid_pair

        total_weight_safe = ops.maximum(total_weight, ops.convert_to_tensor(1.0, dtype=dtype))
        return total_corr / total_weight_safe
    
     # This function avoids problems with passing symbolic 
     # tensors to ops.shape in tensorflow

    def _shape_fn(self,X):
        backend = keras.backend.backend()
        if backend == "tensorflow":
            shape = tf.shape(X)  # handles unknown ranks
        else:
            shape = ops.shape(X)
        return shape
    

    def calculate_redundancy(self, Y, epsilon=1e-8, row_mask=None):
        """
        Args:
            X: Tensor / KerasTensor, shape (N, D). Each column is a variable.
            epsilon: Small constant for numerical stability.

        Returns:
            Scalar tensor: mean(|corr(i, j)|) over all i != j.
        """
        Y = ops.convert_to_tensor(Y)
        Y = ops.cast(Y, "float32")

        if row_mask is None:
            row_mask = ops.cast(self._bool_fill(Y, True), Y.dtype)
        else:
            row_mask = ops.cast(row_mask, Y.dtype)
        row_mask = ops.expand_dims(row_mask, axis=1)

        n_f = ops.sum(row_mask)
        n_safe = ops.maximum(n_f, 1.0)

        # Center columns
        col_mean = ops.sum(Y * row_mask, axis=0, keepdims=True) / n_safe
        Yc = (Y - col_mean) * row_mask

        # Sample-size for covariance
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

        redundancy = total / num_pairs
        return redundancy * ops.cast(n_f > 1.0, redundancy.dtype)



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
            "Path": self._path_array.tolist(),
            "model_list": serialized_model_list,  # Include serialized model list in the configuration
            "regularizer_list": regularized_model_list,
            "tot_num": self.tot_num,
            "ndims": self.ndims,  
            "orthogonalization": self.orthogonalization,
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "train_DLV": self.train_DLV,
            "is_siamese": self.is_siamese,
            "diag_offset": self.diag_offset,
            "sparse_l1_list": self.sparse_l1_list,
            "orthog_weight": self.orthog_weight,
            "order": self.order,
            "order_loss_weight": self.order_loss_weight,
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
        config['Path'] = ops.convert_to_tensor(config["Path"], dtype=keras.backend.floatx())
        
        # Deserialize each model in the model list using a list comprehension
        config['model_list'] = [keras.utils.deserialize_keras_object(model_config) for model_config in config['model_list']]
        config['run_from_config'] = True
        config.pop("order_every", None)
        
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
    
