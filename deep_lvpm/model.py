
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


class _LastValueMetric(keras.metrics.Metric):
    """Track the most recent scalar value instead of averaging across batches."""

    def __init__(self, name="last_value", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        metric_dtype = dtype or keras.backend.floatx()
        self.value = self.add_weight(
            name=f"{name}_value",
            shape=(),
            initializer="zeros",
            dtype=metric_dtype,
        )

    def update_state(self, value):
        self.value.assign(ops.cast(value, self.value.dtype))

    def result(self):
        return self.value

    def reset_state(self):
        self.value.assign(ops.zeros_like(self.value))


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
        attention_mse=False,
        attention_gate=0.3,
        order=False,
        order_type="callback",
        order_loss_weight=1.0,
        order_association_cutoff=None,
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
        self.attention_mse = bool(attention_mse)
        self.attention_gate = float(attention_gate)
        if not (-1.0 <= self.attention_gate <= 1.0):
            raise ValueError("attention_gate must lie in the interval [-1, 1].")
        self.order = bool(order)
        if self.order and self.orthogonalization != 'zca':
            raise ValueError("'order' is only available when orthogonalization='zca'.")
        self.order_type = str(order_type).lower()
        if self.order_type not in {"callback", "loss", "both"}:
            raise ValueError("order_type must be one of: 'callback', 'loss', 'both'.")
        self.order_loss_weight = float(order_loss_weight)
        self.order_association_cutoff = (
            None if order_association_cutoff is None else float(order_association_cutoff)
        )
        if self.order_association_cutoff is not None:
            if not (0.0 < self.order_association_cutoff <= 1.0):
                raise ValueError("order_association_cutoff must lie in the interval (0, 1].")
            if not self.order or self.orthogonalization != "zca":
                raise ValueError(
                    "order_association_cutoff is only available when order=True and orthogonalization='zca'."
                )
        self.retained_order_dims = int(self.ndims)

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
        self.order_strength_tracker = keras.metrics.Mean(name="order_strength")
        n_views = len(self.model_list)
        float_dtype = keras.backend.floatx()
        self._order_moving_omega = self.add_weight(
            name="order_moving_omega",
            shape=(self.ndims, self.ndims),
            initializer="zeros",
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

    def _order_strength_metric(self, omega):
        dtype = ops.dtype(omega)
        omega = ops.cast(omega, dtype)
        omega = 0.5 * (omega + ops.transpose(omega))
        strengths = ops.diagonal(omega)

        if self.ndims < 2:
            return ops.cast(1.0, dtype)

        row_strengths = ops.expand_dims(strengths, axis=1)
        col_strengths = ops.expand_dims(strengths, axis=0)
        pairwise_correct = ops.cast(row_strengths > col_strengths, dtype)

        indices = ops.arange(self.ndims)
        row_indices = ops.expand_dims(indices, axis=1)
        col_indices = ops.expand_dims(indices, axis=0)
        upper_triangle = ops.cast(row_indices < col_indices, dtype)

        correct_pairs = ops.sum(pairwise_correct * upper_triangle)
        total_pairs = ops.maximum(ops.sum(upper_triangle), ops.cast(self.epsilon, dtype))
        return correct_pairs / total_pairs

    def _update_order_moving_omega(self, omega_batch):
        if not self._uses_order_basis():
            return

        current_omega = self._active_order_moving_omega()
        momentum = ops.cast(self.momentum, dtype=current_omega.dtype)
        one = ops.cast(1.0, dtype=current_omega.dtype)
        omega_cast = ops.cast(omega_batch, dtype=current_omega.dtype)
        state_mass = ops.sum(ops.abs(current_omega))
        zero_state = ops.cast(
            state_mass <= ops.cast(self.epsilon, current_omega.dtype),
            current_omega.dtype,
        )
        updated_omega = (
            zero_state * omega_cast
            + (one - zero_state)
            * (momentum * current_omega + (one - momentum) * omega_cast)
        )
        self._set_active_order_moving_omega(updated_omega)


    def _rotation_from_order_moving_omega(self, dtype=None):
        work_dtype = dtype or self._order_moving_omega.dtype
        omega = ops.cast(self._active_order_moving_omega(), work_dtype)
        omega = 0.5 * (omega + ops.transpose(omega))
        omega = omega + ops.cast(self.epsilon, work_dtype) * ops.eye(self.ndims, dtype=work_dtype)
        _, eigvecs = ops.linalg.eigh(omega)
        return ops.flip(eigvecs, axis=1)

    def _use_order_callback(self):
        return self.order and self.order_type in {"callback", "both"}

    def _uses_order_basis(self):
        return self.order

    def _use_order_dimension_pruning(self):
        return self.order_association_cutoff is not None

    def _active_order_moving_omega(self):
        return self._order_moving_omega[: self.ndims, : self.ndims]

    def _set_active_order_moving_omega(self, omega_small):
        omega_small = ops.cast(omega_small, self._order_moving_omega.dtype)

        if keras.backend.backend() == "tensorflow":
            full_dim = tf.shape(self._order_moving_omega)[0]
            small_dim = tf.shape(omega_small)[0]
            pad_dim = full_dim - small_dim
            paddings = tf.stack(
                [
                    tf.stack([0, pad_dim]),
                    tf.stack([0, pad_dim]),
                ]
            )
            full_omega = tf.pad(omega_small, paddings)
        else:
            full_dim = self._order_moving_omega.shape[0]
            small_dim = omega_small.shape[0]
            pad_dim = full_dim - small_dim
            full_omega = torch.nn.functional.pad(omega_small, (0, pad_dim, 0, pad_dim))

        self._order_moving_omega.assign(full_omega)

    def _detach_tensor(self, tensor):
        if keras.backend.backend() == "tensorflow":
            return tf.stop_gradient(tensor)
        return tensor.detach()

    def _apply_callback_ordering(self):
        if not self._use_order_callback():
            return False

        omega_mass = float(
            np.asarray(keras.ops.convert_to_numpy(ops.sum(ops.abs(self._active_order_moving_omega())))).item()
        )
        if omega_mass <= self.epsilon:
            return False

        rotation = self._rotation_from_order_moving_omega()
        self._apply_structural_rotation(rotation)
        return True

    def _retained_dims_from_order_omega(self):
        omega = np.asarray(
            keras.ops.convert_to_numpy(self._active_order_moving_omega()),
            dtype=np.float64,
        )
        omega = 0.5 * (omega + omega.T)
        eigvals = np.linalg.eigvalsh(omega)[::-1]
        strengths = np.maximum(eigvals, 0.0)
        total_strength = float(np.sum(strengths))

        if total_strength <= float(self.epsilon):
            return int(self.ndims)

        cumulative_strength = np.cumsum(strengths) / total_strength
        retained_dims = int(np.searchsorted(cumulative_strength, self.order_association_cutoff) + 1)
        retained_dims = max(1, min(int(self.ndims), retained_dims))
        return retained_dims

    def _rebuild_model_with_resized_zca(self, model, retained_dims):
        last_layer = model.layers[-1]
        if not isinstance(last_layer, ZCALayer):
            raise ValueError("order_association_cutoff currently only supports models ending in ZCALayer.")

        if len(model.layers) < 2:
            raise ValueError("Cannot resize a model that does not have a pre-ZCA feature layer.")

        sliced_project = np.asarray(
            keras.ops.convert_to_numpy(last_layer.project[:, :retained_dims]),
            dtype=keras.backend.floatx(),
        )
        sliced_cov = np.asarray(
            keras.ops.convert_to_numpy(last_layer.moving_conv2[:retained_dims, :retained_dims]),
            dtype=keras.backend.floatx(),
        )
        bn_weights = last_layer.batch_norm1.get_weights()
        run_value = np.asarray(keras.ops.convert_to_numpy(last_layer.run), dtype=keras.backend.floatx())

        penultimate_output = model.layers[-2].output
        model_inputs = model.inputs if len(model.inputs) > 1 else model.inputs[0]
        new_last_layer = ZCALayer(
            kernel_regularizer=last_layer.kernel_regularizer,
            epsilon=last_layer.epsilon,
            momentum=last_layer.momentum,
            diag_offset=last_layer.diag_offset,
            tot_num=last_layer.tot_num,
            ndims=retained_dims,
            sparse_l1=last_layer.sparse_l1,
            name=last_layer.name,
        )
        new_output = new_last_layer(penultimate_output)
        new_model = keras.Model(inputs=model_inputs, outputs=new_output, name=model.name)

        resized_last_layer = new_model.layers[-1]
        resized_last_layer.batch_norm1.set_weights(bn_weights)
        resized_last_layer.project.assign(sliced_project)
        resized_last_layer.moving_conv2.assign(sliced_cov)
        resized_last_layer.run.assign(run_value)

        old_optimizer = getattr(model, "optimizer", None)
        if old_optimizer is not None:
            new_model.compile(self._clone_optimizer(old_optimizer))

        new_model.train_function = None
        new_model.test_function = None
        new_model.predict_function = None
        return new_model

    def _resize_ordered_zca_dimensions(self, retained_dims, rotated_omega=None):
        retained_dims = int(retained_dims)
        self.retained_order_dims = retained_dims

        old_ndims = int(self.ndims)
        if retained_dims >= old_ndims:
            if rotated_omega is not None:
                self._set_active_order_moving_omega(rotated_omega)
            return

        rebuilt_models = {}
        for index, mdl in enumerate(list(self.model_list)):
            model_id = id(mdl)
            if model_id not in rebuilt_models:
                rebuilt_models[model_id] = self._rebuild_model_with_resized_zca(mdl, retained_dims)
            self.model_list[index] = rebuilt_models[model_id]

        self.ndims = retained_dims

        if rotated_omega is None:
            rotated_omega = self._active_order_moving_omega()
        rotated_omega = rotated_omega[:retained_dims, :retained_dims]
        self._set_active_order_moving_omega(rotated_omega)

        self.train_function = None
        self.test_function = None
        self.predict_function = None

    def _finalize_ordered_dimensions(self):
        omega_mass = float(
            np.asarray(keras.ops.convert_to_numpy(ops.sum(ops.abs(self._active_order_moving_omega())))).item()
        )
        if omega_mass <= self.epsilon:
            return False

        rotated = False
        rotation = None
        if self._use_order_callback() or self._use_order_dimension_pruning():
            rotation = self._rotation_from_order_moving_omega()
            self._apply_structural_rotation(rotation)
            rotated = True

        if self._use_order_dimension_pruning():
            omega_before = ops.cast(self._active_order_moving_omega(), dtype=rotation.dtype)
            rotated_omega = ops.matmul(
                ops.transpose(rotation),
                ops.matmul(omega_before, rotation),
            )
            rotated_omega = 0.5 * (rotated_omega + ops.transpose(rotated_omega))
            retained_dims = self._retained_dims_from_order_omega()
            self._resize_ordered_zca_dimensions(retained_dims, rotated_omega=rotated_omega)
            print(
                f"Retained {retained_dims} of {int(rotation.shape[0])} ordered dimensions "
                f"using omega association mass cutoff {self.order_association_cutoff:.2f}."
            )
        elif rotated:
            omega_before = ops.cast(self._active_order_moving_omega(), dtype=rotation.dtype)
            rotated_omega = ops.matmul(
                ops.transpose(rotation),
                ops.matmul(omega_before, rotation),
            )
            rotated_omega = 0.5 * (rotated_omega + ops.transpose(rotated_omega))
            self._set_active_order_moving_omega(rotated_omega)

        return rotated

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
            mse_loss = self.reconstruction_loss(y, y_pred, vie, view_present)
            internal_loss = tf.add_n(model.losses) if model.losses else tf.cast(0.0, mse_loss.dtype)
            loss = mse_loss + internal_loss

        trainable_vars = model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        grads_and_vars = [(grad, var) for grad, var in zip(grads, trainable_vars) if grad is not None]
        if grads_and_vars:
            model.optimizer.apply_gradients(grads_and_vars)

        return loss, mse_loss

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
        mse_loss = self.reconstruction_loss(y, y_pred, vie, view_present)

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
        fixed_grads = []
        for grad, var in zip(grads, trainable_vars):
            if grad is None:
                grad = torch.zeros_like(getattr(var, "value", var))
            fixed_grads.append(grad)
        model.optimizer.apply_gradients(zip(fixed_grads, trainable_vars))

        return loss, mse_loss


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
        total_redundancy = [None] * len(self.model_list)

        # Unpack (tf.data-like packs inputs in a tuple/list)
        inputs = inputs[0]
        be = keras.backend.backend()  # 'tensorflow' | 'torch' | 'jax' (we handle tf/torch)
        inputs_nested = self.organize_inputs_by_model(inputs)

        if be == "tensorflow":
            y_raw, view_present = self._forward_views_with_missing(inputs_nested, training=self.train_DLV)
            # We build y_ortho from a separate full-batch forward here, then re-run
            # each source view under the gradient tape below. This costs extra compute,
            # but keeps only one view's activations in the gradient graph at a time,
            # which substantially reduces peak memory use.
            y_ortho, scale_fact = self._weight_normaliser(y_raw, view_present)
        elif be == "torch":
            with torch.no_grad():
                y_raw, view_present = self._forward_views_with_missing(inputs_nested, training=self.train_DLV)
                # We build y_ortho from a separate full-batch forward here, then re-run
                # each source view with gradients below. This costs extra compute, but
                # avoids retaining all views in one large autograd graph at once.
                y_ortho, scale_fact = self._weight_normaliser(y_raw, view_present)
        else:
            raise NotImplementedError(f"Backend '{be}' not supported in custom train_step.")

        omega_batch = self._batch_structural_matrix(y_raw, view_present)
        order_strength = self._order_strength_metric(omega_batch)
        if self._uses_order_basis():
            self._update_order_moving_omega(omega_batch)

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
                    return zero, zero

                loss, mse_loss = tf.cond(
                    tf.reduce_any(source_mask),
                    run_step,
                    skip_step,
                )
            elif be == "torch":
                if bool(torch.any(source_mask).item()):
                    observed_inputs = self._gather_rows(inputs_nested[vie], source_mask)
                    loss, mse_loss = self._step_torch(
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

            total_loss[vie] = ops.sum(loss)
            total_CC[vie] = self.corr_metric(y_raw, y_raw[:,:,vie], vie, view_present)
            total_redundancy[vie] = self.calculate_redundancy(y_raw[:,:,vie], row_mask=view_present[:, vie])
            total_mse[vie]  = mse_loss

        mean_total_loss = ops.mean(ops.stack(total_loss))
        combined_total_loss = mean_total_loss

        self.loss_tracker_total.update_state(combined_total_loss)
        self.corr_tracker.update_state(ops.stack(total_CC))
        self.loss_tracker_mse.update_state(ops.stack(total_mse))
        self.loss_tracker_redundancy.update_state(ops.stack(total_redundancy))
        self.order_strength_tracker.update_state(order_strength)

        return {
            "total_loss": self.loss_tracker_total.result(),
            "cross_metric": self.corr_tracker.result(),
            "mse_loss": self.loss_tracker_mse.result(),
            "redundancy": self.loss_tracker_redundancy.result(),
            "order_strength": self.order_strength_tracker.result(),
        }

    class _StructuralOrderCallback(keras.callbacks.Callback):
        def __init__(self, model_ref):
            super().__init__()
            self._m = model_ref

        def on_train_end(self, logs=None):
            self._m._finalize_ordered_dimensions()


    def fit(self, *args, **kwargs):
        callbacks = list(kwargs.pop("callbacks", []) or [])
        if self._use_order_callback():
            callbacks.append(self._StructuralOrderCallback(self))
        elif self._use_order_dimension_pruning():
            callbacks.append(self._StructuralOrderCallback(self))
        kwargs["callbacks"] = callbacks
        return super().fit(*args, **kwargs)


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
        be = keras.backend.backend()
        inputs_nested = self.organize_inputs_by_model(inputs)
        y_raw, view_present = self._forward_views_with_missing(inputs_nested, training=False)
        y_ortho, _ = self._weight_normaliser(y_raw, view_present)
        omega_batch = self._batch_structural_matrix(y_raw, view_present)
        order_strength = self._order_strength_metric(omega_batch)

        # if self.order:
        #     self._refresh_order_basis_and_scaling(dtype=ops.dtype(y))

        total_loss = [None] * len(self.model_list)
        total_CC = [None] * len(self.model_list)
        total_mse = [None] * len(self.model_list)
        total_redundancy = [None] * len(self.model_list)

        for vie in range(len(self.model_list)):

            source_mask = view_present[:, vie]
            zero = self._zero_scalar(dtype=ops.dtype(y_ortho))

            if be == "tensorflow":
                def run_eval():
                    observed_inputs = self._gather_rows(inputs_nested[vie], source_mask)
                    y_pred_obs = self.model_list[vie](observed_inputs, training=False)
                    scale_fact = self._scale_factor_for_rows(source_mask, ops.dtype(y_pred_obs))
                    y_pred_obs_norm = self._normalize_pred(y_pred_obs, scale_fact)
                    y_pred = self._scatter_rows_by_index(
                        y_pred_obs_norm,
                        self._row_indices(source_mask),
                        self._shape_fn(y_ortho)[0],
                        dtype=ops.dtype(y_pred_obs_norm),
                    )
                    mse_loss = self.reconstruction_loss(y_ortho, y_pred, vie, view_present)
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

                    loss = mse_loss + internal_loss
                    corr = self.corr_metric(y_raw, y_pred, vie, view_present)
                    return loss, mse_loss, corr

                def skip_eval():
                    return zero, zero, zero

                loss, mse_loss, corr = tf.cond(
                    tf.reduce_any(source_mask),
                    run_eval,
                    skip_eval,
                )
            else:
                if bool(torch.any(source_mask).item()):
                    observed_inputs = self._gather_rows(inputs_nested[vie], source_mask)
                    y_pred_obs = self.model_list[vie](observed_inputs, training=False)
                    scale_fact = self._scale_factor_for_rows(source_mask, y_pred_obs.dtype)
                    y_pred_obs_norm = self._normalize_pred(y_pred_obs, scale_fact)
                    y_pred = self._scatter_rows_by_index(
                        y_pred_obs_norm,
                        self._row_indices(source_mask),
                        self._shape_fn(y_ortho)[0],
                        dtype=y_pred_obs_norm.dtype,
                        reference_tensor=self._reference_input(inputs_nested[vie]),
                    )
                    mse_loss = self.reconstruction_loss(y_ortho, y_pred, vie, view_present)

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

                    loss = mse_loss + internal_loss
                    corr = self.corr_metric(y_raw, y_pred, vie, view_present)
                else:
                    loss = zero
                    mse_loss = zero
                    corr = zero

            total_loss[vie] = ops.sum(loss)
            total_CC[vie]   = corr
            total_mse[vie]  = mse_loss
            total_redundancy[vie] = self.calculate_redundancy(y_raw[:,:,vie], row_mask=view_present[:, vie])

        mean_total_loss = ops.mean(ops.stack(total_loss))
        combined_total_loss = mean_total_loss

        self.loss_tracker_total.update_state(combined_total_loss)
        self.corr_tracker.update_state(ops.stack(total_CC))
        self.loss_tracker_mse.update_state(ops.stack(total_mse))
        self.loss_tracker_redundancy.update_state(ops.stack(total_redundancy))
        self.order_strength_tracker.update_state(order_strength)

        return {
            "total_loss": self.loss_tracker_total.result(),
            "cross_metric": self.corr_tracker.result(),
            "mse_loss": self.loss_tracker_mse.result(),
            "redundancy": self.loss_tracker_redundancy.result(),
            "order_strength": self.order_strength_tracker.result(),
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
            self.order_strength_tracker,
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

    def attention_mse_loss(self, y_true, y_pred, vie, view_present):

        """
        Mean squared error weighted by per-dimension softmax-normalised
        Pearson correlations across connected target views.
        """

        dtype = ops.dtype(y_true)
        eps = ops.convert_to_tensor(self.epsilon, dtype=dtype)
        one = ops.convert_to_tensor(1.0, dtype=dtype)
        zero = ops.convert_to_tensor(0.0, dtype=dtype)
        mask_penalty = ops.convert_to_tensor(30.0, dtype=dtype)
        gate_threshold = ops.convert_to_tensor(self.attention_gate, dtype=dtype)

        source_mask = ops.cast(view_present[:, vie], dtype)
        target_mask = ops.cast(view_present, dtype)
        pair_mask = ops.expand_dims(source_mask, axis=1) * target_mask

        y_pred_exp = ops.expand_dims(y_pred, axis=2)  # (batch, ndims, 1)
        sq_error = ops.square(y_true - y_pred_exp)
        sq_error = sq_error * ops.expand_dims(pair_mask, axis=1)

        counts = ops.sum(pair_mask, axis=0)
        counts_safe = ops.maximum(counts, one)
        se_mean = ops.sum(sq_error, axis=0) / ops.expand_dims(counts_safe, axis=0)

        corr_scores = []
        valid_targets = []
        path_weights = []

        for target_vie in range(len(self.model_list)):
            y_true_target = y_true[:, :, target_vie]
            pair_mask_target = pair_mask[:, target_vie]
            pair_mask_exp = ops.expand_dims(pair_mask_target, axis=1)
            pair_count = counts[target_vie]
            pair_count_safe = counts_safe[target_vie]

            y_true_mean = ops.sum(y_true_target * pair_mask_exp, axis=0) / pair_count_safe
            y_pred_mean = ops.sum(y_pred * pair_mask_exp, axis=0) / pair_count_safe

            y_true_centered = (y_true_target - y_true_mean) * pair_mask_exp
            y_pred_centered = (y_pred - y_pred_mean) * pair_mask_exp

            denom_true = ops.sqrt(ops.sum(ops.square(y_true_centered), axis=0) + eps)
            denom_pred = ops.sqrt(ops.sum(ops.square(y_pred_centered), axis=0) + eps)
            corr_dim = ops.sum(
                (y_true_centered / denom_true) * (y_pred_centered / denom_pred),
                axis=0,
            )

            if self._path_array.ndim == 3:
                path_weight = ops.cast(self.Path[vie, target_vie, :], dtype)
            else:
                path_weight = ops.cast(self.Path[vie, target_vie], dtype) * ops.ones(
                    (self.ndims,),
                    dtype=dtype,
                )

            connected = ops.cast(path_weight > zero, dtype)
            valid_target = connected * ops.cast(pair_count > one, dtype)

            corr_scores.append(corr_dim)
            valid_targets.append(valid_target)
            path_weights.append(path_weight)

        corr_scores = ops.stack(corr_scores, axis=1)   # (ndims, n_views)
        valid_targets = ops.stack(valid_targets, axis=1)  # (ndims, n_views)
        path_weights = ops.stack(path_weights, axis=1)  # (ndims, n_views)

        gated_targets = valid_targets * ops.cast(corr_scores >= gate_threshold, dtype)
        masked_scores = corr_scores - (one - gated_targets) * mask_penalty
        attention_weights = ops.softmax(masked_scores, axis=1)
        comparison_count = ops.maximum(
            ops.sum(gated_targets, axis=1, keepdims=True),
            one,
        )
        attention_weights = attention_weights * comparison_count
        attention_weights = attention_weights * path_weights * gated_targets
        attention_weights = self._detach_tensor(attention_weights)

        mse_loss = ops.sum(se_mean * attention_weights) / 2.0
        return mse_loss

    def reconstruction_loss(self, y_true, y_pred, vie, view_present):
        if self.attention_mse:
            return self.attention_mse_loss(y_true, y_pred, vie, view_present)
        return self.mse_loss(y_true, y_pred, vie, view_present)


    def corr_metric(self, y_true, y_pred, vie, view_present):
        
        """
        Mean Pearson correlation (r) between y_pred (view vie) and connected views in y_true.
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
                pair_corr = torch.sum(corr_dim) / torch.tensor(
                    float(self.ndims), dtype=dtype, device=target_device
                )

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
            "attention_mse": self.attention_mse,
            "attention_gate": self.attention_gate,
            "order": self.order,
            "order_type": self.order_type,
            "order_loss_weight": self.order_loss_weight,
            "order_association_cutoff": self.order_association_cutoff,
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
        config.pop("callback_every", None)
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
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Second-order DLVPM implemented as block-sequential gradient descent.

This version keeps DLVPM itself intact:
- the same measurement-model API,
- the same path matrix,
- the same DLVPM losses,
- the same orthogonalisation machinery,
- the same missing-view handling,
- the same per-view optimizers.

The only algorithmic change is the optimisation schedule:
one view is the active block for `block_steps` mini-batches, only that
view receives a gradient update, and then training moves to the next
view in sequence.

This is "second-order" only in the broad RGCCA / Gauss–Seidel sense
requested by the user: blockwise sequential optimisation. It is not a
Newton or Hessian-based method.
"""

import numpy as np
import keras as keras
from keras import ops


be = keras.backend.backend()  # 'tensorflow' | 'torch'

if be == "tensorflow":
    try:
        import tensorflow as tf
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Tensorflow backend requested but it is not installed. "
            "Install Tensorflow or switch Keras backend to Torch."
        ) from e
elif be == "torch":
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Torch backend requested but it is not installed. "
            "Install Torch or switch Keras backend to TensorFlow."
        ) from e


@keras.utils.register_keras_serializable(
    package="deep_lvpm", name="SecondOrderStructuralModel"
)
class SecondOrderStructuralModel(StructuralModel):
    """
    DLVPM trained with block-sequential gradient descent.

    Compared with StructuralModel, the objective function and layer stack are
    unchanged. The only change is that one view is selected as the active view,
    gradient descent is applied only to that view for `block_steps` batches,
    and then the active view moves to the next view in a cyclic schedule.

    Parameters added on top of StructuralModel
    ------------------------------------------
    block_steps : int
        Number of mini-batches spent on one active view before switching.

    view_sequence : list[int] or None
        Explicit order in which views are updated. If None, the model updates
        every view whose path-model row contains at least one connection.
    """

    def __init__(
        self,
        Path,
        model_list,
        regularizer_list,
        tot_num,
        ndims,
        orthogonalization="Moore-Penrose",
        momentum=0.95,
        epsilon=1e-4,
        train_DLV=True,
        run_from_config=False,
        is_siamese=False,
        diag_offset=1e-3,
        sparse_l1_list=0.0,
        attention_mse=False,
        attention_gate=0.3,
        order=False,
        order_type="callback",
        order_loss_weight=1.0,
        block_steps=4,
        view_sequence=None,
        **kwargs,
    ):
        self.block_steps = int(block_steps)
        if self.block_steps <= 0:
            raise ValueError("block_steps must be a positive integer.")

        self._requested_view_sequence = None if view_sequence is None else list(view_sequence)

        super().__init__(
            Path=Path,
            model_list=model_list,
            regularizer_list=regularizer_list,
            tot_num=tot_num,
            ndims=ndims,
            orthogonalization=orthogonalization,
            momentum=momentum,
            epsilon=epsilon,
            train_DLV=train_DLV,
            run_from_config=run_from_config,
            is_siamese=is_siamese,
            diag_offset=diag_offset,
            sparse_l1_list=sparse_l1_list,
            attention_mse=attention_mse,
            attention_gate=attention_gate,
            order=order,
            order_type=order_type,
            order_loss_weight=order_loss_weight,
            **kwargs,
        )

        self.view_sequence = self._build_view_sequence(self._requested_view_sequence)
        self._schedule_size = len(self.view_sequence)

        # Integer state for the block-sequential schedule.
        self._schedule_position = self.add_weight(
            name="second_order_schedule_position",
            shape=(),
            initializer="zeros",
            dtype="int32",
            trainable=False,
        )
        self._batches_in_active_view = self.add_weight(
            name="second_order_batches_in_active_view",
            shape=(),
            initializer="zeros",
            dtype="int32",
            trainable=False,
        )

        if be == "tensorflow":
            self._view_sequence_tensor = tf.constant(self.view_sequence, dtype=tf.int32)
            self._schedule_size_tensor = tf.constant(self._schedule_size, dtype=tf.int32)

        self.active_view_tracker = _LastValueMetric(name="active_view")

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _build_view_sequence(self, requested_sequence):
        n_views = len(self.model_list)

        if requested_sequence is not None:
            sequence = [int(v) for v in requested_sequence]
            if len(sequence) == 0:
                raise ValueError("view_sequence cannot be empty.")
            for v in sequence:
                if v < 0 or v >= n_views:
                    raise ValueError(
                        f"view_sequence contains invalid view index {v}; "
                        f"valid range is [0, {n_views - 1}]."
                    )
            return sequence

        sequence = [
            vie for vie in range(n_views)
            if np.any(self._path_array[vie, :] != 0.0)
        ]
        if not sequence:
            sequence = list(range(n_views))
        return sequence

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "block_steps": self.block_steps,
                "view_sequence": list(self.view_sequence),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config.setdefault("block_steps", 4)
        config.setdefault("view_sequence", None)
        return super().from_config(config)

    # ------------------------------------------------------------------
    # Compile / fit
    # ------------------------------------------------------------------
    def compile(self, optimizer, run_eagerly=False):
        """
        Compile the per-view submodels exactly as in StructuralModel, but allow
        `run_eagerly` to be set explicitly.
        """
        keras.Model.compile(self, run_eagerly=run_eagerly)

        if isinstance(optimizer, list):
            if len(optimizer) != len(self.model_list):
                raise ValueError(
                    f"Expected {len(self.model_list)} optimizers, got {len(optimizer)}."
                )
            for vie in range(len(self.model_list)):
                self.model_list[vie].compile(optimizer[vie])
        elif isinstance(optimizer, keras.optimizers.Optimizer):
            if self.is_siamese:
                self.model_list[0].compile(optimizer)
            else:
                for vie in range(len(self.model_list)):
                    self.model_list[vie].compile(self._clone_optimizer(optimizer))
        else:
            raise TypeError(
                "optimizer must either be a keras optimizer instance, or a list of them."
            )

    def _reset_second_order_schedule(self):
        zero = ops.convert_to_tensor(0, dtype=self._schedule_position.dtype)
        self._schedule_position.assign(zero)
        self._batches_in_active_view.assign(zero)

    def fit(self, *args, **kwargs):
        callbacks = list(kwargs.pop("callbacks", []) or [])
        has_progbar = any(isinstance(callback, keras.callbacks.ProgbarLogger) for callback in callbacks)
        if not has_progbar:
            callbacks.append(self._SecondOrderProgbarLogger())
        kwargs["callbacks"] = callbacks
        self._reset_second_order_schedule()
        return super().fit(*args, **kwargs)

    # ------------------------------------------------------------------
    # Schedule helpers
    # ------------------------------------------------------------------
    def _current_active_view_tf(self):
        return tf.gather(self._view_sequence_tensor, self._schedule_position)

    def _current_active_view_torch(self):
        position = int(np.asarray(keras.ops.convert_to_numpy(self._schedule_position)).item())
        return int(self.view_sequence[position])

    def _advance_schedule_tf(self):
        one = tf.constant(1, dtype=self._batches_in_active_view.dtype)
        new_batches = self._batches_in_active_view + one
        rotate = new_batches >= tf.cast(self.block_steps, self._batches_in_active_view.dtype)
        next_position = tf.math.floormod(
            self._schedule_position + one,
            self._schedule_size_tensor,
        )

        self._schedule_position.assign(
            tf.where(rotate, next_position, self._schedule_position)
        )
        self._batches_in_active_view.assign(
            tf.where(rotate, tf.zeros_like(new_batches), new_batches)
        )

    def _advance_schedule_torch(self):
        position = int(np.asarray(keras.ops.convert_to_numpy(self._schedule_position)).item())
        batches = int(np.asarray(keras.ops.convert_to_numpy(self._batches_in_active_view)).item())

        batches += 1
        if batches >= self.block_steps:
            batches = 0
            position = (position + 1) % self._schedule_size

        self._schedule_position.assign(
            ops.convert_to_tensor(position, dtype=self._schedule_position.dtype)
        )
        self._batches_in_active_view.assign(
            ops.convert_to_tensor(batches, dtype=self._batches_in_active_view.dtype)
        )

    class _SecondOrderProgbarLogger(keras.callbacks.ProgbarLogger):
        def _maybe_init_progbar(self):
            super()._maybe_init_progbar()
            if self.progbar is not None:
                self.progbar.stateful_metrics.add("active_view")

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def train_step(self, inputs):
        """
        Block-sequential DLVPM training step.

        Targets are still constructed exactly as in the original DLVPM:
        all views are forward-propagated, the DLVPM orthogonalisation / weight
        normalisation step is applied, and only the currently active view then
        receives a gradient update against the fixed targets from that batch.
        """

        total_loss = [None] * len(self.model_list)
        total_CC = [None] * len(self.model_list)
        total_mse = [None] * len(self.model_list)
        total_redundancy = [None] * len(self.model_list)

        inputs = inputs[0]
        inputs_nested = self.organize_inputs_by_model(inputs)
        backend = keras.backend.backend()

        if backend == "tensorflow":
            active_view = self._current_active_view_tf()
            y_raw, view_present = self._forward_views_with_missing(
                inputs_nested,
                training=self.train_DLV,
            )
            y_ortho, scale_fact = self._weight_normaliser(y_raw, view_present)
        elif backend == "torch":
            active_view = self._current_active_view_torch()
            with torch.no_grad():
                y_raw, view_present = self._forward_views_with_missing(
                    inputs_nested,
                    training=self.train_DLV,
                )
                y_ortho, scale_fact = self._weight_normaliser(y_raw, view_present)
        else:  # pragma: no cover
            raise NotImplementedError(
                f"Backend '{backend}' not supported in custom train_step."
            )

        omega_batch = self._batch_structural_matrix(y_raw, view_present)
        order_strength = self._order_strength_metric(omega_batch)
        if self._uses_order_basis():
            self._update_order_moving_omega(omega_batch)

        for vie in range(len(self.model_list)):
            source_mask = view_present[:, vie]

            if backend == "tensorflow":
                zero = self._zero_scalar(dtype=ops.dtype(y_ortho))
                is_active = tf.equal(active_view, tf.cast(vie, active_view.dtype))

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

                def skip_missing():
                    return zero, zero

                def active_branch():
                    return tf.cond(tf.reduce_any(source_mask), run_step, skip_missing)

                def inactive_branch():
                    return zero, zero

                loss, mse_loss = tf.cond(is_active, active_branch, inactive_branch)

            elif backend == "torch":
                if vie == active_view and bool(torch.any(source_mask).item()):
                    observed_inputs = self._gather_rows(inputs_nested[vie], source_mask)
                    loss, mse_loss = self._step_torch(
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

            total_loss[vie] = ops.sum(loss)
            total_CC[vie] = self.corr_metric(y_raw, y_raw[:, :, vie], vie, view_present)
            total_redundancy[vie] = self.calculate_redundancy(
                y_raw[:, :, vie],
                row_mask=view_present[:, vie],
            )
            total_mse[vie] = mse_loss

        # Only one block is optimized per batch, so the natural training loss is
        # the active block loss rather than the mean over all blocks.
        active_total_loss = ops.sum(ops.stack(total_loss))
        active_mse_loss = ops.sum(ops.stack(total_mse))

        self.loss_tracker_total.update_state(active_total_loss)
        self.corr_tracker.update_state(ops.stack(total_CC))
        self.loss_tracker_mse.update_state(active_mse_loss)
        self.loss_tracker_redundancy.update_state(ops.stack(total_redundancy))
        self.order_strength_tracker.update_state(order_strength)
        self.active_view_tracker.update_state(active_view)

        if backend == "tensorflow":
            self._advance_schedule_tf()
        else:
            self._advance_schedule_torch()

        return {
            "total_loss": self.loss_tracker_total.result(),
            "cross_metric": self.corr_tracker.result(),
            "mse_loss": self.loss_tracker_mse.result(),
            "redundancy": self.loss_tracker_redundancy.result(),
            "order_strength": self.order_strength_tracker.result(),
            "active_view": self.active_view_tracker.result(),
        }

    @property
    def metrics(self):
        return super().metrics + [self.active_view_tracker]
