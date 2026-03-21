#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal CLIP-style model.

This class coordinates multiple per-view encoders by appending a simple
Dense projection head of size `ndims` to each, then optimizes a CLIP loss
across all ordered view pairs. It does not use StructuralModel/orthogonalisation
machinery and has no adjacency/Path concept.
"""

import numpy as np
import keras as keras
from keras import ops
from keras import layers


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


@keras.utils.register_keras_serializable(package="deep_lvpm", name="CLIP")
class CLIP(keras.Model):
    """
    A CLIP-style multi-view model mirroring StructuralModel's orchestration.

    Measurement models (one per data view) are appended with a Dense head of
    size `ndims`, and the global model coordinates training to align the views.

    Note: Orthogonalisation layers are not used; only Dense projection heads are
    attached. Losses remain unchanged from StructuralModel at this stage.
    """

    def __init__(self, model_list, regularizer_list, ndims,
                 run_from_config=False, is_siamese=False, **kwargs):

        super().__init__(**kwargs)
        self.ndims = ndims
        self.regularizer_list = regularizer_list
        self.is_siamese = is_siamese

        if not run_from_config:
            if self.is_siamese is True:
                new_model = self.add_CLIP_layer(model_list[0], regularizer_list[0])
                self.model_list = [new_model] * len(model_list)
            else:
                self.model_list = [
                    self.add_CLIP_layer(model, regularizer)
                    for model, regularizer in zip(model_list, regularizer_list)
                ]
        else:
            self.model_list = model_list

        # CLIP objective tracking
        self.clip_loss_tracker = keras.metrics.Mean(name="clip_loss")
        # Learnable temperature (1 / tau) as logit scale
        self.logit_scale = self.add_weight(
            name="logit_scale",
            shape=(),
            initializer=keras.initializers.Constant(float(np.log(1.0 / 0.07))),
            trainable=True,
        )

    def add_CLIP_layer(self, model, regularizer):
        """
        Append a Dense projection head (units=ndims) to the given measurement model.

        Supports both Sequential and Functional Keras models.
        """

        if isinstance(model, keras.Sequential):
            proj = layers.Dense(self.ndims, activation=None, name='clip_projection',
                                kernel_regularizer=regularizer)
            model.add(proj)
        elif isinstance(model, keras.Model):
            x = layers.Dense(self.ndims, activation=None, name='clip_projection',
                             kernel_regularizer=regularizer)(model.output)
            model = keras.Model(inputs=model.input, outputs=x)
        else:
            raise ValueError("The input model must be either a keras.Sequential or a keras.Model instance.")

        return model

    def call(self, inputs, training=False):
        # Return L2-normalized embeddings stacked as (B, d, M)
        return self._encode_all(inputs, training)

    def organize_inputs_by_model(self, data_inputs):
        organized_inputs = []
        data_index = 0

        for model in self.model_list:
            num_inputs = len(model.inputs) if hasattr(model, 'inputs') else 1
            if num_inputs == 1:
                organized_inputs.append(data_inputs[data_index])
                data_index += 1
            else:
                inputs_for_model = data_inputs[data_index:data_index + num_inputs]
                organized_inputs.append(inputs_for_model)
                data_index += num_inputs

        return organized_inputs

    # CLIP helpers

    def _encode_all(self, inputs, training):
        """Run all measurement models and L2-normalize embeddings.

        Returns a tensor Z with shape (B, d, M).
        """
        inputs_nested = self.organize_inputs_by_model(inputs)
        zs = [
            self.model_list[m](inputs_nested[m], training=training)
            for m in range(len(self.model_list))
        ]
        Z = ops.stack(zs, axis=-1)  # (B, d, M)
        eps = ops.convert_to_tensor(1e-7, dtype=ops.dtype(Z))
        denom = ops.sqrt(ops.sum(ops.square(Z), axis=1, keepdims=True)) + eps
        return Z / denom

    def _clip_pair_loss(self, z_m, z_n, scale):
        """Average of CLIP losses m->n and n->m using ops-only math.

        z_m, z_n: (B, d); scale: scalar.
        """
        # m -> n
        logits_mn = scale * ops.matmul(z_m, ops.transpose(z_n))  # (B, B)
        row_lse_mn = ops.logsumexp(logits_mn, axis=1)
        # Diagonal extraction backend-specifically to avoid static int casts
        be = keras.backend.backend()
        if be == "tensorflow":
            diag_mn = tf.linalg.diag_part(logits_mn)
        elif be == "torch":
            import torch
            diag_mn = torch.diagonal(getattr(logits_mn, "value", logits_mn), 0, dim1=-2, dim2=-1)
        else:
            raise NotImplementedError(f"Backend '{be}' not supported for diag extraction.")
        loss_mn = ops.mean(row_lse_mn - diag_mn)

        # n -> m
        logits_nm = scale * ops.matmul(z_n, ops.transpose(z_m))
        row_lse_nm = ops.logsumexp(logits_nm, axis=1)
        if be == "tensorflow":
            diag_nm = tf.linalg.diag_part(logits_nm)
        elif be == "torch":
            import torch
            diag_nm = torch.diagonal(getattr(logits_nm, "value", logits_nm), 0, dim1=-2, dim2=-1)
        else:
            raise NotImplementedError(f"Backend '{be}' not supported for diag extraction.")
        loss_nm = ops.mean(row_lse_nm - diag_nm)

        return 0.5 * (loss_mn + loss_nm)

    def _clip_loss_ops(self, Z):
        """Compute averaged directed CLIP loss over all ordered pairs (m!=n)."""
        M = len(self.model_list)
        scale = ops.exp(self.logit_scale)  # 1 / tau
        total = ops.convert_to_tensor(0.0, dtype=ops.dtype(Z))
        count = 0
        for m in range(M):
            z_m = Z[:, :, m]
            for n in range(M):
                if m == n:
                    continue
                z_n = Z[:, :, n]
                total = total + self._clip_pair_loss(z_m, z_n, scale)
                count += 1
        return total / float(count)

    # Step helpers removed: CLIP trains with a single global loss

    def train_step(self, inputs):
        # Unpack (tf.data-like provides (inputs,) structure)
        inputs = inputs[0]
        be = keras.backend.backend()

        if be == "tensorflow":
            with tf.GradientTape(persistent=True) as tape:
                Z = self._encode_all(inputs, training=True)
                clip_loss = self._clip_loss_ops(Z)
                # Regularization losses from sub-models
                reg = ops.convert_to_tensor(0.0, dtype=ops.dtype(clip_loss))
                for mdl in self.model_list:
                    if mdl.losses:
                        reg = reg + tf.add_n(mdl.losses)
                loss = clip_loss + reg

            # Per-view optimizers; attach temperature update to view 0 to avoid
            # separate optimizer calls on a non-submodel variable in graph mode
            for m, mdl in enumerate(self.model_list):
                vars_m = list(mdl.trainable_variables)
                if m == 0:
                    vars_m = vars_m + [self.logit_scale]
                grads_m = tape.gradient(loss, vars_m)
                mdl.optimizer.apply_gradients(zip(grads_m, vars_m))

            del tape
            self.clip_loss_tracker.update_state(clip_loss)
            return {"clip_loss": self.clip_loss_tracker.result()}

        elif be == "torch":
            Z = self._encode_all(inputs, training=True)
            loss = self._clip_loss_ops(Z)
            # Add regularization from sub-models
            if any(mdl.losses for mdl in self.model_list):
                import torch as _torch
                reg_terms = []
                for mdl in self.model_list:
                    for l in mdl.losses:
                        if _torch.is_tensor(l):
                            reg_terms.append(l)
                        else:
                            reg_terms.append(_torch.tensor(l, dtype=getattr(loss, "dtype", None), device=getattr(getattr(loss, "device", None), "__str__", lambda: None)() or None))
                if reg_terms:
                    loss = loss + _torch.stack(reg_terms).sum()

            # Collect all variables (including temperature)
            import torch
            all_vars = []
            for mdl in self.model_list:
                all_vars.extend([getattr(v, "value", v) for v in mdl.trainable_variables])
            all_vars.append(getattr(self.logit_scale, "value", self.logit_scale))

            grads = torch.autograd.grad(loss, all_vars, retain_graph=False, create_graph=False, allow_unused=True)

            # Apply grads per view and attach the temperature update to view 0.
            idx = 0
            for m, mdl in enumerate(self.model_list):
                n = len(mdl.trainable_variables)
                grads_m = [
                    g if g is not None else torch.zeros_like(getattr(v, "value", v))
                    for g, v in zip(grads[idx:idx+n], mdl.trainable_variables)
                ]
                vars_m = list(mdl.trainable_variables)
                if m == 0:
                    g_temp = grads[-1]
                    grads_m = grads_m + [
                        g_temp if g_temp is not None else torch.zeros_like(getattr(self.logit_scale, "value", self.logit_scale))
                    ]
                    vars_m = vars_m + [self.logit_scale]
                mdl.optimizer.apply_gradients(zip(grads_m, vars_m))
                idx += n

            self.clip_loss_tracker.update_state(loss)
            return {"clip_loss": self.clip_loss_tracker.result()}

        else:
            raise NotImplementedError(f"Backend '{be}' not supported for CLIP train_step.")

    def compile(self, optimizer):
        super().compile()
        if isinstance(optimizer, list):
            for vie in range(len(self.model_list)):
                self.model_list[vie].compile(optimizer[vie])
        elif isinstance(optimizer, keras.optimizers.Optimizer):
            for vie in range(len(self.model_list)):
                self.model_list[0].compile(optimizer)
        else:
            print('Error: optimizer must either be of the keras.optimizer class, or a list of optimizers')

    @property
    def metrics(self):
        # Keras will reset these between epochs/evaluate
        return [self.clip_loss_tracker]

    def test_step(self, inputs):
        inputs = inputs[0]
        Z = self._encode_all(inputs, training=False)
        loss = self._clip_loss_ops(Z)
        self.clip_loss_tracker.update_state(loss)
        return {"clip_loss": self.clip_loss_tracker.result()}

    # DLVPM-specific metrics omitted in CLIP variant

    def get_config(self):
        base_config = super().get_config()
        serialized_model_list = [keras.utils.serialize_keras_object(model) for model in self.model_list]
        regularized_model_list = [keras.utils.serialize_keras_object(regularizer) for regularizer in self.regularizer_list]
        config = {
            "model_list": serialized_model_list,
            "regularizer_list": regularized_model_list,
            "ndims": self.ndims,
            "is_siamese": self.is_siamese,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        config['model_list'] = [keras.utils.deserialize_keras_object(model_config) for model_config in config['model_list']]
        config['run_from_config'] = True
        if 'regularizer_list' in config:
            config['regularizer_list'] = [keras.utils.deserialize_keras_object(regularizer_config) for regularizer_config in config['regularizer_list']]
        return cls(**config)

    def get_compile_config(self):
        return {
            "model_optimizers": [keras.utils.serialize_keras_object(model.optimizer) for model in self.model_list]
        }

    def compile_from_config(self, config):
        optimizer_list = [keras.utils.deserialize_keras_object(optimizer_config) for optimizer_config in config["model_optimizers"]]
        self.compile(optimizer_list)

    def build_from_config(self, config):
        return


@keras.utils.register_keras_serializable(package="deep_lvpm", name="VICReg")
class VICReg(keras.Model):
    """
    Multi-view VICReg model with path-masked view alignment.

    - Appends a Dense(ndims) projection head to each per-view encoder.
    - Uses VICReg loss: invariance (MSE between connected views), variance floor
      per view, and covariance decorrelation per view.
    - Optimizes per-view by looping over views and updating one submodel at a time,
      only against its connected neighbors defined by `Path` (like StructuralModel).

    Metrics reported mirror StructuralModel: total_loss, cross_metric (mean corr), and mse_loss (invariance term).
    """

    def __init__(
        self,
        Path,
        model_list,
        regularizer_list,
        ndims,
        var_weight: float = 25.0,
        inv_weight: float = 25.0,
        cov_weight: float = 1.0,
        gamma: float = 1.0,
        run_from_config: bool = False,
        is_siamese: bool = False,
        eps: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.Path = ops.convert_to_tensor(Path, dtype="float32")
        self.ndims = ndims
        self.var_weight = float(var_weight)
        self.inv_weight = float(inv_weight)
        self.cov_weight = float(cov_weight)
        self.gamma = float(gamma)
        self.is_siamese = is_siamese
        self.eps = float(eps)
        self.regularizer_list = regularizer_list

        if not run_from_config:
            if self.is_siamese:
                new_model = self._add_projection(model_list[0], regularizer_list[0])
                self.model_list = [new_model] * len(model_list)
            else:
                self.model_list = [
                    self._add_projection(model, regularizer)
                    for model, regularizer in zip(model_list, regularizer_list)
                ]
        else:
            self.model_list = model_list

        # Trackers similar to StructuralModel
        self.loss_tracker_total = keras.metrics.Mean(name="total_loss")
        self.corr_tracker = keras.metrics.Mean(name="cross_metric")
        self.loss_tracker_mse = keras.metrics.Mean(name="mse_loss")
        self.loss_tracker_redundancy = keras.metrics.Mean(name="redundancy")

    def _add_projection(self, model, regularizer):
        if isinstance(model, keras.Sequential):
            model.add(layers.Dense(self.ndims, activation=None, name="vicreg_projection", kernel_regularizer=regularizer))
            return model
        elif isinstance(model, keras.Model):
            x = layers.Dense(self.ndims, activation=None, name="vicreg_projection", kernel_regularizer=regularizer)(model.output)
            return keras.Model(inputs=model.input, outputs=x)
        else:
            raise ValueError("The input model must be either a keras.Sequential or a keras.Model instance.")

    def organize_inputs_by_model(self, data_inputs):
        organized_inputs = []
        data_index = 0
        for model in self.model_list:
            num_inputs = len(model.inputs) if hasattr(model, "inputs") else 1
            if num_inputs == 1:
                organized_inputs.append(data_inputs[data_index])
                data_index += 1
            else:
                inputs_for_model = data_inputs[data_index : data_index + num_inputs]
                organized_inputs.append(inputs_for_model)
                data_index += num_inputs
        return organized_inputs

    def call(self, inputs, training=False):
        inputs_nested = self.organize_inputs_by_model(inputs)
        out = ops.stack(
            [self.model_list[v](inputs_nested[v], training=training) for v in range(len(self.model_list))],
            axis=-1,
        )
        return out  # (B, d, M)

    @property
    def metrics(self):
        return [self.loss_tracker_total, self.corr_tracker, self.loss_tracker_mse, self.loss_tracker_redundancy]

    def _shape_fn(self, X):
        backend = keras.backend.backend()
        if backend == "tensorflow":
            shape = tf.shape(X)
        else:
            shape = ops.shape(X)
        return shape

    def _stop_grad(self, x):
        be = keras.backend.backend()
        if be == "tensorflow":
            return tf.stop_gradient(x)
        elif be == "torch":
            import torch
            return getattr(x, "value", x).detach()
        else:
            raise NotImplementedError(f"Backend '{be}' not supported for stop_gradient.")

    def _variance_loss(self, z):
        # z: (B, d)
        z = ops.convert_to_tensor(z)
        zc = z - ops.mean(z, axis=0)
        var = ops.mean(ops.square(zc), axis=0)
        std = ops.sqrt(var + self.eps)
        return ops.mean(ops.relu(self.gamma - std))

    def _covariance_loss(self, z):
        # z: (B, d)
        z = ops.convert_to_tensor(z)
        zc = z - ops.mean(z, axis=0)
        n = self._shape_fn(zc)[0]
        n_f = ops.cast(n, z.dtype)
        denom = ops.maximum(n_f - 1.0, 1.0)
        cov = ops.matmul(ops.transpose(zc), zc) / denom  # (d, d)
        d = self._shape_fn(cov)[0]
        eye = ops.cast(ops.eye(d), cov.dtype)
        off = cov * (ops.convert_to_tensor(1.0, dtype=ops.dtype(cov)) - eye)
        # average squared off-diagonal entries
        d_f = ops.cast(d, cov.dtype)
        num = ops.maximum(d_f * (d_f - 1.0), 1.0)
        return ops.sum(ops.square(off)) / num

    def _pair_mse(self, a, b):
        diff = a - b
        return ops.mean(ops.square(diff))

    def _corr_pair(self, a, b):
        # Mean Pearson correlation across embedding dimensions (matches StructuralModel)
        eps = ops.convert_to_tensor(self.eps, dtype=ops.dtype(a))
        a_c = a - ops.mean(a, axis=0)
        b_c = b - ops.mean(b, axis=0)
        a_n = a_c / (ops.sqrt(ops.sum(ops.square(a_c), axis=0)) + eps)
        b_n = b_c / (ops.sqrt(ops.sum(ops.square(b_c), axis=0)) + eps)
        # Sum over batch gives per-dimension Pearson r; then mean over dims
        return ops.mean(ops.sum(a_n * b_n, axis=0))

    def calculate_redundancy(self, Y, epsilon=1e-8):
        """
        Mean absolute off-diagonal correlation between dimensions of Y (N, D).
        Mirrors StructuralModel.calculate_redundancy for comparable reporting.
        """
        Y = ops.convert_to_tensor(Y)
        Y = ops.cast(Y, "float32")

        # Center columns
        col_mean = ops.mean(Y, axis=0, keepdims=True)
        Yc = Y - col_mean

        # Sample-size for covariance
        n = self._shape_fn(Yc)[0]
        n_f = ops.cast(n, Y.dtype)
        denom_n = ops.maximum(n_f - 1.0, 1.0)

        # Covariance between columns: (D x D)
        cov = ops.matmul(ops.transpose(Yc), Yc) / denom_n

        # Column std devs (D,)
        var = ops.sum(Yc * Yc, axis=0) / denom_n
        std = ops.sqrt(ops.maximum(var, epsilon))

        # Correlation matrix: cov / (std_i * std_j)
        std_col = ops.reshape(std, (-1, 1))
        denom = std_col * ops.transpose(std_col)
        corr = cov / ops.maximum(denom, epsilon)

        # Mean absolute correlation over off-diagonal entries
        corr_abs = ops.abs(corr)
        D = self._shape_fn(corr_abs)[0]
        mask = ops.ones_like(corr_abs) - ops.cast(ops.eye(D), corr_abs.dtype)
        total = ops.sum(corr_abs * mask)
        D_f = ops.cast(D, corr_abs.dtype)
        num_pairs = ops.maximum(D_f * (D_f - 1.0), 1.0)
        return total / num_pairs

    def compile(self, optimizer):
        super().compile()
        if isinstance(optimizer, list):
            for v in range(len(self.model_list)):
                self.model_list[v].compile(optimizer[v])
        elif isinstance(optimizer, keras.optimizers.Optimizer):
            for v in range(len(self.model_list)):
                self.model_list[0].compile(optimizer)
        else:
            print("Error: optimizer must either be of the keras.optimizer class, or a list of optimizers")

    def train_step(self, inputs):
        inputs = inputs[0]
        be = keras.backend.backend()
        inputs_nested = self.organize_inputs_by_model(inputs)

        total_loss = []
        total_mse = []
        total_CC = []
        total_redundancy = []

        if be == "tensorflow":
            for vie in range(len(self.model_list)):
                mdl = self.model_list[vie]
                with tf.GradientTape() as tape:
                    z_v = mdl(inputs_nested[vie], training=True)  # (B, d)

                    # Mask over connected neighbors
                    mask = ops.cast(self.Path[vie, :], ops.dtype(z_v))  # (M,)
                    inv_loss = ops.convert_to_tensor(0.0, dtype=ops.dtype(z_v))
                    cc_num = ops.convert_to_tensor(0.0, dtype=ops.dtype(z_v))
                    for n in range(len(self.model_list)):
                        z_n = self._stop_grad(self.model_list[n](inputs_nested[n], training=True))
                        w = mask[n]
                        mse_n = self._pair_mse(z_v, z_n)
                        inv_loss = inv_loss + w * mse_n
                        cc_num = cc_num + w * self._corr_pair(z_v, z_n)

                    n_conn = ops.sum(mask)
                    n_conn = ops.maximum(n_conn, ops.convert_to_tensor(1.0, dtype=ops.dtype(z_v)))
                    cc_mean = cc_num / n_conn

                    var_loss = self._variance_loss(z_v)
                    cov_loss = self._covariance_loss(z_v)
                    red = self.calculate_redundancy(z_v)

                    reg = tf.add_n(mdl.losses) if mdl.losses else ops.convert_to_tensor(0.0, dtype=ops.dtype(var_loss))
                    loss = (
                        self.inv_weight * inv_loss + self.var_weight * var_loss + self.cov_weight * cov_loss + reg
                    )

                grads = tape.gradient(loss, mdl.trainable_variables)
                mdl.optimizer.apply_gradients(zip(grads, mdl.trainable_variables))

                total_loss.append(ops.sum(loss))
                total_mse.append(ops.sum(inv_loss))
                total_CC.append(cc_mean)
                total_redundancy.append(red)

        elif be == "torch":
            import torch
            for vie in range(len(self.model_list)):
                mdl = self.model_list[vie]
                z_v = mdl(inputs_nested[vie], training=True)

                mask = ops.cast(self.Path[vie, :], ops.dtype(z_v))
                inv_loss = ops.convert_to_tensor(0.0, dtype=ops.dtype(z_v))
                cc_num = ops.convert_to_tensor(0.0, dtype=ops.dtype(z_v))
                for n in range(len(self.model_list)):
                    z_n = self._stop_grad(self.model_list[n](inputs_nested[n], training=True))
                    w = mask[n]
                    inv_loss = inv_loss + w * self._pair_mse(z_v, z_n)
                    cc_num = cc_num + w * self._corr_pair(z_v, z_n)

                n_conn = ops.sum(mask)
                n_conn = ops.maximum(n_conn, ops.convert_to_tensor(1.0, dtype=ops.dtype(z_v)))
                cc_mean = cc_num / n_conn

                var_loss = self._variance_loss(z_v)
                cov_loss = self._covariance_loss(z_v)
                red = self.calculate_redundancy(z_v)
                reg = (
                    torch.stack([
                        l if torch.is_tensor(l) else torch.tensor(float(l), dtype=getattr(z_v, "dtype", None), device=getattr(getattr(z_v, "device", None), "__str__", lambda: None)() or None)
                        for l in mdl.losses
                    ]).sum()
                    if mdl.losses
                    else ops.convert_to_tensor(0.0, dtype=ops.dtype(var_loss))
                )
                loss = self.inv_weight * inv_loss + self.var_weight * var_loss + self.cov_weight * cov_loss + reg

                vars_for_grad = [getattr(v, "value", v) for v in mdl.trainable_variables]
                grads = torch.autograd.grad(loss, vars_for_grad, retain_graph=False, create_graph=False, allow_unused=True)
                fixed_grads = [g if g is not None else torch.zeros_like(getattr(v, "value", v)) for g, v in zip(grads, mdl.trainable_variables)]
                mdl.optimizer.apply_gradients(zip(fixed_grads, mdl.trainable_variables))

                total_loss.append(ops.sum(loss))
                total_mse.append(ops.sum(inv_loss))
                total_CC.append(cc_mean)
                total_redundancy.append(red)

        else:
            raise NotImplementedError(f"Backend '{be}' not supported in VICReg train_step.")

        # Update trackers
        self.loss_tracker_total.update_state(ops.stack(total_loss))
        self.loss_tracker_mse.update_state(ops.stack(total_mse))
        self.corr_tracker.update_state(ops.stack(total_CC))
        self.loss_tracker_redundancy.update_state(ops.stack(total_redundancy))

        return {
            "total_loss": self.loss_tracker_total.result(),
            "cross_metric": self.corr_tracker.result(),
            "mse_loss": self.loss_tracker_mse.result(),
            "redundancy": self.loss_tracker_redundancy.result(),
        }

    def test_step(self, inputs):
        inputs = inputs[0]
        inputs_nested = self.organize_inputs_by_model(inputs)

        total_loss = []
        total_mse = []
        total_CC = []
        total_redundancy = []

        for vie in range(len(self.model_list)):
            mdl = self.model_list[vie]
            z_v = mdl(inputs_nested[vie], training=False)

            mask = ops.cast(self.Path[vie, :], ops.dtype(z_v))
            inv_loss = ops.convert_to_tensor(0.0, dtype=ops.dtype(z_v))
            cc_num = ops.convert_to_tensor(0.0, dtype=ops.dtype(z_v))
            for n in range(len(self.model_list)):
                z_n = self.model_list[n](inputs_nested[n], training=False)
                w = mask[n]
                inv_loss = inv_loss + w * self._pair_mse(z_v, z_n)
                cc_num = cc_num + w * self._corr_pair(z_v, z_n)

            n_conn = ops.sum(mask)
            n_conn = ops.maximum(n_conn, ops.convert_to_tensor(1.0, dtype=ops.dtype(z_v)))
            cc_mean = cc_num / n_conn

            var_loss = self._variance_loss(z_v)
            cov_loss = self._covariance_loss(z_v)
            red = self.calculate_redundancy(z_v)
            loss = self.inv_weight * inv_loss + self.var_weight * var_loss + self.cov_weight * cov_loss

            total_loss.append(ops.sum(loss))
            total_mse.append(ops.sum(inv_loss))
            total_CC.append(cc_mean)
            total_redundancy.append(red)

        self.loss_tracker_total.update_state(ops.stack(total_loss))
        self.loss_tracker_mse.update_state(ops.stack(total_mse))
        self.corr_tracker.update_state(ops.stack(total_CC))
        self.loss_tracker_redundancy.update_state(ops.stack(total_redundancy))

        return {
            "total_loss": self.loss_tracker_total.result(),
            "cross_metric": self.corr_tracker.result(),
            "mse_loss": self.loss_tracker_mse.result(),
            "redundancy": self.loss_tracker_redundancy.result(),
        }

    def get_config(self):
        base_config = super().get_config()
        serialized_model_list = [keras.utils.serialize_keras_object(model) for model in self.model_list]
        serialized_regularizers = [keras.utils.serialize_keras_object(r) for r in self.regularizer_list]
        return {
            **base_config,
            "Path": np.asarray(self.Path).tolist(),
            "model_list": serialized_model_list,
            "regularizer_list": serialized_regularizers,
            "ndims": self.ndims,
            "var_weight": self.var_weight,
            "inv_weight": self.inv_weight,
            "cov_weight": self.cov_weight,
            "gamma": self.gamma,
            "is_siamese": self.is_siamese,
            "eps": self.eps,
        }

    @classmethod
    def from_config(cls, config):
        config["model_list"] = [keras.utils.deserialize_keras_object(mc) for mc in config["model_list"]]
        if "regularizer_list" in config:
            config["regularizer_list"] = [keras.utils.deserialize_keras_object(rc) for rc in config["regularizer_list"]]
        config["run_from_config"] = True
        return cls(**config)

    def get_compile_config(self):
        return {"model_optimizers": [keras.utils.serialize_keras_object(m.optimizer) for m in self.model_list]}

    def compile_from_config(self, config):
        optimizers = [keras.utils.deserialize_keras_object(o) for o in config["model_optimizers"]]
        self.compile(optimizers)
