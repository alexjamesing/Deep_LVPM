#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FactorLayer: iterative Moore-Penrose deflation orthogonalization layer.

Algorithm
---------
``FactorLayer`` is appended to the end of each measurement model and produces
``ndims`` orthogonal deep latent variables (DLVs) from the model's output.

The DLVs are constructed by iterative deflation:

  DLV₁ = X w₁
  DLV₂ = (X − DLV₁ β₁) w₂          ← X residualized w.r.t. DLV₁
  DLVₖ = (X − [DLV₁…DLVₖ₋₁] β) wₖ  ← X residualized w.r.t. all prior DLVs

where wₖ are learned projection vectors and β is the regression coefficient of
X onto the previous DLVs (computed via the Moore-Penrose pseudoinverse).

Trainable vs. static weights
-----------------------------
Two sets of projection vectors are maintained:

  - ``linear_layer_list``  — *trainable*; receive gradients in pass 2 of
                             StructuralModel's training loop.
  - ``static_{i}`` buffers — *non-trainable copies*; used as the deflation
                             basis during pass 2 so that gradients flow only
                             through the projection step, not through the
                             orthogonalization directions.

``weight_normalizer`` (called at the end of pass 1) normalizes both sets and
copies the trainable weights into the static buffers, keeping them in sync
before each gradient update.

Train vs. test behaviour
------------------------
During training, orthogonalization uses batch statistics (mean / std of the
current mini-batch).  During inference, it uses the moving statistics
(``DLV_mean``, ``DLV_var``, ``moving_convX``) accumulated over training.
"""

import torch
import torch.nn as nn


class FactorLayer(nn.Module):
    """Orthogonalization layer implementing iterative Moore-Penrose deflation.

    Appended automatically to each view's measurement model by
    ``StructuralModel``.  Produces ``ndims`` orthogonal DLVs from the
    measurement model's output.

    Attributes
    ----------
    kernel_regularizer : tuple (l1, l2) or None
        L1/L2 penalty applied to the trainable projection weights.
    epsilon : float
        Small constant for numerical stability in batch normalisation
        and orthogonalization.
    momentum : float
        Momentum for moving statistics (mean, variance, covariance).
        Close to 1 = slow update (Keras convention).  The underlying
        ``nn.BatchNorm1d`` receives ``1 - momentum`` so that both use
        the same convention.
    tot_num : int
        Total number of training samples; used to scale the moving
        cross-covariance ``moving_convX`` to dataset scale.
    ndims : int
        Number of DLVs to extract.
    batch_norm1 : nn.BatchNorm1d
        BatchNormalization layer applied to inputs.
    linear_layer_list : nn.ParameterList
        Trainable projection vectors, one per DLV.
    static_{i} : buffer
        Non-trainable copies of the projection vectors used as the deflation
        basis during gradient updates.  Kept in sync with ``linear_layer_list``
        by ``weight_normalizer``.
    DLV_mean : buffer, shape (ndims, 1)
        Moving mean of the DLVs.
    DLV_var : buffer, shape (ndims, 1)
        Moving variance of the DLVs.
    moving_convX : buffer, shape (ndims, n_features)
        Moving cross-covariance between normalized DLVs and inputs, scaled to
        dataset size.  Dividing by ``tot_num`` in ``_orthogonalise_test`` gives
        the regression coefficient β = Cov(DLV, X) / Var(DLV).
    """

    def __init__(
        self,
        kernel_regularizer=None,
        epsilon: float = 1e-3,
        momentum: float = 0.99,
        tot_num: int = None,
        ndims: int = None,
    ):
        super().__init__()
        self.kernel_regularizer = kernel_regularizer
        self.epsilon = epsilon
        self.momentum = momentum
        self.tot_num = tot_num
        self.ndims = ndims
        self._built = False

    # ------------------------------------------------------------------
    # Lazy build — called on first forward() or explicitly
    # ------------------------------------------------------------------

    def build(self, input_dim: int):
        """Initialise all weights for the given feature dimension.

        Parameters
        ----------
        input_dim : int
            Number of features in the input tensor ``(batch, input_dim)``.
        """
        if self._built:
            return

        # PyTorch BatchNorm1d momentum convention is the inverse of Keras:
        # PyTorch momentum = 1 - Keras momentum.
        self.batch_norm1 = nn.BatchNorm1d(
            input_dim,
            momentum=1.0 - self.momentum,
            eps=self.epsilon,
        )

        # Trainable projection vectors — gradients flow through these during
        # pass 2 of the StructuralModel training loop.
        self.linear_layer_list = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim, 1)) for _ in range(self.ndims)]
        )

        # Non-trainable copies used as the deflation basis during pass 2.
        # Kept in sync with linear_layer_list by weight_normalizer so that
        # the orthogonalization directions are stable within each step.
        for i in range(self.ndims):
            self.register_buffer(f"static_{i}", torch.randn(input_dim, 1))

        # Moving statistics used at test time (independent of batch content).
        self.register_buffer("DLV_mean", torch.zeros(self.ndims, 1))
        self.register_buffer("DLV_var", torch.ones(self.ndims, 1))
        # moving_convX stores E[DLV_norm · X] scaled to dataset size.
        self.register_buffer("moving_convX", torch.zeros(self.ndims, input_dim))
        # Tracks whether this is the first forward pass (used in
        # _update_moving_variables to set momentum = 0 on the first call,
        # which initializes the moving statistics from scratch rather than
        # blending with the zero-initialized values).
        self.register_buffer("_initialized", torch.tensor(False))

        self._built = True

    # ------------------------------------------------------------------
    # Helpers: static buffer access
    # ------------------------------------------------------------------

    def _static(self, i: int) -> torch.Tensor:
        return getattr(self, f"static_{i}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute orthogonal DLVs from the measurement model's output.

        During training: deflation uses static projection vectors as the
        orthogonalization basis (so gradients only flow through the trainable
        projection step), and moving statistics are updated.

        During inference: deflation uses moving statistics accumulated during
        training so that the computation is independent of batch content.

        Parameters
        ----------
        inputs : Tensor, shape (batch, n_features)

        Returns
        -------
        Tensor, shape (batch, ndims)
        """
        if not self._built:
            self.build(inputs.shape[1])
            self.to(inputs.device)

        X = self.batch_norm1(inputs)

        if self.training:
            # Pass 1 target: DLVs computed with static (non-gradient) weights.
            DLV_all = self._calculate_batch_DLV_static(X)
            # Pass 2 gradient path: same deflation basis, trainable projection.
            out = self._calculate_batch_DLV_train(X, DLV_all)
            with torch.no_grad():
                self._update_moving_variables(X, DLV_all)
        else:
            out = self._calculate_DLV_test(X)

        return out

    # ------------------------------------------------------------------
    # Weight normaliser (called from StructuralModel, already no_grad)
    # ------------------------------------------------------------------

    def weight_normalizer(
        self,
        y: torch.Tensor,
        scale_fact: torch.Tensor,
        train_DLV: bool,
    ) -> torch.Tensor:
        """Normalize projection vectors to unit L2 norm and copy to static.

        Called at the end of pass 1 by ``StructuralModel._weight_normaliser``.
        After normalization, static weights are updated so that the
        orthogonalization basis in pass 2 uses the freshly normalized vectors.

        Parameters
        ----------
        y : Tensor, shape (batch, ndims)
            Current batch's DLVs (from static weights, pass 1).
        scale_fact : Tensor
            ``tot_num / batch_size``; rescales the per-batch L2 norm to a
            dataset-scale estimate.
        train_DLV : bool
            Unused here; kept for interface consistency with ZCALayer.

        Returns
        -------
        Tensor, shape (batch, ndims)
            Normalized DLVs.
        """
        with torch.no_grad():
            for i in range(self.ndims):
                yi = y[:, i]
                denom = torch.sqrt(scale_fact * (yi**2).sum())
                # Normalize trainable weight and copy the same value into the
                # static buffer so both sets stay in sync.
                new_w = self.linear_layer_list[i] / denom
                self.linear_layer_list[i].data.copy_(new_w)
                self._static(i).copy_(new_w)

            y_denom = torch.sqrt(scale_fact * (y**2).sum(dim=0))
            out_y = y / y_denom
        return out_y

    # ------------------------------------------------------------------
    # Moving variable update
    # ------------------------------------------------------------------

    def _update_moving_variables(self, X: torch.Tensor, DLV_all: torch.Tensor) -> None:
        """Update moving mean, variance, and cross-covariance from the batch.

        On the very first call (``_initialized == False``), momentum is forced
        to 0 so that the moving statistics are initialized directly from the
        first batch rather than blended with the zero-initialized values.

        Parameters
        ----------
        X : Tensor, shape (batch, n_features)
            Batch-normalized input.
        DLV_all : Tensor, shape (batch, ndims)
            DLVs computed with static weights.
        """
        batch_size = X.shape[0]
        scale_fact = float(self.tot_num) / float(batch_size)

        # Bootstrap: use momentum=0 on the first call to avoid blending with
        # the zero-initialized moving statistics.
        m = 0.0 if not self._initialized else self.momentum
        one_m = 1.0 - m

        batch_DLV_mean = DLV_all.mean(dim=0).unsqueeze(1)  # (ndims, 1)
        batch_DLV_var = DLV_all.var(dim=0).unsqueeze(1)  # (ndims, 1)

        self.DLV_mean.copy_(m * self.DLV_mean + one_m * batch_DLV_mean)
        self.DLV_var.copy_(m * self.DLV_var + one_m * batch_DLV_var)

        # Normalize DLVs before computing cross-covariance so that
        # moving_convX stores the regression coefficient (not raw covariance).
        batch_DLV_norm = (DLV_all - batch_DLV_mean.T) / (
            batch_DLV_var.sqrt().T + self.epsilon
        )

        # Scale by scale_fact so dividing by tot_num in _orthogonalise_test
        # recovers the per-sample regression coefficient β.
        self.moving_convX.copy_(
            m * self.moving_convX + scale_fact * one_m * (batch_DLV_norm.T @ X)
        )

        self._initialized.fill_(True)

    # ------------------------------------------------------------------
    # Orthogonalisation helpers
    # ------------------------------------------------------------------

    def _orthogonalise_train(
        self, X: torch.Tensor, DLV_prev: torch.Tensor
    ) -> torch.Tensor:
        """Residualize X w.r.t. previous DLVs using batch statistics.

        Computes X − DLV_prev β where β = (DLV_prev^T DLV_prev)⁻¹ DLV_prev^T X
        estimated from the current mini-batch.

        Parameters
        ----------
        X : Tensor, shape (batch, n_features)
        DLV_prev : Tensor, shape (batch, k)
            Previously computed DLVs to deflate from X.

        Returns
        -------
        Tensor, shape (batch, n_features)
            Residualized X.
        """
        # Standardize DLV_prev so the regression coefficient is scale-invariant.
        DLV_batch = (DLV_prev - DLV_prev.mean(dim=0)) / (
            DLV_prev.std(dim=0) + self.epsilon
        )
        denom = float(X.shape[0])
        beta = (DLV_batch.T @ X) / denom
        return X - DLV_batch @ beta

    def _orthogonalise_test(
        self, X: torch.Tensor, DLV_prev: torch.Tensor
    ) -> torch.Tensor:
        """Residualize X w.r.t. previous DLVs using moving statistics.

        Uses the accumulated ``moving_convX`` and ``DLV_mean`` / ``DLV_var``
        instead of the current batch, making inference independent of batch
        content.

        Parameters
        ----------
        X : Tensor, shape (batch, n_features)
        DLV_prev : Tensor, shape (batch, k)
            DLVs computed so far at test time.

        Returns
        -------
        Tensor, shape (batch, n_features)
            Residualized X.
        """
        i = DLV_prev.shape[1]  # number of prior DLVs to deflate

        DLV_norm = (DLV_prev - self.DLV_mean[:i, :].T) / (
            self.DLV_var[:i, :].sqrt().T + self.epsilon
        )
        # Dividing moving_convX by tot_num recovers the regression coefficient
        # β = Cov(DLV_norm, X) / N  (moving_convX was accumulated scaled by N).
        beta = self.moving_convX[:i, :] / float(self.tot_num)
        return X - DLV_norm @ beta

    # ------------------------------------------------------------------
    # DLV calculation methods
    # ------------------------------------------------------------------

    def _calculate_batch_DLV_static(self, X: torch.Tensor) -> torch.Tensor:
        """Compute all DLVs using the non-trainable (static) projection vectors.

        Used during training to generate the deflation basis for
        ``_calculate_batch_DLV_train``.  Because these weights are
        non-trainable, no gradient flows through this computation.

        Parameters
        ----------
        X : Tensor, shape (batch, n_features)
            Batch-normalized input.

        Returns
        -------
        Tensor, shape (batch, ndims)
        """
        DLV_all = X @ self._static(0)

        for i in range(1, self.ndims):
            # Deflate X w.r.t. all previously computed static DLVs.
            ortho = self._orthogonalise_train(X, DLV_all)
            DLV = ortho @ self._static(i)
            DLV_all = torch.cat([DLV_all, DLV], dim=1)

        return DLV_all

    def _calculate_batch_DLV_train(
        self, X: torch.Tensor, DLV_all: torch.Tensor
    ) -> torch.Tensor:
        """Compute all DLVs using the *trainable* projection vectors.

        Uses ``DLV_all`` (from static weights) as the deflation basis so that
        gradients flow only through the trainable linear projection step, not
        through the orthogonalization directions.

        Parameters
        ----------
        X : Tensor, shape (batch, n_features)
            Batch-normalized input.
        DLV_all : Tensor, shape (batch, ndims)
            DLVs from ``_calculate_batch_DLV_static``, used as the
            deflation basis.

        Returns
        -------
        Tensor, shape (batch, ndims)
        """
        out = X @ self.linear_layer_list[0]

        for i in range(1, self.ndims):
            # Deflate using the static DLVs (gradient does not flow through this).
            ortho = self._orthogonalise_train(X, DLV_all[:, :i])
            out_i = ortho @ self.linear_layer_list[i]
            out = torch.cat([out, out_i], dim=1)

        return out

    def _calculate_DLV_test(self, X: torch.Tensor) -> torch.Tensor:
        """Compute DLVs at test time using accumulated moving statistics.

        Parameters
        ----------
        X : Tensor, shape (batch, n_features)
            Batch-normalized input.

        Returns
        -------
        Tensor, shape (batch, ndims)
        """
        out = X @ self.linear_layer_list[0]

        for i in range(1, self.ndims):
            # Deflate using moving statistics accumulated during training.
            ortho = self._orthogonalise_test(X, out)
            out_i = ortho @ self.linear_layer_list[i]
            out = torch.cat([out, out_i], dim=1)

        return out

    # ------------------------------------------------------------------
    # Regularisation loss
    # ------------------------------------------------------------------

    def regularization_loss(self) -> torch.Tensor:
        """L1/L2 penalty on trainable projection weights, or zero."""
        device = self.linear_layer_list[0].device
        dtype = self.linear_layer_list[0].dtype
        penalty = torch.zeros((), device=device, dtype=dtype)

        if self.kernel_regularizer is None:
            return penalty

        l1, l2 = self.kernel_regularizer
        for w in self.linear_layer_list:
            if l1 > 0:
                penalty = penalty + l1 * w.abs().sum()
            if l2 > 0:
                penalty = penalty + l2 * (w**2).sum()
        return penalty
