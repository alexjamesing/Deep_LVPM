#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StructuralModel — pure PyTorch implementation.

A custom nn.Module that wraps a collection of per-view encoder
networks and coordinates joint training to find orthogonal Deep Latent
Variables (DLVs) that capture shared structure across heterogeneous
data modalities.

The association structure between views is defined by a binary
adjacency matrix (``Path``).
"""

from __future__ import annotations

import numpy as np
import pydot
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from deep_lvpm.layers.FactorLayer import FactorLayer
from deep_lvpm.layers.ZCALayer import ZCALayer


class StructuralModel(nn.Module):
    """
    Multi-view Deep Latent Variable Path Model.

    Parameters
    ----------
    Path : array-like, shape (n_views, n_views)
        Binary adjacency matrix.  ``Path[i, j] == 1`` means view i and
        view j are associated.
    model_list : list of nn.Module
        One encoder per data view.  Each will be wrapped with a
        FactorLayer or ZCALayer automatically.
    regularizer_list : list of tuple or None
        Per-view regularisation as ``(l1, l2)`` tuples or ``None``.
    tot_num : int
        Total training-set size (used for covariance scaling).
    ndims : int
        Number of orthogonal DLVs to extract per view.
    orthogonalization : str
        ``'Moore-Penrose'`` (default) or ``'zca'``.
    momentum : float
        EMA momentum for moving statistics (Keras convention).
    epsilon : float
        Numerical stability constant.
    train_DLV : bool
        If True (default), DLV targets during training are computed
        from the current batch.  If False, moving-average statistics
        are used instead.
    is_siamese : bool
        If True, all views share the same encoder (weights tied).
    diag_offset : float
        Diagonal regularisation for ZCA covariance inversion.
    device : str or None
        ``'cpu'``, ``'cuda'``, etc.  Auto-detected when None.
    """

    def __init__(
        self,
        Path,
        model_list: list,
        regularizer_list: list,
        tot_num: int,
        ndims: int,
        orthogonalization: str = "Moore-Penrose",
        momentum: float = 0.95,
        epsilon: float = 1e-4,
        train_DLV: bool = True,
        is_siamese: bool = False,
        diag_offset: float = 1e-3,
        device: str | None = None,
    ):
        super().__init__()

        self.tot_num = tot_num
        self.ndims = ndims
        self.momentum = momentum
        self.epsilon = epsilon
        self.orthogonalization = orthogonalization
        self.regularizer_list = regularizer_list
        self.train_DLV = train_DLV
        self.is_siamese = is_siamese
        self.diag_offset = diag_offset

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Path stored as a buffer so it moves with the model
        path_tensor = torch.tensor(np.asarray(Path), dtype=torch.float32)
        self.register_buffer("Path", path_tensor)

        # Build wrapped models
        if is_siamese:
            wrapped = self._add_dlvpm_layer(model_list[0], regularizer_list[0])
            wrapped_list = [wrapped] * len(model_list)
            # nn.ModuleList with the same object repeated: PyTorch registers
            # the module once, but we need independent list entries for
            # index access, so we store the unique module separately.
            self._siamese_module = wrapped
            self.model_list = nn.ModuleList(wrapped_list)
        else:
            wrapped_list = [
                self._add_dlvpm_layer(m, r)
                for m, r in zip(model_list, regularizer_list)
            ]
            self.model_list = nn.ModuleList(wrapped_list)

        self.optimizers: list | None = None

    # ------------------------------------------------------------------
    # Layer wrapping
    # ------------------------------------------------------------------

    def _add_dlvpm_layer(self, model: nn.Module, regularizer) -> nn.Sequential:
        """Append a FactorLayer or ZCALayer to the given encoder."""
        if self.orthogonalization == "Moore-Penrose":
            factor = FactorLayer(
                kernel_regularizer=regularizer,
                tot_num=self.tot_num,
                ndims=self.ndims,
                momentum=self.momentum,
                epsilon=self.epsilon,
            )
        elif self.orthogonalization == "zca":
            factor = ZCALayer(
                kernel_regularizer=regularizer,
                tot_num=self.tot_num,
                ndims=self.ndims,
                momentum=self.momentum,
                epsilon=self.epsilon,
                diag_offset=self.diag_offset,
            )
        else:
            raise ValueError(
                f"Unknown orthogonalization '{self.orthogonalization}'. "
                "Must be 'Moore-Penrose' or 'zca'."
            )

        wrapped = nn.Sequential(model, factor)
        # Carry forward the n_inputs attribute if present
        wrapped.n_inputs = getattr(model, "n_inputs", 1)
        return wrapped

    # ------------------------------------------------------------------
    # Input organisation
    # ------------------------------------------------------------------

    def organize_inputs_by_model(self, data_inputs: list) -> list:
        """Distribute a flat input list to per-view sub-lists."""
        organized = []
        idx = 0
        for model in self.model_list:
            n = getattr(model, "n_inputs", 1)
            if n == 1:
                organized.append(data_inputs[idx])
                idx += 1
            else:
                organized.append(data_inputs[idx : idx + n])
                idx += n
        return organized

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs: list) -> torch.Tensor:
        """Return stacked DLVs: (batch, ndims, n_views)."""
        inputs_nested = self.organize_inputs_by_model(inputs)
        outputs = []
        for v in range(len(self.model_list)):
            outputs.append(self.model_list[v](inputs_nested[v]))
        return torch.stack(outputs, dim=2)

    def build(self, X_list: list) -> None:
        """
        Trigger lazy initialisation of FactorLayer / ZCALayer weights.

        Call this with a small sample of training data before creating
        per-view optimizers, so that ``model.parameters()`` is non-empty.

        Parameters
        ----------
        X_list : list of array-like
            One array per input tensor (same format as passed to ``fit``).
        """
        tensors = [
            torch.as_tensor(np.asarray(x)[:2], dtype=torch.float32).to(self.device)
            for x in X_list
        ]
        self.to(self.device)
        with torch.no_grad():
            self.forward(tensors)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def compile(self, optimizer):
        """
        Store per-view optimizers.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer or list of torch.optim.Optimizer
            When a single optimizer is given it is shared across all
            views (intended for siamese networks).  When a list is
            given each view gets its own optimizer.
        """
        if isinstance(optimizer, list):
            self.optimizers = optimizer
        elif isinstance(optimizer, torch.optim.Optimizer):
            self.optimizers = [optimizer] * len(self.model_list)
        else:
            raise ValueError(
                "optimizer must be a torch.optim.Optimizer or a list thereof."
            )

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _normalize_pred(
        self, y_pred: torch.Tensor, scale_fact: torch.Tensor
    ) -> torch.Tensor:
        eps = torch.tensor(self.epsilon, dtype=y_pred.dtype, device=y_pred.device)
        denom = torch.sqrt(scale_fact) * torch.sqrt((y_pred**2).sum(dim=0) + eps)
        return y_pred / denom

    def _weight_normaliser(self, inputs: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Phase-1 forward pass: normalise projection weights.
        Must be called inside ``torch.no_grad()``.
        """
        if self.train_DLV:
            self.train()
        else:
            self.eval()

        y = self.forward(inputs)

        y_dtype = y.dtype
        scale_fact = torch.tensor(
            self.tot_num, dtype=y_dtype, device=y.device
        ) / torch.tensor(y.shape[0], dtype=y_dtype, device=y.device)

        y_list = []
        for v in range(len(self.model_list)):
            y_view = y[:, :, v]
            factor_layer = self.model_list[v][-1]  # last element of Sequential
            y_view = factor_layer.weight_normalizer(y_view, scale_fact, self.train_DLV)
            y_list.append(y_view)

        y = torch.stack(y_list, dim=2)
        return y, scale_fact

    def _step(
        self,
        vie: int,
        inputs_v,
        y: torch.Tensor,
        scale_fact: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-view forward + backward pass."""
        opt = self.optimizers[vie]
        model = self.model_list[vie]

        opt.zero_grad()

        model.train()
        y_pred = model(inputs_v)
        y_pred = self._normalize_pred(y_pred, scale_fact)
        mse = self.mse_loss(y, y_pred, vie)

        reg = model[-1].regularization_loss()
        loss = mse + reg

        loss.backward()
        opt.step()

        with torch.no_grad():
            corr = self.corr_metric(y, y_pred, vie)

        return loss.detach(), mse.detach(), corr.detach()

    # ------------------------------------------------------------------
    # fit / evaluate / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X_list: list,
        batch_size: int = 32,
        epochs: int = 10,
        verbose: bool | int = True,
        validation_data=None,
    ) -> dict:
        """
        Train the model.

        Parameters
        ----------
        X_list : list of array-like
            One array per data view (numpy or tensor).
        batch_size : int
        epochs : int
        verbose : bool or int
            0 / False = silent; 1 / True = one line per epoch.
        validation_data : list or None
            If provided, evaluate on this data after each epoch and
            include val metrics in the returned history.

        Returns
        -------
        dict with keys ``total_loss``, ``cross_metric``, ``mse_loss``,
        ``redundancy`` (and ``val_*`` counterparts when validation_data
        is given), each a list of per-epoch mean values.
        """
        if self.optimizers is None:
            raise RuntimeError("Call compile(optimizer) before fit().")

        self.to(self.device)

        tensors = [torch.as_tensor(x, dtype=torch.float32) for x in X_list]
        dataset = TensorDataset(*tensors)
        # drop_last=True prevents single-sample final batches that break BatchNorm
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        # Trigger lazy build of FactorLayer/ZCALayer before the training loop.
        # Optimizers created before fit() will have empty param groups if the
        # layers hadn't been built yet; rebuild those optimizers now.
        with torch.no_grad():
            sample = [t.to(self.device) for t in next(iter(loader))]
            self.forward(sample)
        for i, opt in enumerate(self.optimizers):
            if all(len(pg["params"]) == 0 for pg in opt.param_groups):
                self.optimizers[i] = type(opt)(
                    self.model_list[i].parameters(), **opt.defaults
                )

        history: dict = {
            "total_loss": [],
            "cross_metric": [],
            "mse_loss": [],
            "redundancy": [],
        }

        for epoch in range(epochs):
            self.train()
            sums = dict(loss=0.0, corr=0.0, mse=0.0, red=0.0)
            n_batches = 0

            for batch_tensors in loader:
                inputs = [t.to(self.device) for t in batch_tensors]

                # Phase 1: weight normalisation (no gradient tracking)
                with torch.no_grad():
                    y, scale_fact = self._weight_normaliser(inputs)

                # Phase 2: per-view gradient updates
                inputs_nested = self.organize_inputs_by_model(inputs)
                view_losses, view_corrs, view_mses = [], [], []

                for v in range(len(self.model_list)):
                    loss, mse, corr = self._step(v, inputs_nested[v], y, scale_fact)
                    view_losses.append(loss.item())
                    view_mses.append(mse.item())
                    view_corrs.append(corr.item())

                with torch.no_grad():
                    red = float(
                        torch.stack(
                            [
                                self.calculate_redundancy(y[:, :, v])
                                for v in range(len(self.model_list))
                            ]
                        )
                        .mean()
                        .item()
                    )

                sums["loss"] += float(np.mean(view_losses))
                sums["corr"] += float(np.mean(view_corrs))
                sums["mse"] += float(np.mean(view_mses))
                sums["red"] += red
                n_batches += 1

            n = max(n_batches, 1)
            epoch_metrics = {k: v / n for k, v in sums.items()}

            history["total_loss"].append(epoch_metrics["loss"])
            history["cross_metric"].append(epoch_metrics["corr"])
            history["mse_loss"].append(epoch_metrics["mse"])
            history["redundancy"].append(epoch_metrics["red"])

            if validation_data is not None:
                val_metrics = self.evaluate(validation_data, verbose=False)
                for k, v in val_metrics.items():
                    history.setdefault(f"val_{k}", []).append(v)

            if verbose:
                msg = (
                    f"Epoch {epoch + 1}/{epochs} — "
                    f"loss: {epoch_metrics['loss']:.4f}  "
                    f"corr: {epoch_metrics['corr']:.4f}  "
                    f"red: {epoch_metrics['red']:.4f}"
                )
                if validation_data is not None:
                    val_corr = history.get("val_cross_metric", [None])[-1]
                    if val_corr is not None:
                        msg += f"  val_corr: {val_corr:.4f}"
                print(msg)

        return history

    def evaluate(
        self,
        X_list: list,
        batch_size: int = 256,
        verbose: bool | int = True,
    ) -> dict:
        """
        Evaluate the model on a dataset.

        Returns
        -------
        dict with keys ``total_loss``, ``cross_metric``, ``mse_loss``,
        ``redundancy``.
        """
        self.to(self.device)
        self.eval()

        tensors = [torch.as_tensor(x, dtype=torch.float32) for x in X_list]
        dataset = TensorDataset(*tensors)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        sums = dict(loss=0.0, corr=0.0, mse=0.0, red=0.0)
        n_batches = 0

        with torch.no_grad():
            for batch_tensors in loader:
                inputs = [t.to(self.device) for t in batch_tensors]

                self.eval()
                y = self.forward(inputs)

                inputs_nested = self.organize_inputs_by_model(inputs)
                view_losses, view_corrs, view_mses = [], [], []

                for v in range(len(self.model_list)):
                    y_pred = self.model_list[v](inputs_nested[v])
                    mse = self.mse_loss(y, y_pred, v)
                    reg = self.model_list[v][-1].regularization_loss()
                    loss = (mse + reg).item()
                    corr = self.corr_metric(y, y_pred, v).item()
                    view_losses.append(loss)
                    view_corrs.append(corr)
                    view_mses.append(mse.item())

                red = float(
                    torch.stack(
                        [
                            self.calculate_redundancy(y[:, :, v])
                            for v in range(len(self.model_list))
                        ]
                    )
                    .mean()
                    .item()
                )

                sums["loss"] += float(np.mean(view_losses))
                sums["corr"] += float(np.mean(view_corrs))
                sums["mse"] += float(np.mean(view_mses))
                sums["red"] += red
                n_batches += 1

        n = max(n_batches, 1)
        metrics = {
            "total_loss": sums["loss"] / n,
            "cross_metric": sums["corr"] / n,
            "mse_loss": sums["mse"] / n,
            "redundancy": sums["red"] / n,
        }

        if verbose:
            print(
                f"Eval — loss: {metrics['total_loss']:.4f}  "
                f"corr: {metrics['cross_metric']:.4f}  "
                f"red: {metrics['redundancy']:.4f}"
            )

        return metrics

    def predict(
        self,
        X_list: list,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Run inference and return DLVs as a numpy array.

        Returns
        -------
        np.ndarray, shape (n_samples, ndims, n_views)
        """
        self.to(self.device)
        self.eval()

        tensors = [torch.as_tensor(x, dtype=torch.float32) for x in X_list]
        dataset = TensorDataset(*tensors)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        chunks = []
        with torch.no_grad():
            for batch_tensors in loader:
                inputs = [t.to(self.device) for t in batch_tensors]
                out = self.forward(inputs)
                chunks.append(out.cpu())

        return torch.cat(chunks, dim=0).numpy()

    # ------------------------------------------------------------------
    # Loss / metric functions
    # ------------------------------------------------------------------

    def mse_loss(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        vie: int,
    ) -> torch.Tensor:
        """
        MSE between y_pred (view vie) and connected views in y_true.

        y_true : (batch, ndims, n_views)
        y_pred : (batch, ndims)
        """
        y_pred_exp = y_pred.unsqueeze(2)  # (batch, ndims, 1)
        se_mean = ((y_true - y_pred_exp) ** 2).mean(dim=0)  # (ndims, n_views)

        mask = self.Path[vie, :].to(se_mean.dtype)  # (n_views,)
        se_mean_masked = se_mean * mask.unsqueeze(0)  # (ndims, n_views)

        return se_mean_masked.sum() / 2.0

    def corr_metric(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        vie: int,
    ) -> torch.Tensor:
        """Mean correlation between y_pred (view vie) and connected views."""
        eps = torch.tensor(self.epsilon, dtype=y_true.dtype, device=y_true.device)

        y_true_c = y_true - y_true.mean(dim=0)
        y_pred_c = y_pred - y_pred.mean(dim=0)

        denom_true = ((y_true_c**2).sum(dim=0) + eps).sqrt()
        denom_pred = ((y_pred_c**2).sum(dim=0) + eps).sqrt()

        y_true_n = y_true_c / denom_true
        y_pred_n = y_pred_c / denom_pred

        y_pred_n_exp = y_pred_n.unsqueeze(2)  # (batch, ndims, 1)
        corr_mat = (y_true_n * y_pred_n_exp).sum(dim=0)  # (ndims, n_views)

        mask = self.Path[vie, :].to(corr_mat.dtype)
        corr_masked = corr_mat * mask.unsqueeze(0)

        n_conn = mask.sum()
        n_conn_safe = torch.clamp(n_conn, min=1.0)
        return corr_masked.sum() / (n_conn_safe * float(self.ndims))

    def calculate_redundancy(
        self, Y: torch.Tensor, epsilon: float = 1e-8
    ) -> torch.Tensor:
        """Mean |corr(i, j)| over all off-diagonal pairs in Y columns."""
        Y = Y.float()
        col_mean = Y.mean(dim=0, keepdim=True)
        Yc = Y - col_mean

        n_f = float(Yc.shape[0])
        denom_n = max(n_f - 1.0, 1.0)

        cov = (Yc.T @ Yc) / denom_n
        var = (Yc * Yc).sum(dim=0) / denom_n
        std = (var.clamp(min=epsilon)).sqrt()

        std_col = std.unsqueeze(1)
        denom = (std_col @ std_col.T).clamp(min=epsilon)
        corr = cov / denom

        corr_abs = corr.abs()
        D = corr_abs.shape[0]
        mask = torch.ones_like(corr_abs) - torch.eye(
            D, device=corr_abs.device, dtype=corr_abs.dtype
        )
        total = (corr_abs * mask).sum()
        num_pairs = max(float(D) * (float(D) - 1.0), 1.0)
        return total / num_pairs

    def calculate_corrmat(self, DLVs) -> list:
        """
        Pearson correlation matrices for a 3-D tensor.

        Parameters
        ----------
        DLVs : array-like, shape (n_samples, ndims, n_views)

        Returns
        -------
        list of (n_views × n_views) tensors, one per DLV dimension.
        """
        if not isinstance(DLVs, torch.Tensor):
            DLVs = torch.tensor(DLVs, dtype=torch.float32)

        if DLVs.ndim != 3:
            raise ValueError("Input must be a 3-D tensor (n_samples, ndims, n_views).")

        n_samples = float(DLVs.shape[0])
        eps = 1e-7
        n_dims = DLVs.shape[1]

        corr_matrices = []
        for dim in range(n_dims):
            x = DLVs[:, dim, :]  # (n_samples, n_views)
            x_centered = x - x.mean(dim=0)
            std = x.std(dim=0) + eps
            normalized = x_centered / std
            corr = (normalized.T @ normalized) / n_samples
            corr_matrices.append(corr)

        return corr_matrices

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and config to a file."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "Path": self.Path.cpu().numpy(),
                    "tot_num": self.tot_num,
                    "ndims": self.ndims,
                    "orthogonalization": self.orthogonalization,
                    "momentum": self.momentum,
                    "epsilon": self.epsilon,
                    "train_DLV": self.train_DLV,
                    "is_siamese": self.is_siamese,
                    "diag_offset": self.diag_offset,
                },
            },
            path,
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_structural_model(self, outputname: str) -> None:
        """Render the path model as a directed graph (PNG)."""
        graph = pydot.Dot(graph_type="digraph", rankdir="TB")

        layer_counts = [len(list(m.modules())) for m in self.model_list]
        for i in range(len(self.model_list)):
            label = f"Measurement Model {i}, {layer_counts[i]} modules"
            node = pydot.Node(str(i), label=label, shape="record")
            graph.add_node(node)

        adj = self.Path.cpu().numpy()
        for i, row in enumerate(adj):
            for j, val in enumerate(row):
                if val == 1:
                    graph.add_edge(pydot.Edge(str(i), str(j)))

        graph.write_png(outputname)
