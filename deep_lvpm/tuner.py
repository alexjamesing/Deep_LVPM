#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Custom coordinate-descent tuner built on top of KerasTuner."""

from __future__ import annotations

import copy
import inspect
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import keras
import keras_tuner as kt
from keras.utils import Progbar

from deep_lvpm.model import StructuralModel


def _ensure_per_view_config(config: Optional[Iterable[Dict[str, Any]]], n_views: int) -> List[Dict[str, Any]]:
    """Broadcasts a single config dict (or None) across all views."""

    if config is None:
        return [dict() for _ in range(n_views)]

    if isinstance(config, (list, tuple)):
        if len(config) != n_views:
            raise ValueError(f"Config must have length {n_views}, got {len(config)}")
        return [dict(cfg) for cfg in config]

    return [dict(config) for _ in range(n_views)]


def _sample_sparse_value(
    hp: kt.HyperParameters,
    cfg: Dict[str, Any],
    view_index: int,
) -> float:
    name = f"sparse_l1_view{view_index}"
    if "values" in cfg:
        values = cfg["values"]
        if not values:
            raise ValueError("Sparse config 'values' list must not be empty.")
        default = cfg.get("default", values[0])
        return float(hp.Choice(name, values=values, default=default))

    minimum = cfg.get("min", 0.0)
    maximum = cfg.get("max", 1e-4)
    sampling = cfg.get("sampling", "log")
    default = cfg.get("default", minimum)
    return float(
        hp.Float(name, min_value=minimum, max_value=maximum, sampling=sampling, default=default)
    )


def _sample_regularizer(
    hp: kt.HyperParameters,
    cfg: Dict[str, Any],
    view_index: int,
) -> Optional[keras.regularizers.Regularizer]:
    choices = cfg.get("choices", ["none", "l2", "l1l2"])
    if not choices:
        raise ValueError("Regulariser 'choices' list must not be empty.")

    choice = hp.Choice(f"regularizer_view{view_index}", values=choices, default=choices[0])

    if choice == "none":
        return None

    def _sample_range(range_cfg: Dict[str, Any], suffix: str) -> float:
        min_value = range_cfg.get("min", 1e-6)
        max_value = range_cfg.get("max", 1e-2)
        sampling = range_cfg.get("sampling", "log")
        default = range_cfg.get("default", min_value)
        return float(
            hp.Float(
                f"regularizer_view{view_index}_{suffix}",
                min_value=min_value,
                max_value=max_value,
                sampling=sampling,
                default=default,
            )
        )

    if choice == "l2":
        lam = _sample_range(cfg.get("l2_range", {}), "l2")
        return keras.regularizers.l2(l=lam)

    if choice == "l1":
        lam = _sample_range(cfg.get("l1_range", {}), "l1")
        return keras.regularizers.l1(l=lam)

    # default to l1_l2
    l1 = _sample_range(cfg.get("l1_range", {}), "l1")
    l2 = _sample_range(cfg.get("l2_range", {}), "l2")
    return keras.regularizers.l1_l2(l1=l1, l2=l2)


def _sample_order_loss_weight(
    hp: kt.HyperParameters,
    cfg: Dict[str, Any],
) -> float:
    name = "order_loss_weight"
    if "values" in cfg:
        values = cfg["values"]
        if not values:
            raise ValueError("Order-loss config 'values' list must not be empty.")
        default = cfg.get("default", values[0])
        return float(hp.Choice(name, values=values, default=default))

    minimum = cfg.get("min", 1e-4)
    maximum = cfg.get("max", 1.0)
    sampling = cfg.get("sampling", "log")
    default = cfg.get("default", minimum)
    return float(
        hp.Float(name, min_value=minimum, max_value=maximum, sampling=sampling, default=default)
    )


def sample_structural_hparams(
    hp: kt.HyperParameters,
    n_views: int,
    target_view: int,
    current_sparse: Sequence[float],
    current_regularizers: Sequence[Optional[keras.regularizers.Regularizer]],
    current_order_loss_weight: float = 1.0,
    sparse_config: Optional[Iterable[Dict[str, Any]]] = None,
    regularizer_config: Optional[Iterable[Dict[str, Any]]] = None,
    order_loss_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Samples per-view sparsity/regulariser values and folds them into lists.

    Only the entry for ``target_view`` is resampled; the remaining entries use the
    lists provided in ``current_sparse`` and ``current_regularizers``. When
    ``order_loss_config`` is provided, a global ``order_loss_weight`` is also
    sampled for the current structural-model trial.
    """

    sparse_cfgs = _ensure_per_view_config(sparse_config, n_views)
    regularizer_cfgs = _ensure_per_view_config(regularizer_config, n_views)

    if not (0 <= target_view < n_views):
        raise ValueError(f"target_view must be in [0, {n_views})")

    sparse_list = list(current_sparse)
    reg_list = list(current_regularizers)

    sparse_list[target_view] = _sample_sparse_value(hp, sparse_cfgs[target_view], target_view)
    reg_list[target_view] = _sample_regularizer(hp, regularizer_cfgs[target_view], target_view)
    order_loss_weight = (
        _sample_order_loss_weight(hp, order_loss_config)
        if order_loss_config is not None
        else float(current_order_loss_weight)
    )

    return {
        "sparse_l1_list": sparse_list,
        "regularizer_list": reg_list,
        "order_loss_weight": order_loss_weight,
    }


class Tuner:
    """Coordinate-descent tuner that iterates over measurement models.

    Parameters
    ----------
    view_builders : Sequence[Callable[[kt.HyperParameters, int], keras.Model]]
        One callable per data-view. Each callable receives a ``HyperParameters``
        object plus the view index and must return a compiled measurement model
        (without the DLVPM projection head).
    structural_kwargs : dict
        Keyword arguments forwarded to :class:`~deep_lvpm.model.StructuralModel`
        when instantiating candidates. ``model_list`` and regularisation lists
        are supplied internally by the tuner.
    n_loops : int, optional
        Number of global coordinate-descent sweeps.
    max_trials_per_view : int, optional
        Number of random hyperparameter samples evaluated per view in each loop.
    sparse_config, regularizer_config : iterable of dict or dict, optional
        User-defined hyperparameter ranges per view. See
        :func:`sample_structural_hparams` for the accepted keys.
    order_loss_config : dict, optional
        Global sampling configuration for the learnable ordering-loss weight used
        when ``StructuralModel(order=True)``. Supports either ``{"values": [...]}``
        or ``{"min": ..., "max": ..., "sampling": ...}``.
    metric : {"correlation", "redundancy"}, optional
        Metric optimised for each view. Currently only ``"correlation"`` is
        implemented, which maximises the average Pearson correlation between the
        target view and its connected partners.
    seed : int, optional
        Seed used when sampling hyperparameters via ``keras_tuner``.
    """

    def __init__(
        self,
        view_builders: Sequence[Callable[[kt.HyperParameters, int], keras.Model]],
        structural_kwargs: Dict[str, Any],
        *,
        n_loops: int = 3,
        max_trials_per_view: int = 5,
        sparse_config: Optional[Iterable[Dict[str, Any]]] = None,
        regularizer_config: Optional[Iterable[Dict[str, Any]]] = None,
        order_loss_config: Optional[Dict[str, Any]] = None,
        metric: str = "correlation",
        seed: Optional[int] = None,
        search_run_eagerly: bool = True,
    ) -> None:
        if not view_builders:
            raise ValueError("view_builders must contain at least one entry.")

        self.view_builders = list(view_builders)
        self.structural_kwargs = copy.deepcopy(structural_kwargs)
        self.n_loops = int(n_loops)
        self.max_trials_per_view = int(max_trials_per_view)
        self.metric = metric
        self.seed = seed
        self.search_run_eagerly = search_run_eagerly

        # Structural defaults and validation
        required_keys = ("Path", "tot_num", "ndims")
        for key in required_keys:
            if key not in self.structural_kwargs:
                raise ValueError(f"structural_kwargs must include '{key}'")

        self.Path = np.asarray(self.structural_kwargs["Path"], dtype=np.float32)
        self.n_views = len(self.view_builders)
        if self.Path.shape[0] != self.n_views:
            raise ValueError("Path dimension must match number of view builders.")

        base_sparse = self.structural_kwargs.get("sparse_l1_list")
        if base_sparse is None:
            base_sparse = [0.0] * self.n_views
        if len(base_sparse) != self.n_views:
            raise ValueError("Length of sparse_l1_list must equal number of views.")

        base_reg = self.structural_kwargs.get("regularizer_list")
        if base_reg is None:
            base_reg = [None] * self.n_views
        if len(base_reg) != self.n_views:
            raise ValueError("Length of regularizer_list must equal number of views.")

        self.current_sparse_l1_list = list(base_sparse)
        self.current_regularizer_list = list(base_reg)
        self.current_order_loss_weight = float(self.structural_kwargs.get("order_loss_weight", 1.0))

        self.sparse_config = sparse_config
        self.regularizer_config = regularizer_config
        self.order_loss_config = order_loss_config

        self.view_hp_store: List[Optional[Dict[str, Any]]] = [None] * self.n_views
        # Track globally accepted cross metric across all views
        self.current_global_cross: float = float("-inf")
        self._best_structural_metrics: Dict[str, Any] = {}

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if hasattr(value, "numpy"):
            value = value.numpy()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _extract_global_metrics(cls, metrics: Any) -> Dict[str, Optional[float]]:
        if isinstance(metrics, dict):
            return {
                "cross_metric": cls._to_float(metrics.get("cross_metric")),
                "mse_loss": cls._to_float(metrics.get("mse_loss")),
                "redundancy": cls._to_float(metrics.get("redundancy")),
                "order_loss": cls._to_float(metrics.get("order_loss")),
            }

        if isinstance(metrics, (list, tuple)):
            values = list(metrics)
            cross = values[1] if len(values) > 1 else None
            mse = values[2] if len(values) > 2 else None
            redundancy = values[3] if len(values) > 3 else None
            return {
                "cross_metric": cls._to_float(cross),
                "mse_loss": cls._to_float(mse),
                "redundancy": cls._to_float(redundancy),
                "order_loss": cls._to_float(values[4] if len(values) > 4 else None),
            }

        return {"cross_metric": None, "mse_loss": None, "redundancy": None, "order_loss": None}

    @staticmethod
    def _format_metric(value: Optional[float]) -> str:
        return "nan" if value is None else f"{value:.4f}"

    @staticmethod
    def _clone_optimizer(opt):
        def _plain_value(value: Any) -> Any:
            if isinstance(value, (bool, int, float, str, type(None))):
                return value
            if isinstance(value, np.ndarray):
                return value
            try:
                array_value = np.asarray(keras.ops.convert_to_numpy(value))
                if array_value.shape == ():
                    return array_value.item()
                return array_value
            except Exception:
                return value

        optimizer_cls = opt.__class__
        signature = inspect.signature(optimizer_cls.__init__)
        kwargs = {}

        for name, parameter in signature.parameters.items():
            if name in {"self", "args", "kwargs"}:
                continue
            if not hasattr(opt, name):
                continue
            if parameter.kind not in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }:
                continue
            kwargs[name] = _plain_value(getattr(opt, name))

        try:
            return optimizer_cls(**kwargs)
        except Exception:
            learning_rate = _plain_value(getattr(opt, "learning_rate", 1e-3))
            return optimizer_cls(learning_rate=learning_rate)

    def _clone_optimizers(self, optimizers: Any):
        if isinstance(optimizers, (list, tuple)):
            return [self._clone_optimizer(opt) for opt in optimizers]
        return self._clone_optimizer(optimizers)

    def _build_view_from_store(self, view_index: int) -> keras.Model:
        builder = self.view_builders[view_index]
        hp = kt.HyperParameters()
        stored = self.view_hp_store[view_index]
        if stored:
            for name, value in stored.items():
                hp.values[name] = value
        return builder(hp, view_index)

    def _initialise_view_if_needed(self, view_index: int) -> None:
        if self.view_hp_store[view_index] is not None:
            return
        hp = kt.HyperParameters()
        _ = self.view_builders[view_index](hp, view_index)
        self.view_hp_store[view_index] = {
            k: v for k, v in hp.values.items() if k.startswith(f"view{view_index}_")
        }

    def _make_structural_model(
        self,
        model_list: Sequence[keras.Model],
        sparse_l1_list: Sequence[float],
        regularizer_list: Sequence[Optional[keras.regularizers.Regularizer]],
        order_loss_weight: float,
    ) -> StructuralModel:
        kwargs = {
            "Path": self.structural_kwargs["Path"],
            "model_list": list(model_list),
            "regularizer_list": list(regularizer_list),
            "tot_num": self.structural_kwargs["tot_num"],
            "ndims": self.structural_kwargs["ndims"],
            "orthogonalization": self.structural_kwargs.get("orthogonalization", "Moore-Penrose"),
            "momentum": self.structural_kwargs.get("momentum", 0.95),
            "epsilon": self.structural_kwargs.get("epsilon", 1e-4),
            "train_DLV": self.structural_kwargs.get("train_DLV", True),
            "is_siamese": self.structural_kwargs.get("is_siamese", False),
            "diag_offset": self.structural_kwargs.get("diag_offset", 1e-3),
            "sparse_l1_list": list(sparse_l1_list),
            "orthog_weight": self.structural_kwargs.get("orthog_weight", 0.0),
            "order": self.structural_kwargs.get("order", False),
            "order_loss_weight": order_loss_weight,
        }

        return StructuralModel(**kwargs)

    def _evaluate_view(self, model: StructuralModel, view_index: int, eval_data: Any) -> float:
        preds = model.predict(eval_data, verbose=0)
        if not isinstance(preds, np.ndarray):
            preds = np.asarray(preds)

        if preds.ndim != 3:
            raise ValueError("Predictions must have shape (samples, ndims, n_views)")

        target = preds[:, :, view_index]
        eps = 1e-8
        corrs = []
        for j in range(self.n_views):
            if j == view_index or self.Path[view_index, j] == 0:
                continue
            other = preds[:, :, j]
            tgt_c = target - target.mean(axis=0, keepdims=True)
            oth_c = other - other.mean(axis=0, keepdims=True)
            denom = np.sqrt((tgt_c**2).sum(axis=0) + eps) * np.sqrt((oth_c**2).sum(axis=0) + eps)
            corr = (tgt_c * oth_c).sum(axis=0) / denom
            corrs.append(np.mean(corr))

        return float(np.mean(corrs)) if corrs else 0.0

    def search(
        self,
        train_data: Any,
        *,
        optimizers: Any,
        validation_data: Optional[Any] = None,
        loops: Optional[int] = None,
        max_trials_per_view: Optional[int] = None,
        fit_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Runs the coordinate-descent hyperparameter search.

        Parameters
        ----------
        train_data : Any
            Training inputs for all measurement models.
        optimizers : Any
            Optimizer or list of optimizers to compile the StructuralModel with.
        validation_data : Any, optional
            Separate validation inputs used to score each trial. Defaults to
            ``train_data`` when ``None``.
        loops, max_trials_per_view, fit_kwargs : see class documentation.
        """

        fit_kwargs = dict(fit_kwargs or {})
        val_data = validation_data if validation_data is not None else train_data
        loops = loops or self.n_loops
        max_trials = max_trials_per_view or self.max_trials_per_view

        for idx in range(self.n_views):
            self._initialise_view_if_needed(idx)

        loop_prog = Progbar(loops, verbose=1, unit_name="loop")
        for loop in range(loops):
            view_prog = Progbar(self.n_views, verbose=1, unit_name="view")
            for view_idx in range(self.n_views):
                print(f"Loop {loop + 1}/{loops} – Optimising view {view_idx + 1}/{self.n_views}")
                trial_prog = Progbar(max_trials, verbose=0, unit_name="trial")
                for trial in range(max_trials):
                    hp = kt.HyperParameters()
                    if self.seed is not None:
                        hp.values["seed"] = self.seed + trial

                    model_list = []
                    for idx in range(self.n_views):
                        if idx == view_idx:
                            model_list.append(self.view_builders[idx](hp, idx))
                        else:
                            model_list.append(self._build_view_from_store(idx))

                    struct_cfg = sample_structural_hparams(
                        hp,
                        n_views=self.n_views,
                        target_view=view_idx,
                        current_sparse=self.current_sparse_l1_list,
                        current_regularizers=self.current_regularizer_list,
                        current_order_loss_weight=self.current_order_loss_weight,
                        sparse_config=self.sparse_config,
                        regularizer_config=self.regularizer_config,
                        order_loss_config=self.order_loss_config,
                    )

                    struct_model = self._make_structural_model(
                        model_list,
                        sparse_l1_list=struct_cfg["sparse_l1_list"],
                        regularizer_list=struct_cfg["regularizer_list"],
                        order_loss_weight=struct_cfg["order_loss_weight"],
                    )
                    struct_model.compile(self._clone_optimizers(optimizers))
                    if self.search_run_eagerly and hasattr(struct_model, "run_eagerly"):
                        struct_model.run_eagerly = True
                    struct_model.fit(train_data, **fit_kwargs)

                    metrics = struct_model.evaluate(val_data, verbose=0)
                    global_metrics = self._extract_global_metrics(metrics)
                    score = global_metrics["cross_metric"]
                    if score is None:
                        score = self._evaluate_view(struct_model, view_idx, val_data)
                    trial_prog.update(trial + 1, values=[("cross", score)])

                    # Accept only when the global cross metric improves vs current best
                    if score is not None and score > self.current_global_cross:
                        self.view_hp_store[view_idx] = {
                            k: v for k, v in hp.values.items() if k.startswith(f"view{view_idx}_")
                        }
                        self.current_sparse_l1_list = list(struct_cfg["sparse_l1_list"])
                        self.current_regularizer_list = list(struct_cfg["regularizer_list"])
                        self.current_order_loss_weight = float(struct_cfg["order_loss_weight"])
                        self.current_global_cross = score
                        self._best_structural_metrics = {
                            "loop": loop,
                            "view": view_idx,
                            "cross": score,
                            "order_loss_weight": self.current_order_loss_weight,
                        }
                        cross = score
                        mse = global_metrics["mse_loss"]
                        red = global_metrics["redundancy"]
                        order_loss = global_metrics["order_loss"]
                        print(
                            f"[Improved] Loop {loop + 1}, view {view_idx + 1}: "
                            f"cross={self._format_metric(cross)}, "
                            f"mse={self._format_metric(mse)}, "
                            f"redundancy={self._format_metric(red)}, "
                            f"order_loss={self._format_metric(order_loss)}, "
                            f"order_loss_weight={self.current_order_loss_weight:.4g}"
                        )
                view_prog.update(view_idx + 1)
            loop_prog.update(loop + 1)

    def build_best_model(self, optimizers: Any, run_eagerly: bool = False) -> StructuralModel:
        """Rebuilds a StructuralModel from the best discovered configuration."""

        model_list = [self._build_view_from_store(i) for i in range(self.n_views)]

        struct_model = self._make_structural_model(
            model_list,
            sparse_l1_list=self.current_sparse_l1_list,
            regularizer_list=self.current_regularizer_list,
            order_loss_weight=self.current_order_loss_weight,
        )
        struct_model.compile(self._clone_optimizers(optimizers))
        struct_model.run_eagerly = run_eagerly
        return struct_model
