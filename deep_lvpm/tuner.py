#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small coordinate-descent tuner for the PyTorch DLVPM implementation."""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from deep_lvpm import regularizers
from deep_lvpm.model import StructuralModel


class HyperParameters:
    """Minimal local hyperparameter sampler."""

    def __init__(self, seed: int | None = None):
        self.values: dict[str, Any] = {}
        self.rng = random.Random(seed)

    def Choice(self, name, values, default=None):
        if name in self.values:
            return self.values[name]
        value = default if default is not None else self.rng.choice(list(values))
        if default is None:
            value = self.rng.choice(list(values))
        self.values[name] = value
        return value

    def Float(self, name, min_value, max_value, sampling=None, default=None, step=None):
        if name in self.values:
            return self.values[name]
        if default is not None:
            value = float(default)
        elif sampling == "log":
            log_min = math.log(float(min_value))
            log_max = math.log(float(max_value))
            value = math.exp(self.rng.uniform(log_min, log_max))
        elif step is not None:
            choices = list(np.arange(min_value, max_value + 0.5 * step, step))
            value = float(self.rng.choice(choices))
        else:
            value = self.rng.uniform(float(min_value), float(max_value))
        self.values[name] = value
        return value

    def Int(self, name, min_value, max_value, step=1, default=None):
        if name in self.values:
            return self.values[name]
        if default is not None:
            value = int(default)
        else:
            choices = list(range(int(min_value), int(max_value) + 1, int(step)))
            value = int(self.rng.choice(choices))
        self.values[name] = value
        return value


def _ensure_per_view_config(config: Optional[Iterable[Dict[str, Any]]], n_views: int) -> List[Dict[str, Any]]:
    if config is None:
        return [dict() for _ in range(n_views)]
    if isinstance(config, (list, tuple)):
        if len(config) != n_views:
            raise ValueError(f"Config must have length {n_views}, got {len(config)}")
        return [dict(cfg) for cfg in config]
    return [dict(config) for _ in range(n_views)]


def _sample_sparse_value(hp: HyperParameters, cfg: Dict[str, Any], view_index: int) -> float:
    name = f"sparse_l1_view{view_index}"
    if "values" in cfg:
        values = cfg["values"]
        if not values:
            raise ValueError("Sparse config 'values' list must not be empty.")
        return float(hp.Choice(name, values=values, default=cfg.get("default")))

    return float(
        hp.Float(
            name,
            min_value=cfg.get("min", 0.0),
            max_value=cfg.get("max", 1e-4),
            sampling=cfg.get("sampling", "log"),
            default=cfg.get("default"),
        )
    )


def _sample_regularizer(hp: HyperParameters, cfg: Dict[str, Any], view_index: int):
    choices = cfg.get("choices", ["none", "l2", "l1l2"])
    if not choices:
        raise ValueError("Regularizer 'choices' list must not be empty.")

    choice = hp.Choice(f"regularizer_view{view_index}", values=choices, default=cfg.get("default_choice"))

    def _sample_range(range_cfg: Dict[str, Any], suffix: str) -> float:
        return float(
            hp.Float(
                f"regularizer_view{view_index}_{suffix}",
                min_value=range_cfg.get("min", 1e-6),
                max_value=range_cfg.get("max", 1e-2),
                sampling=range_cfg.get("sampling", "log"),
                default=range_cfg.get("default"),
            )
        )

    if choice == "none":
        return None
    if choice == "l2":
        return regularizers.l2(_sample_range(cfg.get("l2_range", {}), "l2"))
    if choice == "l1":
        return regularizers.l1(_sample_range(cfg.get("l1_range", {}), "l1"))
    return regularizers.l1_l2(
        l1=_sample_range(cfg.get("l1_range", {}), "l1"),
        l2=_sample_range(cfg.get("l2_range", {}), "l2"),
    )


def sample_structural_hparams(
    hp: HyperParameters,
    n_views: int,
    target_view: int,
    current_sparse: Sequence[float],
    current_regularizers: Sequence[Any],
    sparse_config: Optional[Iterable[Dict[str, Any]]] = None,
    regularizer_config: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    sparse_cfgs = _ensure_per_view_config(sparse_config, n_views)
    regularizer_cfgs = _ensure_per_view_config(regularizer_config, n_views)

    if not (0 <= target_view < n_views):
        raise ValueError(f"target_view must be in [0, {n_views})")

    sparse_list = list(current_sparse)
    reg_list = list(current_regularizers)
    sparse_list[target_view] = _sample_sparse_value(hp, sparse_cfgs[target_view], target_view)
    reg_list[target_view] = _sample_regularizer(hp, regularizer_cfgs[target_view], target_view)
    return {"sparse_l1_list": sparse_list, "regularizer_list": reg_list}


class Tuner:
    """Coordinate-descent tuner over view builders and structural parameters."""

    def __init__(
        self,
        view_builders: Sequence[Callable[[HyperParameters, int], torch.nn.Module]],
        structural_kwargs: Dict[str, Any],
        *,
        n_loops: int = 3,
        max_trials_per_view: int = 5,
        sparse_config: Optional[Iterable[Dict[str, Any]]] = None,
        regularizer_config: Optional[Iterable[Dict[str, Any]]] = None,
        metric: str = "correlation",
        seed: Optional[int] = None,
        search_run_eagerly: bool = True,
    ) -> None:
        del search_run_eagerly
        if not view_builders:
            raise ValueError("view_builders must contain at least one entry.")
        for key in ("Path", "tot_num", "ndims"):
            if key not in structural_kwargs:
                raise ValueError(f"structural_kwargs must include '{key}'")

        self.view_builders = list(view_builders)
        self.structural_kwargs = copy.deepcopy(structural_kwargs)
        self.n_loops = int(n_loops)
        self.max_trials_per_view = int(max_trials_per_view)
        self.metric = metric
        self.seed = seed
        self.Path = np.asarray(self.structural_kwargs["Path"], dtype=np.float32)
        self.n_views = len(self.view_builders)
        self.sparse_config = sparse_config
        self.regularizer_config = regularizer_config

        base_sparse = self.structural_kwargs.get("sparse_l1_list", [0.0] * self.n_views)
        base_reg = self.structural_kwargs.get("regularizer_list", [None] * self.n_views)
        self.current_sparse_l1_list = list(base_sparse)
        self.current_regularizer_list = list(base_reg)
        self.view_hp_store: list[dict[str, Any] | None] = [None] * self.n_views
        self.current_global_cross = float("-inf")
        self._best_structural_metrics: dict[str, Any] = {}

    @staticmethod
    def _clone_optimizer(opt, parameters):
        cls = opt.__class__
        kwargs = dict(opt.defaults)
        return cls(parameters, **kwargs)

    def _clone_optimizers(self, optimizers, model: StructuralModel):
        if isinstance(optimizers, (list, tuple)):
            return [
                self._clone_optimizer(opt, view_model.parameters())
                for opt, view_model in zip(optimizers, model.model_list)
            ]
        return [
            self._clone_optimizer(optimizers, view_model.parameters())
            for view_model in model.model_list
        ]

    def _build_view_from_store(self, view_index: int):
        hp = HyperParameters(seed=self.seed)
        stored = self.view_hp_store[view_index]
        if stored:
            hp.values.update(stored)
        return self.view_builders[view_index](hp, view_index)

    def _initialise_view_if_needed(self, view_index: int) -> None:
        if self.view_hp_store[view_index] is not None:
            return
        hp = HyperParameters(seed=self.seed)
        _ = self.view_builders[view_index](hp, view_index)
        self.view_hp_store[view_index] = {
            key: value for key, value in hp.values.items() if key.startswith(f"view{view_index}_")
        }

    def _make_structural_model(self, model_list, sparse_l1_list, regularizer_list) -> StructuralModel:
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
            "order": self.structural_kwargs.get("order", False),
            "order_association_cutoff": self.structural_kwargs.get("order_association_cutoff"),
            "device": self.structural_kwargs.get("device"),
        }
        return StructuralModel(**kwargs)

    def search(
        self,
        train_data,
        *,
        optimizers,
        validation_data=None,
        loops: Optional[int] = None,
        max_trials_per_view: Optional[int] = None,
        fit_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        fit_kwargs = dict(fit_kwargs or {})
        val_data = validation_data if validation_data is not None else train_data
        loops = loops or self.n_loops
        max_trials = max_trials_per_view or self.max_trials_per_view

        for view_index in range(self.n_views):
            self._initialise_view_if_needed(view_index)

        for loop in range(loops):
            for view_index in range(self.n_views):
                print(f"Loop {loop + 1}/{loops} - optimising view {view_index + 1}/{self.n_views}")
                for trial in range(max_trials):
                    hp = HyperParameters(seed=None if self.seed is None else self.seed + loop * max_trials + trial)
                    model_list = []
                    for index in range(self.n_views):
                        if index == view_index:
                            model_list.append(self.view_builders[index](hp, index))
                        else:
                            model_list.append(self._build_view_from_store(index))

                    struct_cfg = sample_structural_hparams(
                        hp,
                        n_views=self.n_views,
                        target_view=view_index,
                        current_sparse=self.current_sparse_l1_list,
                        current_regularizers=self.current_regularizer_list,
                        sparse_config=self.sparse_config,
                        regularizer_config=self.regularizer_config,
                    )
                    model = self._make_structural_model(
                        model_list,
                        sparse_l1_list=struct_cfg["sparse_l1_list"],
                        regularizer_list=struct_cfg["regularizer_list"],
                    )
                    model.compile(self._clone_optimizers(optimizers, model))
                    model.fit(train_data, **fit_kwargs)
                    metrics = model.evaluate(val_data, verbose=False)
                    score = metrics.get("cross_metric", 0.0)
                    print(f"  trial {trial + 1}/{max_trials}: cross={score:.4f}")

                    if score > self.current_global_cross:
                        self.view_hp_store[view_index] = {
                            key: value for key, value in hp.values.items()
                            if key.startswith(f"view{view_index}_")
                        }
                        self.current_sparse_l1_list = list(struct_cfg["sparse_l1_list"])
                        self.current_regularizer_list = list(struct_cfg["regularizer_list"])
                        self.current_global_cross = score
                        self._best_structural_metrics = {
                            "loop": loop,
                            "view": view_index,
                            "cross": score,
                        }
                        print(f"  improved: cross={score:.4f}")

    def build_best_model(self, optimizers, run_eagerly: bool = False) -> StructuralModel:
        del run_eagerly
        model_list = [self._build_view_from_store(index) for index in range(self.n_views)]
        model = self._make_structural_model(
            model_list,
            sparse_l1_list=self.current_sparse_l1_list,
            regularizer_list=self.current_regularizer_list,
        )
        model.compile(self._clone_optimizers(optimizers, model))
        return model
