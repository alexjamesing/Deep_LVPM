"""Small PyTorch-friendly regularizer helpers.

Regularizers are plain ``(l1, l2)`` tuples, dictionaries with ``l1``/``l2``
keys, or ``None``.
"""

from __future__ import annotations

from typing import Iterable

import torch


def l1(value: float = 0.01) -> tuple[float, float]:
    return (float(value), 0.0)


def l2(value: float = 0.01) -> tuple[float, float]:
    return (0.0, float(value))


def l1_l2(l1: float = 0.01, l2: float = 0.01) -> tuple[float, float]:
    return (float(l1), float(l2))


def coefficients(regularizer) -> tuple[float, float]:
    if regularizer is None:
        return 0.0, 0.0

    if isinstance(regularizer, dict):
        return float(regularizer.get("l1", 0.0)), float(regularizer.get("l2", 0.0))

    if isinstance(regularizer, (list, tuple)):
        if len(regularizer) == 0:
            return 0.0, 0.0
        if len(regularizer) == 1:
            return 0.0, float(regularizer[0])
        return float(regularizer[0]), float(regularizer[1])

    if hasattr(regularizer, "l1") or hasattr(regularizer, "l2"):
        return float(getattr(regularizer, "l1", 0.0)), float(getattr(regularizer, "l2", 0.0))

    return 0.0, float(regularizer)


def penalty(parameters: Iterable[torch.Tensor], regularizer, reference: torch.Tensor | None = None) -> torch.Tensor:
    l1_value, l2_value = coefficients(regularizer)
    params = list(parameters)

    if reference is not None:
        total = torch.zeros((), dtype=reference.dtype, device=reference.device)
    else:
        if not params:
            return torch.tensor(0.0)
        total = torch.zeros((), dtype=params[0].dtype, device=params[0].device)

    for param in params:
        if l1_value:
            total = total + l1_value * param.abs().sum()
        if l2_value:
            total = total + l2_value * torch.square(param).sum()

    return total
