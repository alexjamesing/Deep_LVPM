from deep_lvpm.layers.ConfoundLayer import ConfoundLayer
from deep_lvpm.layers.FactorLayer import FactorLayer
from deep_lvpm.layers.ZCALayer import ZCALayer


def l1_l2(l1: float = 0.0, l2: float = 0.0):
    """
    Convenience helper — drop-in replacement for keras.regularizers.L1L2().

    Returns a (l1, l2) tuple accepted by FactorLayer/ZCALayer, or None
    when both penalties are zero.
    """
    if l1 == 0.0 and l2 == 0.0:
        return None
    return (float(l1), float(l2))


__all__ = ["FactorLayer", "ZCALayer", "ConfoundLayer", "l1_l2"]
