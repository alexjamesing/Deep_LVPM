"""
Legacy entry point for Deep LVPM tutorials.

The tutorials now live in dedicated scripts:

    python -m deep_lvpm.tutorial.tutorial_mnist_tf
    python -m deep_lvpm.tutorial.tutorial_tcga_tf
    python -m deep_lvpm.tutorial.tutorial_siamese_tf

This file is kept for backward compatibility and simply prints guidance.
"""

from __future__ import annotations

import sys

MESSAGE = (
    "The tutorial aggregator has been replaced with dedicated scripts.\n"
    "Run one of:\n"
    "  python -m deep_lvpm.tutorial.tutorial_mnist_tf\n"
    "  python -m deep_lvpm.tutorial.tutorial_tcga_tf\n"
    "  python -m deep_lvpm.tutorial.tutorial_siamese_tf\n"
)


if __name__ == "__main__":
    sys.stderr.write(MESSAGE)
    raise SystemExit(1)
