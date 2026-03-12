#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""TCGA CLIP Tutorial (TensorFlow backend)

This script mirrors the TCGA DLVPM tutorial but uses the CLIP class
(`deep_lvpm.multi_model.CLIP`) to learn a shared embedding across five
modalities (histology features, RNA‑seq, methylation, miRNA, SNVs).

It builds one small encoder per modality, appends a Dense projection head of
size `ndims`, and trains with the averaged directed CLIP loss across all
ordered modality pairs. No tuner is used here.
"""

import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf
import keras
from keras import layers, regularizers, optimizers
from importlib import resources

from deep_lvpm.multi_model import CLIP


def load_tcga_train():
    with resources.as_file(resources.files("deep_lvpm.data") / "Lung_multiomics_sample_train.npz") as f:
        arrays = np.load(f)
        rnaseq = arrays["rnaseq"].astype("float32")
        snv = arrays["snv"].astype("float32")
        methylation = arrays["methylation"].astype("float32")
        mirna = arrays["mirna"].astype("float32")
        histo20 = arrays["histo20"].astype("float32")
    return [histo20, rnaseq, methylation, mirna, snv]


def load_tcga_test():
    with resources.as_file(resources.files("deep_lvpm.data") / "Lung_multiomics_sample_test.npz") as f:
        arrays = np.load(f)
        rnaseq = arrays["rnaseq"].astype("float32")
        snv = arrays["snv"].astype("float32")
        methylation = arrays["methylation"].astype("float32")
        mirna = arrays["mirna"].astype("float32")
        histo20 = arrays["histo20"].astype("float32")
    return [histo20, rnaseq, methylation, mirna, snv]


def build_encoder(input_dim: int, name: str) -> keras.Model:
    """Simple MLP encoder; CLIP attaches the Dense(ndims) head internally."""
    inputs = keras.Input(shape=(input_dim,), name=f"{name}_in")
    x = layers.Dense(512, activation="relu", name=f"{name}_dense1")(inputs)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Dropout(0.2, name=f"{name}_drop1")(x)
    x = layers.Dense(512, activation="relu", name=f"{name}_dense2")(x)
    outputs = layers.BatchNormalization(name=f"{name}_bn2")(x)
    return keras.Model(inputs, outputs, name=f"{name}_enc")


def main():
    # Data --------------------------------------------------------------
    X_train = load_tcga_train()
    X_test = load_tcga_test()

    n_views = len(X_train)
    ndims = 64  # CLIP embedding size
    tot_num = X_train[0].shape[0]

    # Measurement models -----------------------------------------------
    encoders = [
        build_encoder(X_train[v].shape[1], name) for v, name in enumerate(["histo20", "rnaseq", "meth", "mirna", "snv"])
    ]

    # Projection-layer regularizers (optional)
    proj_regs = [regularizers.L2(1e-4) for _ in range(n_views)]

    # Build CLIP model --------------------------------------------------
    model = CLIP(model_list=encoders, regularizer_list=proj_regs, ndims=ndims)

    # Per-view optimizers (same LR here, could be different per view)
    lr = 1e-3
    opt_list = [keras.optimizers.Adam(learning_rate=lr) for _ in range(n_views)]
    model.compile(opt_list)

    # Train -------------------------------------------------------------
    batch_size = 256
    epochs = 100
    history = model.fit(X_train, batch_size=batch_size, epochs=epochs, verbose=True)

    # Evaluate clip loss on train and test ------------------------------
    train_metrics = model.evaluate(X_train, verbose=False)
    test_metrics = model.evaluate(X_test, verbose=False)
    print("Train metrics:", train_metrics)
    print("Test metrics:", test_metrics)

    # (Optional) quick retrieval@1 estimate across first two modalities ---
    Z_test = model.predict(X_test, verbose=False)  # (B, d, M)
    # compute retrieval@1 from modality 0 -> 1
    z0 = Z_test[:, :, 0]
    z1 = Z_test[:, :, 1]
    sims = z0 @ z1.T
    r1 = (np.argmax(sims, axis=1) == np.arange(sims.shape[0])).mean()
    print(f"Top‑1 retrieval (view0→view1): {r1:.3f}")


if __name__ == "__main__":
    tf.config.run_functions_eagerly(False)
    main()
