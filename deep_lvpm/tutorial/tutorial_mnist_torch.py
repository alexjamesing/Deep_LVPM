"""
MNIST tutorial using Deep LVPM with the PyTorch backend of Keras 3.

Run from the command line:

    python -m deep_lvpm.tutorial.tutorial_mnist_torch

This script mirrors the TensorFlow example but keeps the image and label
processing models in pure PyTorch while leveraging Deep LVPM's Keras-based
StructuralModel layer stack under the torch backend.
"""

from __future__ import annotations

import os

# Ensure the torch backend is active before importing Keras.
os.environ.setdefault("KERAS_BACKEND", "torch")


# # Allow PyTorch to fall back to CPU for ops missing on Apple’s MPS backend.
if os.environ.get("KERAS_BACKEND", "").lower() == "torch":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")



import numpy as np
import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)  # enable once
import keras
from keras.optimizers import Adam

from sklearn.manifold import TSNE

from deep_lvpm.models.StructuralModel import StructuralModel


def _evaluate_structural_model(model: StructuralModel, data) -> dict[str, float]:
    """Return eval metrics as plain floats regardless of backend response type."""

    results = model.evaluate(data, verbose=False)
    if isinstance(results, dict):
        return {key: float(value) for key, value in results.items()}
    return {f"metric_{idx}": float(val) for idx, val in enumerate(results)}


class TorchModuleLayer(keras.layers.Layer):
    """Wrap a pure PyTorch nn.Module for use inside a Keras model."""

    def __init__(self, torch_module: nn.Module, **kwargs) -> None:
        super().__init__(**kwargs)
        self.torch_module = torch_module
        self._current_device: str | None = None
        self._feature_dim: int | None = None

    def _prepare_tensor(self, inputs) -> torch.Tensor:
        tensor = torch.as_tensor(inputs, dtype=torch.float32)
        if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(0, 3, 1, 2).contiguous()
        return tensor

    def build(self, input_shape):
        device = torch.device("cpu")
        self.torch_module.to(device)

        spatial_shape = [int(dim) for dim in input_shape[1:]]
        dummy = torch.zeros((1, *spatial_shape), dtype=torch.float32, device=device)
        dummy = self._prepare_tensor(dummy)

        with torch.no_grad():
            features = self.torch_module(dummy)
            if features.ndim > 2:
                features = torch.flatten(features, start_dim=1)

        self._feature_dim = int(features.shape[-1])
        self._current_device = device.type

        super().build(input_shape)

    def call(self, inputs, training: bool = False):
        tensor = self._prepare_tensor(inputs)

        device = tensor.device
        device_type = device.type

        if device_type == "meta":
            batch = tensor.shape[0]
            feature_dim = self._feature_dim or 1
            return torch.zeros(
                (batch, feature_dim),
                device=device,
                dtype=tensor.dtype,
            )

        if self._current_device != device_type:
            self.torch_module.to(device)
            self._current_device = device_type

        self.torch_module.train(training)
        features = self.torch_module(tensor)

        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)

        return features


if __name__ == "__main__":
    print("=== Deep LVPM MNIST tutorial (PyTorch backend) ===")
    backend_name = getattr(keras.backend, "backend", lambda: "unknown")()
    print(f"Keras backend: {backend_name}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

    # ------------------------------------------------------------------
    # Step 1. Load and prepare MNIST just as in the TensorFlow tutorial.
    # ------------------------------------------------------------------
    # The preprocessing mirrors the TF script so the measurement models are
    # interchangeable: normalise pixels to [0, 1], cast to float32, and add a
    # channel dimension.  Labels stay as one-hot vectors for the identity view.
    num_classes = 10
    print("Loading MNIST data...")
    (x_train, y_train_cat), (x_test, y_test_cat) = keras.datasets.mnist.load_data()

    print("Preprocessing images and labels...")
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    y_train = keras.utils.to_categorical(y_train_cat, num_classes)
    y_test = keras.utils.to_categorical(y_test_cat, num_classes)

    data_train = [x_train, y_train]
    data_test = [x_test, y_test]

    # ------------------------------------------------------------------
    # Step 2. Define the pure PyTorch measurement modules.
    # ------------------------------------------------------------------
    # Each measurement model is authored entirely in torch.nn.  We wrap the
    # modules with TorchModuleLayer so they plug into the Keras-based
    # StructuralModel without rewriting the training loop.
    print("Building PyTorch measurement modules...")
    image_torch_module = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 64, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(64 * 5 * 5, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
    )

    label_torch_module = nn.Sequential(
        nn.Identity(),
    )

    adjacency = np.array([[0, 1], [1, 0]], dtype="float32")
    ndims = 9
    momentum = 0.95
    epsilon = 1e-4
    tot_num = x_train.shape[0]

    image_input = keras.Input(shape=x_train.shape[1:], name="mnist_image_in")
    image_features = TorchModuleLayer(
        torch_module=image_torch_module,
        name="mnist_image_torch",
    )(image_input)
    image_model = keras.Model(image_input, image_features, name="mnist_image_encoder")

    label_input = keras.Input(shape=(num_classes,), name="mnist_label_in")
    label_features = TorchModuleLayer(
        torch_module=label_torch_module,
        name="mnist_label_torch",
    )(label_input)
    label_model = keras.Model(label_input, label_features, name="mnist_label_encoder")

    # ------------------------------------------------------------------
    # Step 3. Assemble and train the StructuralModel.
    # ------------------------------------------------------------------
    structural_model = StructuralModel(
        Path=adjacency,
        model_list=[image_model, label_model],
        regularizer_list=[None, None],
        tot_num=tot_num,
        ndims=ndims,
        orthogonalization="zca",
        momentum=momentum,
        epsilon=epsilon,
        train_DLV=True,
    )

    # Matching optimisation strategy: one Adam instance per measurement view.
    print("Compiling StructuralModel...")
    image_optimizer = Adam(learning_rate=1e-5)
    label_optimizer = Adam(learning_rate=1e-5)
    structural_model.compile(optimizer=[image_optimizer, label_optimizer])

    print("Training...")
    history = structural_model.fit(
        data_train,
        batch_size=256,
        epochs=20,
        verbose=True,
        validation_split=0.1,
    )

    print("Evaluating on the test split...")
    metrics = _evaluate_structural_model(structural_model, data_test)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")

    print("Predicting latent representations...")
    latent = structural_model.predict(data_test, verbose=False)
    image_latent = structural_model.model_list[0].predict(data_test[0], verbose=False)

    latent_array = np.asarray(latent)
    image_latent_array = np.asarray(image_latent)

    tsne = TSNE(n_components=2, random_state=42)
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(
        image_latent_array.shape[0],
        size=min(200, image_latent_array.shape[0]),
        replace=False,
    )
    tsne_projection = tsne.fit_transform(image_latent_array[sample_indices])

    print("Latent tensor shape:", latent_array.shape)
    print("t-SNE projection shape:", tsne_projection.shape)
    print("Training history keys:", list(history.history))
