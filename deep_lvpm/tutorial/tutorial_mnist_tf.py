"""
MNIST tutorial using Deep LVPM with the TensorFlow backend.

Run from the command line:

    python -m deep_lvpm.tutorial.tutorial_mnist_tf

The script trains a two-view StructuralModel that links MNIST images to
dummy-coded labels and then reports evaluation metrics plus a small t-SNE
projection of the learned image factors.
"""

from __future__ import annotations

import os

# Ensure TensorFlow backend before importing Keras.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf

import keras
from keras import layers, regularizers
from keras.optimizers import Adam

from sklearn.manifold import TSNE

from deep_lvpm.models.StructuralModel import StructuralModel


def _evaluate_structural_model(model: StructuralModel, data) -> dict[str, float]:
    """Return eval metrics as plain floats regardless of backend response type."""

    results = model.evaluate(data, verbose=False)
    if isinstance(results, dict):
        return {key: float(value) for key, value in results.items()}
    return {f"metric_{idx}": float(val) for idx, val in enumerate(results)}


if __name__ == "__main__":
    print("=== Deep LVPM MNIST tutorial (TensorFlow backend) ===")
    print(f"Keras backend: {keras.backend.backend()}")
    print(f"Physical GPUs: {tf.config.list_physical_devices('GPU')}")

    num_classes = 10
    input_shape = (28, 28, 1)

    (x_train, y_train_cat), (x_test, y_test_cat) = keras.datasets.mnist.load_data()

    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]

    y_train = keras.utils.to_categorical(y_train_cat, num_classes)
    y_test = keras.utils.to_categorical(y_test_cat, num_classes)

    data_train = [x_train, y_train]
    data_test = [x_test, y_test]

    image_encoder = keras.Sequential(
        [
            keras.Input(shape=input_shape, name="mnist_image_in"),
            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
        ],
        name="mnist_image_encoder",
    )

    labels_input = keras.Input(shape=(num_classes,), name="mnist_label_in")
    labels_output = layers.Activation("linear", name="mnist_label_id")(labels_input)
    label_encoder = keras.Model(labels_input, labels_output, name="mnist_label_encoder")

    adjacency = np.array([[0, 1], [1, 0]], dtype="float32")

    model = StructuralModel(
        Path=adjacency,
        model_list=[image_encoder, label_encoder],
        regularizer_list=[None, None],
        tot_num=x_train.shape[0],
        ndims=9,
        orthogonalization="Moore-Penrose",
        momentum=0.95,
        epsilon=1e-4,
        train_DLV=False,
    )

    optimizers = [Adam(learning_rate=1e-4), Adam(learning_rate=1e-4)]
    model.compile(optimizer=optimizers)

    history = model.fit(
        data_train,
        batch_size=256,
        epochs=20,
        verbose=True,
        validation_split=0.1,
    )

    metrics = _evaluate_structural_model(model, data_test)
    print("Evaluation metrics:", metrics)

    latent = model.predict(data_test, verbose=False)
    image_latent = model.model_list[0].predict(data_test[0], verbose=False)

    tsne = TSNE(n_components=2, random_state=42)
    indices = np.random.default_rng(42).choice(image_latent.shape[0], size=min(200, image_latent.shape[0]), replace=False)
    tsne_projection = tsne.fit_transform(image_latent[indices])

    print("Latent tensor shape:", latent.shape)
    print("t-SNE projection shape:", tsne_projection.shape)
    print("Training history keys:", list(history.history))
