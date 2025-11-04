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

    # ------------------------------------------------------------------
    # Step 1. Load and prepare the MNIST dataset.
    # ------------------------------------------------------------------
    # The canonical MNIST loader returns uint8 digits with shape (n, 28, 28).
    # We normalise the pixel intensities to [0, 1], convert to float32, and
    # expand a trailing singleton channel dimension so the CNN sees greyscale
    # images in NHWC format.
    num_classes = 10
    input_shape = (28, 28, 1)
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
    # Step 2. Build the measurement models exactly as in the original tutorial.
    # ------------------------------------------------------------------
    # View 1: an image encoder built from Conv2D, pooling, and dense layers.
    # View 2: an identity mapping because labels are already one-hot encoded.
    print("Building measurement models...")
    image_encoder = keras.Sequential(name="mnist_image_encoder")
    image_encoder.add(layers.InputLayer(input_shape=input_shape, name="mnist_image_in"))
    image_encoder.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),
        )
    )
    image_encoder.add(layers.MaxPooling2D((2, 2)))
    image_encoder.add(
        layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),
        )
    )
    image_encoder.add(layers.MaxPooling2D((2, 2)))
    image_encoder.add(layers.Flatten())
    image_encoder.add(layers.Dense(128, activation="relu"))
    image_encoder.add(layers.Dropout(rate=0.5))

    labels_input = keras.Input(shape=(num_classes,), name="mnist_label_in")
    labels_output = layers.Activation("linear", name="mnist_label_id")(labels_input)
    label_encoder = keras.Model(labels_input, labels_output, name="mnist_label_encoder")

    # ------------------------------------------------------------------
    # Step 3. Configure the StructuralModel with the classic two-view setup.
    # ------------------------------------------------------------------
    # The 2x2 adjacency matrix encodes a symmetric relationship: each view
    # connects to the other.  ``tot_num`` tells FactorLayer how many samples
    # exist in the full dataset so it can keep running covariance statistics.
    adjacency = np.array([[0, 1], [1, 0]], dtype="float32")
    total_examples = x_train.shape[0]

    structural_model = StructuralModel(
        Path=adjacency,
        model_list=[image_encoder, label_encoder],
        regularizer_list=[None, None],
        tot_num=total_examples,
        ndims=9,
        orthogonalization="Moore-Penrose",
        momentum=0.95,
        epsilon=1e-4,
        train_DLV=False,
    )

    # Give each view its own Adam optimiser to keep learning rates independent.
    print("Compiling StructuralModel...")
    image_optimizer = Adam(learning_rate=1e-4)
    label_optimizer = Adam(learning_rate=1e-4)
    structural_model.compile(optimizer=[image_optimizer, label_optimizer])

    # ------------------------------------------------------------------
    # Step 4. Train the structural model then evaluate on held-out data.
    # ------------------------------------------------------------------
    # Keras 3 still accepts lists-of-arrays for multi-view training.  We keep a
    # small validation split to monitor convergence and watch the redundancy
    # metric during training.
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

    # ------------------------------------------------------------------
    # Step 5. Inspect the learned latent representations.
    # ------------------------------------------------------------------
    # ``predict`` returns a (n_samples, ndims, n_views) tensor of deep latent
    # variables.  We also grab the standalone encoder output for the image view
    # so we can visualise a subset with t-SNE.
    print("Predicting latent representations...")
    latent = structural_model.predict(data_test, verbose=False)
    image_latent = structural_model.model_list[0].predict(data_test[0], verbose=False)

    tsne = TSNE(n_components=2, random_state=42)
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(image_latent.shape[0], size=min(200, image_latent.shape[0]), replace=False)
    tsne_projection = tsne.fit_transform(image_latent[sample_indices])

    print("Latent tensor shape:", latent.shape)
    print("t-SNE projection shape:", tsne_projection.shape)
    print("Training history keys:", list(history.history))
