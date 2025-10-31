"""
CIFAR-10 Siamese tutorial using Deep LVPM with the TensorFlow backend.

Run from the command line:

    python -m deep_lvpm.tutorial.tutorial_siamese_tf

The script trains a Siamese StructuralModel that shares a convolutional encoder
between two augmented views, reports validation metrics (including redundancy),
and evaluates the learned embeddings with a linear probe.
"""

from __future__ import annotations

import os

# Force TensorFlow backend for Keras 3 before importing keras/tf.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf

import keras
from keras import layers
from keras.optimizers import Adam
from keras.mixed_precision import set_global_policy

from deep_lvpm.models.StructuralModel import StructuralModel


def _evaluate_structural_model(model: StructuralModel, data) -> dict[str, float]:
    """Return eval metrics as plain floats regardless of backend response type."""

    results = model.evaluate(data, verbose=False)
    if isinstance(results, dict):
        return {key: float(value) for key, value in results.items()}
    return {f"metric_{idx}": float(val) for idx, val in enumerate(results)}


def _make_augmenter() -> keras.Sequential:
    """Create the augmentation pipeline used to form Siamese pairs."""

    return keras.Sequential(
        [
            layers.RandomCrop(24, 24),
            layers.Resizing(32, 32),
            layers.RandomFlip("horizontal"),
            layers.Lambda(
                lambda x: tf.where(
                    tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) < 0.2,
                    tf.tile(tf.image.rgb_to_grayscale(x), [1, 1, 1, 3]),
                    x,
                )
            ),
        ],
        name="cifar_augment",
    )


def _make_dataset(images: np.ndarray, batch_size: int, seed: int, augment: keras.Model, training: bool) -> tf.data.Dataset:
    """Yield batches of paired augmentations for Siamese training."""

    autotune = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices(images)
    if training:
        ds = ds.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=training)

    def map_batch(batch):
        view_one = augment(batch, training=training)
        view_two = augment(batch, training=training)
        return ([view_one, view_two],)

    return ds.map(map_batch, num_parallel_calls=autotune).prefetch(autotune)


def _remove_last_layers(model: keras.Model, n: int) -> keras.Model:
    """Return a copy of ``model`` without the final ``n`` layers."""

    if n == 0:
        return model
    if n >= len(model.layers):
        raise ValueError(f"Cannot remove {n} layers from model with only {len(model.layers)} layers.")
    cutoff = model.layers[-(n + 1)].output
    return keras.Model(inputs=model.inputs, outputs=cutoff, name=f"{model.name}_truncated")


if __name__ == "__main__":
    print("=== Deep LVPM Siamese tutorial (TensorFlow backend) ===")
    print(f"Keras backend: {keras.backend.backend()}")
    print(f"Physical GPUs: {tf.config.list_physical_devices('GPU')}")

    set_global_policy("float32")
    tf.config.run_functions_eagerly(False)

    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except Exception:
            pass

    (x_train, y_train_cat), (x_test, y_test_cat) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train_cat = y_train_cat.squeeze()
    y_test_cat = y_test_cat.squeeze()

    y_train = keras.utils.to_categorical(y_train_cat, 10)
    y_test = keras.utils.to_categorical(y_test_cat, 10)

    seed = 1337
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x_train))
    cutoff = int(len(x_train) * 0.9)
    x_tr, x_val = x_train[indices[:cutoff]], x_train[indices[cutoff:]]

    batch_size = 512
    epochs = 500

    augment = _make_augmenter()
    train_ds = _make_dataset(x_tr, batch_size=batch_size, seed=seed, augment=augment, training=True)
    val_ds = _make_dataset(x_val, batch_size=batch_size, seed=seed, augment=augment, training=False)

    encoder = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3), name="cifar_in"),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(256, 3, padding="same", activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(512),
            layers.BatchNormalization(),
        ],
        name="cifar_encoder",
    )

    adjacency = tf.constant([[0, 1], [1, 0]], dtype="float32")

    model = StructuralModel(
        Path=adjacency,
        model_list=[encoder, encoder],
        regularizer_list=[None, None],
        tot_num=len(x_train),
        ndims=512,
        orthogonalization="zca",
        train_DLV=True,
        is_siamese=True,
        diag_offset=1e-4,
    )

    optimisers = [Adam(learning_rate=1e-4), Adam(learning_rate=1e-4)]
    model.compile(optimizer=optimisers)

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=True)
    metrics = _evaluate_structural_model(model, val_ds)
    print("Validation metrics:", metrics)
    print("Training history keys:", list(history.history))

    truncated_encoder = _remove_last_layers(model.model_list[0], n=3)
    train_latent = truncated_encoder.predict(x_train, batch_size=256, verbose=False)
    test_latent = truncated_encoder.predict(x_test, batch_size=256, verbose=False)

    linear_classifier = keras.Sequential(
        [
            keras.Input(shape=(train_latent.shape[1],)),
            layers.Dense(10, activation="softmax"),
        ],
        name="cifar_linear_probe",
    )
    linear_classifier.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    linear_classifier.fit(
        train_latent,
        y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=256,
        verbose=2,
    )

    probabilities = linear_classifier.predict(test_latent, batch_size=256, verbose=False)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = float((predictions == y_test_cat).mean())

    print(f"Linear probe accuracy on CIFAR-10 test set: {accuracy:.4f}")
