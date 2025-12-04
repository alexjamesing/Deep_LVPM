import os
import random

# Keras 3 defaults to JAX; force the TensorFlow backend before importing keras.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

# Configure TensorFlow runtime to favour deterministic, memory-efficient execution.
os.environ.update({
    "TF_XLA_FLAGS": "--tf_xla_auto_jit=0",
    "XLA_FLAGS": "--xla_gpu_strict_conv_algorithm_picker=false",
    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
    "TF_DETERMINISTIC_OPS": "1",
    "TF_CUDNN_DETERMINISTIC": "1",
    "TF_CUDNN_AUTOTUNE_DEFAULT": "0",
    "TF_CUDNN_USE_FRONTEND": "0",
    "NVIDIA_TF32_OVERRIDE": "0",
})

import numpy as np
import tensorflow as tf
import keras
from keras import layers, mixed_precision, Sequential
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from deep_lvpm.model import StructuralModel

# Keep computations in float32 for stability with the siamese objective.
mixed_precision.set_global_policy("float32")

# Enable on-demand GPU allocation so TensorFlow does not grab all memory up front.
for device in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(device, True)
    except Exception:
        pass

# Use graph execution for performance.
tf.config.run_functions_eagerly(False)

# Define core dataset metadata.
NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)

# Load CIFAR-10 and flatten label arrays.
(x_train, y_train_cat), (x_test, y_test_cat) = keras.datasets.cifar10.load_data()
y_train_cat = y_train_cat.squeeze()
y_test_cat = y_test_cat.squeeze()

# Normalise images to [0, 1] so the model trains stably.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Prepare one-hot encodings for downstream evaluation (and compatibility with other DLVPM code paths).
y_train = keras.utils.to_categorical(y_train_cat, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test_cat, NUM_CLASSES)

# Fix seeds to keep runs reproducible.
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

# Split training set into train/validation partitions.
VAL_FRACTION = 0.1
num_train = x_train.shape[0]
indices = np.arange(num_train)
rng = np.random.default_rng(SEED)
rng.shuffle(indices)
cutoff = int(num_train * (1 - VAL_FRACTION))
x_tr, x_val = x_train[indices[:cutoff]], x_train[indices[cutoff:]]

# Build stochastic augmentation pipeline used to form siamese views.
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 2048
augment = Sequential(
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
    name="augment",
)


def make_siamese_views_dataset(x, batch_size=256, shuffle=True, training=True):
    """Return a dataset that yields pairs of augmented views."""
    ds = tf.data.Dataset.from_tensor_slices(x)
    if shuffle:
        ds = ds.shuffle(len(x), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(int(batch_size), drop_remainder=training)

    def map_batch(batch):
        # Apply independent augmentations to create positive pairs.
        view_one = augment(batch, training=training)
        view_two = augment(batch, training=training)
        return ([view_one, view_two],)

    return ds.map(map_batch, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


# Create datasets for siamese training.
train_ds = make_siamese_views_dataset(
    x_tr, batch_size=BATCH_SIZE, shuffle=False, training=True
)
val_ds = make_siamese_views_dataset(
    x_val, batch_size=BATCH_SIZE, shuffle=False, training=True
)

# Build the shared encoder used by both branches of the structural model.
WEIGHT_DECAY = 0
NDIMS = 2048

CIFAR_image_model = keras.Sequential(
    [
        keras.Input(shape=INPUT_SHAPE),
        layers.Conv2D(
            64,
            3,
            padding="same",
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
        ),
        layers.MaxPooling2D(2),
        layers.Conv2D(
            128,
            3,
            padding="same",
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
        ),
        layers.MaxPooling2D(2),
        layers.Conv2D(
            256,
            3,
            padding="same",
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
        ),
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(NDIMS, activation="relu"),
        layers.BatchNormalization()

    ],
    name="cifar_image_model",
)

# Build siamese structural model with shared encoder replicas.
model_list = [CIFAR_image_model, CIFAR_image_model]
adjacency = tf.constant([[0, 1], [1, 0]], dtype="float32")
regularizers = [keras.regularizers.l2(WEIGHT_DECAY),keras.regularizers.l2(WEIGHT_DECAY)]

dlvpm_model = StructuralModel(
    adjacency,
    model_list,
    regularizers,
    x_train.shape[0],
    NDIMS,
    orthogonalization="zca",
    train_DLV=True,
    is_siamese=True,
    diag_offset=1e-4,
)

# Compile with branch-specific optimisers.
optimizers = [
    keras.optimizers.Adam(learning_rate=1e-4),
    keras.optimizers.Adam(learning_rate=1e-4),
]
dlvpm_model.compile(optimizers)

# Train the siamese model and monitor validation performance.
EPOCHS = 200
dlvpm_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=True)


def remove_last_layers(model: keras.Model, n: int = 1, name: str | None = None) -> keras.Model:
    """Return a copy of `model` without its final `n` layers."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    if n == 0:
        return model
    total_layers = len(model.layers)
    if n >= total_layers:
        raise ValueError(f"n ({n}) must be < number of layers ({total_layers})")
    cutoff_layer = model.layers[total_layers - n - 1]
    new_outputs = cutoff_layer.output
    return keras.Model(
        inputs=model.inputs, outputs=new_outputs, name=name or f"{model.name}_minus{n}"
    )


# Strip the projection head before exporting embeddings.
image_model = remove_last_layers(dlvpm_model.model_list[0], n=4)

# Generate embeddings for downstream linear evaluation.
train_dlvs = image_model.predict(x_train, batch_size=32, verbose=1)
test_dlvs = image_model.predict(x_test, batch_size=32, verbose=1)

test_ds = make_siamese_views_dataset(
    x_test, batch_size=BATCH_SIZE, shuffle=False, training=True
)

dlvpm_model.evaluate(test_ds)

print(f"Train DLVs shape: {train_dlvs.shape}")
print(f"Test  DLVs shape: {test_dlvs.shape}")

# Fit a linear SVM on top of the learned embeddings.
svm_clf = Pipeline(
    [
        ("scaler", StandardScaler(with_mean=True)),
        ("svm", LinearSVC(C=1.0, max_iter=10000, random_state=42)),
    ]
)
svm_clf.fit(train_dlvs, y_train_cat)
predictions = svm_clf.predict(test_dlvs)
accuracy = accuracy_score(y_test_cat, predictions)

print(f"\nSVM accuracy on CIFAR-10 test set: {accuracy:.4f}\n")
print("Classification report:")
print(classification_report(y_test_cat, predictions, digits=4))
print("Confusion matrix:")
print(confusion_matrix(y_test_cat, predictions))
