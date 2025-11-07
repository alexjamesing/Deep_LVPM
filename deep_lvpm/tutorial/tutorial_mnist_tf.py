# """
# MNIST tutorial using Deep LVPM with the TensorFlow backend.

# Run from the command line:

#     python -m deep_lvpm.tutorial.tutorial_mnist_tf

# The script trains a two-view StructuralModel that links MNIST images to
# dummy-coded labels and then reports evaluation metrics plus a small t-SNE
# projection of the learned image factors.
# """

# from __future__ import annotations

# import os

# # Ensure TensorFlow backend before importing Keras.
# os.environ.setdefault("KERAS_BACKEND", "tensorflow")

# import numpy as np
# import tensorflow as tf

# import keras
# from keras import layers, regularizers
# from keras.optimizers import Adam

# from sklearn.manifold import TSNE

# from deep_lvpm.models.StructuralModel import StructuralModel


# def _evaluate_structural_model(model: StructuralModel, data) -> dict[str, float]:
#     """Return eval metrics as plain floats regardless of backend response type."""

#     results = model.evaluate(data, verbose=False)
#     if isinstance(results, dict):
#         return {key: float(value) for key, value in results.items()}
#     return {f"metric_{idx}": float(val) for idx, val in enumerate(results)}


# if __name__ == "__main__":
#     print("=== Deep LVPM MNIST tutorial (TensorFlow backend) ===")
#     print(f"Keras backend: {keras.backend.backend()}")
#     print(f"Physical GPUs: {tf.config.list_physical_devices('GPU')}")

#     # ------------------------------------------------------------------
#     # Step 1. Load and prepare the MNIST dataset.
#     # ------------------------------------------------------------------
#     # The canonical MNIST loader returns uint8 digits with shape (n, 28, 28).
#     # We normalise the pixel intensities to [0, 1], convert to float32, and
#     # expand a trailing singleton channel dimension so the CNN sees greyscale
#     # images in NHWC format.
#     num_classes = 10
#     input_shape = (28, 28, 1)
#     print("Loading MNIST data...")
#     (x_train, y_train_cat), (x_test, y_test_cat) = keras.datasets.mnist.load_data()

#     print("Preprocessing images and labels...")
#     x_train = x_train.astype("float32") / 255.0
#     x_test = x_test.astype("float32") / 255.0
#     x_train = np.expand_dims(x_train, axis=-1)
#     x_test = np.expand_dims(x_test, axis=-1)

#     y_train = keras.utils.to_categorical(y_train_cat, num_classes)
#     y_test = keras.utils.to_categorical(y_test_cat, num_classes)

#     data_train = [x_train, y_train]
#     data_test = [x_test, y_test]

#     # ------------------------------------------------------------------
#     # Step 2. Build the measurement models exactly as in the original tutorial.
#     # ------------------------------------------------------------------
#     # View 1: an image encoder built from Conv2D, pooling, and dense layers.
#     # View 2: an identity mapping because labels are already one-hot encoded.
#     print("Building measurement models...")
#     image_encoder = keras.Sequential(name="mnist_image_encoder")
#     image_encoder.add(layers.InputLayer(input_shape=input_shape, name="mnist_image_in"))
#     image_encoder.add(
#         layers.Conv2D(
#             32,
#             (3, 3),
#             activation="relu"
#         )
#     )
#     image_encoder.add(layers.MaxPooling2D((2, 2)))
#     image_encoder.add(
#         layers.Conv2D(
#             64,
#             (3, 3),
#             activation="relu"
#         )
#     )
#     image_encoder.add(layers.MaxPooling2D((2, 2)))
#     image_encoder.add(layers.Flatten())
#     image_encoder.add(layers.Dense(128, activation="relu"))
#     image_encoder.add(layers.Dropout(rate=0.1))

#     labels_input = keras.Input(shape=(num_classes,), name="mnist_label_in")
#     labels_output = layers.Activation("linear", name="mnist_label_id")(labels_input)
#     label_encoder = keras.Model(labels_input, labels_output, name="mnist_label_encoder")

#     # ------------------------------------------------------------------
#     # Step 3. Configure the StructuralModel with the classic two-view setup.
#     # ------------------------------------------------------------------
#     # The 2x2 adjacency matrix encodes a symmetric relationship: each view
#     # connects to the other.  ``tot_num`` tells FactorLayer how many samples
#     # exist in the full dataset so it can keep running covariance statistics.
#     adjacency = np.array([[0, 1], [1, 0]], dtype="float32")
#     total_examples = x_train.shape[0]

#     structural_model = StructuralModel(
#         Path=adjacency,
#         model_list=[image_encoder, label_encoder],
#         regularizer_list=[None, None],
#         tot_num=total_examples,
#         ndims=9,
#         orthogonalization="Moore-Penrose",
#         momentum=0.95,
#         epsilon=1e-4,
#         train_DLV=False,
#     )

#     # Give each view its own Adam optimiser to keep learning rates independent.
#     print("Compiling StructuralModel...")
#     image_optimizer = Adam(learning_rate=1e-4)
#     label_optimizer = Adam(learning_rate=1e-4)
#     structural_model.compile(optimizer=[image_optimizer, label_optimizer])

#     # ------------------------------------------------------------------
#     # Step 4. Train the structural model then evaluate on held-out data.
#     # ------------------------------------------------------------------
#     # Keras 3 still accepts lists-of-arrays for multi-view training.  We keep a
#     # small validation split to monitor convergence and watch the redundancy
#     # metric during training.
#     print("Training...")
#     history = structural_model.fit(
#         data_train,
#         batch_size=256,
#         epochs=20,
#         verbose=True,
#         validation_split=0.1,
#     )

#     print("Evaluating on the test split...")
#     metrics = _evaluate_structural_model(structural_model, data_test)
#     for metric_name, metric_value in metrics.items():
#         print(f"{metric_name}: {metric_value:.6f}")

#     # ------------------------------------------------------------------
#     # Step 5. Inspect the learned latent representations.
#     # ------------------------------------------------------------------
#     # ``predict`` returns a (n_samples, ndims, n_views) tensor of deep latent
#     # variables.  We also grab the standalone encoder output for the image view
#     # so we can visualise a subset with t-SNE.
#     print("Predicting latent representations...")
#     latent = structural_model.predict(data_test, verbose=False)
#     image_latent = structural_model.model_list[0].predict(data_test[0], verbose=False)

#     tsne = TSNE(n_components=2, random_state=42)
#     rng = np.random.default_rng(42)
#     sample_indices = rng.choice(image_latent.shape[0], size=min(200, image_latent.shape[0]), replace=False)
#     tsne_projection = tsne.fit_transform(image_latent[sample_indices])

#     print("Latent tensor shape:", latent.shape)
#     print("t-SNE projection shape:", tsne_projection.shape)
#     print("Training history keys:", list(history.history))

############ Tutorial 1 ############

# import all necessary packages required for this tutorial
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import numpy as np
import deep_lvpm
import keras
from keras import layers
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from deep_lvpm.models.StructuralModel import StructuralModel ## Here, we import the main StructuralModel class used in deep-lvpm

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train_cat), (x_test, y_test_cat) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train_cat, num_classes)
y_test = keras.utils.to_categorical(y_test_cat, num_classes)


data_train_list = [x_train, y_train]
data_test_list = [x_test, y_test]

MNIST_image_model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5)

    ]
)

data_input = keras.Input(shape = (10,))
data_output = keras.layers.Activation('linear', name='identity')(data_input)
MNIST_label_model=keras.Model(inputs=data_input,outputs=data_output)
  

# Define a model list, which will then be used as an input to the DLVPM model
model_list = [MNIST_image_model, MNIST_label_model] 

# Here, we define a new adjacency matrix, which defines which data views to connect
Path = tf.constant([[0,1],
            [1,0]])

regularizer_list = [None,None] ## regularizer_list 

ndims = 9 # the number of DLVs we wish to extract
tot_num = x_train.shape[0] # the total number of samples, which is used for internal normalisation
batch_size = 256
epochs = 20

DLVPM_Model = StructuralModel(Path, model_list, regularizer_list, tot_num, ndims, orthogonalization="zca", train_DLV=True)

optimizer_list = [keras.optimizers.Adam(learning_rate=1e-5),keras.optimizers.Adam(learning_rate=1e-5)]

DLVPM_Model.compile(optimizer=optimizer_list)

DLVPM_Model.fit(data_train_list, batch_size=batch_size, epochs=epochs,verbose=True, validation_split=0.1)

metrics = DLVPM_Model.evaluate(data_test_list)

DLVs = DLVPM_Model.predict(data_test_list)

Cmat1 = np.corrcoef(DLVs[:,0,:].T)

image_DLVs = DLVPM_Model.model_list[0].predict(data_test_list[0])

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

## Here, we randomy select 100 examples for plotting
random_indices = np.random.choice(image_DLVs.shape[0], size=100, replace=False)

image_DLVs_plot = image_DLVs[random_indices,:]
y_test_plot = y_test[random_indices,:]

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(image_DLVs_plot)

# Plot
plt.figure(figsize=(12, 8))

for i in range(y_test_plot.shape[1]):
    points = tsne_results[y_test_plot[:, i] == 1]
    plt.scatter(points[:, 0], points[:, 1], label=f'Category {i+1}')

plt.title('t-SNE projection of the dataset')
plt.legend()
