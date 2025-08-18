# # import all necessary packages required for this tutorial
# import tensorflow as tf
# import numpy as np
# import deep_lvpm
# from tensorflow import keras
# from keras import layers
# import numpy as np
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from deep_lvpm.models.StructuralModel import StructuralModel ## Here, we import the main StructuralModel class used in deep-lvpm

# # Model / data parameters
# num_classes = 10
# input_shape = (28, 28, 1)

# # Load the data and split it between train and test sets
# (x_train, y_train_cat), (x_test, y_test_cat) = keras.datasets.mnist.load_data()

# # Scale images to the [0, 1] range
# x_train = x_train.astype("float32") / 255
# x_test = x_test.astype("float32") / 255
# # Make sure images have shape (28, 28, 1)
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)

# print("x_train shape:", x_train.shape)
# print(x_train.shape[0], "train samples")
# print(x_test.shape[0], "test samples")

# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train_cat, num_classes)
# y_test = keras.utils.to_categorical(y_test_cat, num_classes)

# data_train_list = [x_train, y_train]
# data_test_list = [x_test, y_test]

# MNIST_image_model = keras.Sequential(
#     [
#         keras.Input(shape=input_shape),
#         layers.Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-5)),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-5)),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Flatten(),
#         layers.Dense(100, activation='relu'),
#         layers.Dropout(0.5)

#     ]
# )

# data_input = keras.Input(shape = 10)
# MNIST_label_model=keras.Model(inputs=data_input,outputs=data_input)
  
# model_list = [MNIST_image_model, MNIST_label_model] 

# # Here, we define a new adjacency matrix, which defines which data views to connect
# Path = tf.constant([[0,1],
#             [1,0]])

# regularizer_list = [None,None] ## regularizer_list 

# ndims = 9 # the number of DLVs we wish to extract
# tot_num = x_train.shape[0] # the total number of samples, which is used for internal normalisation
# batch_size = 32
# epochs = 20

# DLVPM_Model = StructuralModel(Path, model_list, regularizer_list, tot_num, ndims, epochs, batch_size)

# optimizer_list = [keras.optimizers.Adam(learning_rate=1e-4),keras.optimizers.Adam(learning_rate=1e-4)]

# DLVPM_Model.compile(optimizer=optimizer_list)

# DLVPM_Model.fit(data_train_list, batch_size=batch_size, epochs=epochs,verbose=True, validation_split=0.1)

# metrics = DLVPM_Model.evaluate(data_test_list)

# DLVs = DLVPM_Model.predict(data_test_list)

# #DLVPM_Model.save('output_folder/DLVPM_Model.keras')

# image_DLVs = DLVPM_Model.model_list[0].predict(data_test_list[0])

# ## Here, we randomy select 100 examples for plotting
# random_indices = np.random.choice(image_DLVs.shape[0], size=100, replace=False)

# image_DLVs_plot = image_DLVs[random_indices,:]
# y_test_plot = y_test[random_indices,:]

# # Apply t-SNE
# tsne = TSNE(n_components=2, random_state=42)
# tsne_results = tsne.fit_transform(image_DLVs_plot)

# # Plot
# plt.figure(figsize=(12, 8))

# for i in range(y_test_plot.shape[1]):
#     points = tsne_results[y_test_plot[:, i] == 1]
#     plt.scatter(points[:, 0], points[:, 1], label=f'Category {i+1}')

# plt.title('t-SNE projection of the dataset')
# plt.legend()
# plt.savefig('/Users/ing/Downloads/figure_out.png')
# plt.show()

# Keras 3 / multi-backend MNIST demo for StructuralModel
# No tf imports; model/layers defined with pure keras APIs.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import keras
from keras import layers, ops

# If your deep_lvpm package is on PYTHONPATH and contains the ops-based classes:
from deep_lvpm.models.StructuralModel import StructuralModel

# ----------------------------
# 1) Data
# ----------------------------
num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train_cat), (x_test, y_test_cat) = keras.datasets.mnist.load_data()

# Scale to [0,1] and add channel dim
x_train = (x_train.astype("float32") / 255.0)[..., None]
x_test  = (x_test.astype("float32") / 255.0)[..., None]

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# One-hot labels
y_train = keras.utils.to_categorical(y_train_cat, num_classes)
y_test  = keras.utils.to_categorical(y_test_cat, num_classes)

data_train_list = [x_train, y_train]
data_test_list  = [x_test,  y_test]

# ----------------------------
# 2) Measurement models (per view)
# ----------------------------
# View 1: image encoder (ends with our Factor/ZCA layer added by StructuralModel)
MNIST_image_model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu",
                      kernel_regularizer=keras.regularizers.l1_l2(1e-5, 1e-5)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu",
                      kernel_regularizer=keras.regularizers.l1_l2(1e-5, 1e-5)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(100, activation="relu"),
        layers.Dropout(0.5),
        # (Factor/ZCA layer is appended automatically by StructuralModel)
    ]
)

# View 2: labels passthrough (identity)
data_input = keras.Input(shape = (10,))
data_output = keras.layers.Activation('linear', name='identity')(data_input)
MNIST_label_model=keras.Model(inputs=data_input,outputs=data_output)

model_list = [MNIST_image_model, MNIST_label_model]

# ----------------------------
# 3) Graph (adjacency) and hyperparams
# ----------------------------
# Connect images <-> labels (bidirectional)
Path = np.array([[0, 1],
                 [1, 0]], dtype="int32")

regularizer_list = [None, None]
ndims = 9                        # number of DLVs to extract
tot_num = x_train.shape[0]       # used internally for scaling
batch_size = 256
epochs = 20

# ----------------------------
# 4) Build StructuralModel (ops-based version)
# ----------------------------
DLVPM_Model = StructuralModel(
    Path=Path,
    model_list=model_list,
    regularizer_list=regularizer_list,
    tot_num=tot_num,
    ndims=ndims,
    orthogonalization="Moore-Penrose",  # uses FactorLayer; switch to "zca" to use ZCALayer
    momentum=0.95,
    epsilon=1e-4,
    train_DLV=False,                    # compute targets in inference mode (as in your original)
)

# Per-view optimizers (or pass a single optimizer to share)
optimizer_list = [
    keras.optimizers.Adam(learning_rate=1e-4),
    keras.optimizers.Adam(learning_rate=1e-4),
]
DLVPM_Model.compile(optimizer=optimizer_list)

# ----------------------------
# 5) Train / Evaluate / Predict
# ----------------------------
DLVPM_Model.fit(
    data_train_list,
    batch_size=batch_size,
    epochs=epochs,
    verbose=True,
    validation_split=0.1,
)

metrics = DLVPM_Model.evaluate(data_test_list, verbose=False)
print("Test metrics:", metrics)

DLVs = DLVPM_Model.predict(data_test_list, verbose=False)  # (n_samples, ndims, n_views)

# Example: DLVs for the image view (view 0)
image_DLVs = DLVPM_Model.model_list[0].predict(data_test_list[0], verbose=False)

# ----------------------------
# 6) Visualize with t-SNE
# ----------------------------
np.random.seed(42)
random_indices = np.random.choice(image_DLVs.shape[0], size=100, replace=False)
image_DLVs_plot = image_DLVs[random_indices, :]
y_test_plot = y_test[random_indices, :]

tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(image_DLVs_plot)

plt.figure(figsize=(12, 8))
for i in range(y_test_plot.shape[1]):
    points = tsne_results[y_test_plot[:, i] == 1]
    if len(points):
        plt.scatter(points[:, 0], points[:, 1], label=f"Category {i}")
plt.title("t-SNE projection of image DLVs")
plt.legend()
plt.tight_layout()
plt.show()



