
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
from deep_lvpm.model import StructuralModel ## Here, we import the main StructuralModel class used in deep-lvpm

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
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.1)

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

DLVPM_Model = StructuralModel(Path, model_list, regularizer_list, tot_num, ndims, orthogonalization="zca", train_DLV=False)

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
