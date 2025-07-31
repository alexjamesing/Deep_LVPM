
############ Tutorial 1 ############

# import all necessary packages required for this tutorial
import tensorflow as tf
import numpy as np
import deep_lvpm
from tensorflow import keras
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
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-5)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-5)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
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
epochs = 10

DLVPM_Model = StructuralModel(Path, model_list, regularizer_list, tot_num, ndims)

optimizer_list = [keras.optimizers.Adam(learning_rate=1e-4),keras.optimizers.Adam(learning_rate=1e-4)]

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


####### Tutorial 2 #########


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers
from importlib import resources

import deep_lvpm as DLVPM
from deep_lvpm.models.StructuralModel import StructuralModel

tf.config.run_functions_eagerly(False)   # keep graph mode for performance

with resources.as_file(resources.files("deep_lvpm.data") /
                       "Lung_multiomics_sample_train.npz") as f:
    arrays = np.load(f)
    rnaseq      = arrays["rnaseq"]
    snv         = arrays["snv"]
    methylation = arrays["methylation"]
    mirna       = arrays["mirna"]
    histo20     = arrays["histo20"]

X_arr = [histo20, rnaseq, methylation, mirna, snv]   # preserve this order!



def residual_block(
        input_dim: int,
        kernel_reg_l1: float = 0.01,
        kernel_reg_l2: float = 0.01,
        dropout_rate: float = 0.5,
        name: str = "residual_block"
    ) -> tf.keras.Model:
    """
    Builds a simple fully‑connected residual block.

    Parameters
    ----------
    input_dim : int
        Number of features in the (flat) input vector.
    kernel_reg_l1 : float, optional
        L1 regularisation factor for dense layers (default 0.01).
    kernel_reg_l2 : float, optional
        L2 regularisation factor for dense layers (default 0.01).
    dropout_rate : float, optional
        Drop‑out probability applied after the residual connection (default 0.5).
    name : str, optional
        Name for the returned `tf.keras.Model`.

    Returns
    -------
    tf.keras.Model
        A Keras `Model` representing the residual block.
    """
    # -------- input --------
    inputs = tf.keras.Input(shape=(input_dim,), name=f"{name}_in")

    # -------- first linear projection --------
    x = tf.keras.layers.Dense(
        input_dim,
        activation="linear",
        kernel_initializer=tf.keras.initializers.Identity(),
        kernel_regularizer=tf.keras.regularizers.l1_l2(
            l1=kernel_reg_l1, l2=kernel_reg_l2
        ),
        name=f"{name}_dense1",
    )(inputs)

    # -------- normalise & non‑linear activation --------
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    x = tf.keras.layers.ReLU(name=f"{name}_relu")(x)

    # -------- second linear projection --------
    x = tf.keras.layers.Dense(
        input_dim,
        activation="linear",
        kernel_initializer=tf.keras.initializers.Identity(),
        kernel_regularizer=tf.keras.regularizers.l1_l2(
            l1=kernel_reg_l1, l2=kernel_reg_l2
        ),
        name=f"{name}_dense2",
    )(x)

    # -------- residual connection --------
    x = tf.keras.layers.Add(name=f"{name}_add")([inputs, x])

    # -------- optional regularisation --------
    x = tf.keras.layers.Dropout(dropout_rate, name=f"{name}_drop")(x)

    # -------- wrap into a model --------
    return tf.keras.Model(inputs=inputs, outputs=x, name=name)


model_list = [
    residual_block(histo20.shape[1], name="histo20_enc"),
    residual_block(rnaseq.shape[1],  name="rnaseq_enc"),
    residual_block(methylation.shape[1], name="meth_enc"),
    residual_block(mirna.shape[1],   name="mirna_enc"),
    residual_block(snv.shape[1],     name="snv_enc"),
]


ndims = 5        # number of latent factors

Path = np.array([
    # F₁ F₂ F₃ F₄ F₅
    [0, 1, 0, 0, 0],  # F₁ ← F₂
    [1, 0, 1, 1, 1],  # F₂ ← F₁,F₃,F₄,F₅
    [0, 1, 0, 0, 0],  # F₃ ← F₂
    [0, 1, 0, 0, 0],  # F₄ ← F₂
    [0, 1, 0, 0, 0],  # F₅ ← F₂
], dtype="float32")

batch_size  = 256
epochs      = 300
total_steps = int(rnaseq.shape[0] / batch_size) * epochs

init_lr, final_lr = 1e-4, 1e-5

lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=init_lr,
    decay_steps=total_steps,
    decay_rate=final_lr / init_lr,
    staircase=False
)

tot_num = rnaseq.shape[0] ## This is the total number of samples under analysis and is needed by DLVPM


from tensorflow.keras import regularizers

regularizer_list = [regularizers.L1L2(l1=0.01, l2=0.01),regularizers.L1L2(l1=0.01, l2=0.01),regularizers.L1L2(l1=0.01, l2=0.01),regularizers.L1L2(l1=0.01, l2=0.01),regularizers.L1L2(l1=0.01, l2=0.01)] ## These regularizers are applied to the final "projection" layer of the DLVPM model, used internally

DLVPM_Structural_instance = StructuralModel(Path, model_list, regularizer_list, tot_num, ndims, momentum=0.95,epsilon=0.001, orthogonalization='Moore-Penrose')

opt_list = [tf.keras.optimizers.Adam(learning_rate=lr_schedule),tf.keras.optimizers.Adam(learning_rate=lr_schedule),tf.keras.optimizers.Adam(learning_rate=lr_schedule),tf.keras.optimizers.Adam(learning_rate=lr_schedule),tf.keras.optimizers.Adam(learning_rate=lr_schedule)]
DLVPM_Structural_instance.compile(optimizer=opt_list)


DLVPM_Structural_instance.fit(X_arr, batch_size=batch_size, epochs=epochs,verbose=True)
mean_corr = DLVPM_Structural_instance.evaluate(X_arr)

print('The mean correlation between data-types connected by the path model is r=' + str(mean_corr[1]))


with resources.as_file(resources.files("deep_lvpm.data") /
                       "Lung_multiomics_sample_test.npz") as f:
    arrays = np.load(f)
    rnaseq_test      = arrays["rnaseq"]
    snv_test         = arrays["snv"]
    methylation_test = arrays["methylation"]
    mirna_test       = arrays["mirna"]
    histo20_test     = arrays["histo20"]

X_arr_test = [histo20_test, rnaseq_test, methylation_test, mirna_test, snv_test]   # Here, is the full test dataset list
mean_corr_test = DLVPM_Structural_instance.evaluate(X_arr_test)

print('The mean correlation between data-types connected by the path model is r=' + str(mean_corr_test[1]))

test_DLVs = DLVPM_Structural_instance.predict(X_arr_test) ## Here, we obtain the full set of test_DLVs

## Associations between the first set of DLVs are:
print(np.corrcoef(test_DLVs[:,0,:].T))
## Associations between the second set of DLVs are:
print(np.corrcoef(test_DLVs[:,1,:].T))
