
![Alt text](dlvpm_logo_final.png)

**Deep LVPM**

Deep Latent Variable Path Modelling (DLVPM) is a method for path/structural equation modelling utilising deep neural networks. The aim of the method is to connect different data types together via sets of orthogonal deep latent variables (DLVs). 

The user must specify a path model defining which data types should be linked by the DLVPM model, along with a neural network model for each data-view, which is then used to optimise associations/linkages between DLVs derived from each data-type.

The implementation of this method is built around a custom keras/tensorflow model called 'StructuralModel', which utilises several custom keras/tensorflow layers for constructing Deep Latent Variables (DLVs) from input data. Using the high-level keras API, we can define new DLVPM models in just a few lines of code. Once a Deep LVPM model is defined, standard keras/tensorflow functions such as model.fit() and model.evaluate() can be used to train  and test DLVPM models. Users who are unfamiliar with keras/tensorflow can find documentation on these projects here: https://www.tensorflow.org/guide/keras


**Installation**

This package was most recently tested on Tensorflow 2.16.2, which is compatible with Python 3.12. This version of tensorflow will be installed automatically with Deep_LVPM.


~~~

conda create -n myenv python=3.12

~~~

Activate the environment:


~~~

conda activate myenv

~~~

Then, you can download and install this package using the following command (requires Git):


~~~

pip install git+https://github.com/alexjamesing/Deep_LVPM.git

~~~


**Example Application**

In the tutorial below, we give a very simple example of how DLVPM can be used. This tutorial uses the MNIST dataset (the hello world! of machine learning!!), designed to give the user an idea of how a DLVPM StructuralModel can be instantiated, then trained, tested and saved. 

In this example, the DLVPM method constructs deep latent variables (DLVs) that are highly correlated between image and categorical data. This is the simplest kind of DLVPM model, with just two data-types. It should be possible to train this model on a modern laptop, using only CPU, in just a few minutes. 

First, we need to download the MNIST dataset, and prepare it for use with the DLVPM model:

~~~

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

~~~

We then enter the training and testing data into lists, for later use in training and testing the DLVPM model:

~~~

data_train_list = [x_train, y_train]
data_test_list = [x_test, y_test]

~~~

We then need to define keras/tensorflow models to be used to process data from the different data-views. Here, we use a convolutional neural network to process the image data. For the label data, we only define the input shape, as further processing of categorical labels is not required here. Models can be defined using any of the keras APIs i.e. functional, sequential, or using model subclassing.

~~~

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
MNIST_label_model=keras.Model(inputs=data_input,outputs=data_input)
  
~~~

We then define parameters for specifying the StructuralModel, which is the core class of DLVPM. The two models we defined above should be entered as lists. In DLVPM parlance, these models are referred to as 'measurement models'. model_list should be of the same length as the number of data-types we wish to connect.

~~~

# Define a model list, which will then be used as an input to the DLVPM model
model_list = [MNIST_image_model, MNIST_label_model] 

~~~

We must also define a path model, in the form of an adjacency matrix that defines the data views that are connected to one another. In this case, the path model is trivial, we are only connecting two data-types, so it is written as:

~~~

# Here, we define a new adjacency matrix, which defines which data views to connect
Path = tf.constant([[0,1],
            [1,0]])

~~~

The model list and path model are used as inputs to the structural model. The structural model also takes a number of other inputs (see section on StructuralModel for more information).

~~~

regularizer_list = [None,None] ## regularizer_list 

ndims = 9 # the number of DLVs we wish to extract
tot_num = x_train.shape[0] # the total number of samples, which is used for internal normalisation
batch_size = 256
epochs = 10

DLVPM_Model = StructuralModel(Path, model_list, regularizer_list, tot_num, ndims)

~~~

It is then necessary to compile the model before training. Here, we must define an optimizer for each data view to be connected. These are entered into the model as a list:

~~~

optimizer_list = [keras.optimizers.Adam(learning_rate=1e-4),keras.optimizers.Adam(learning_rate=1e-4)]

DLVPM_Model.compile(optimizer=optimizer_list)

~~~

We then run model training using the fit function:

~~~

DLVPM_Model.fit(data_train_list, batch_size=batch_size, epochs=epochs,verbose=True, validation_split=0.1)

~~~

We can then evaluate the model:

~~~

metrics = DLVPM_Model.evaluate(data_test_list)

~~~

The first metric in the list that is produced here is the mean squared error. The second metric is the mean pearson's correlation coefficient between models/data connected by the path model.

We can use the predict function to obtain the deep latent variables for different data views. 

~~~

DLVs = DLVPM_Model.predict(data_test_list)

~~~

This function gives the Deep Latent Variables (DLVs) in the form of a 3-D tensor of size tot_num x ndims x len(model_list). We can then examine the association between individual latent variables. For example:

~~~

Cmat1 = np.corrcoef(DLVs[:,0,:].T)

~~~

Gives a 2-D matrix of associations between the first of the DLVs, for each data view. Associations are high as the DLVPM algorithm is designed to optimise associations between DLVs constructed from different data types. 


Once the model has been trained, we can save it for future use using:

~~~

DLVPM_Model.save('/output_folder/DLVPM_Model.keras')

~~~

Note that, as the model utilises custom keras/tensorflow models and layers, it is necessary to save the model in the new .keras format. Problems may arise from saving the model in older formats utilised by keras/tensorflow.

We can obtain DLVs from a single data-type by calling that model in the model_list, e.g.:

~~~

image_DLVs = DLVPM_Model.model_list[0].predict(data_test_list[0])

~~~

If we apply a tsne to these image DLVs, we can see that the DLVPM model has learned a mapping between the image data and the image labels, which means that DLVs naturally segregate image labels in a tnse plot:

~~~

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
plt.savefig('/output_folder/figure_out.png')
plt.show()

~~~

In the text below, we give an explanation of each of the custom model and layer types that are used in the DLVPM toolbox.

# StructuralModel

`StructuralModel` is a custom TensorFlow/Keras class designed for uncovering deep latent variables that represent correlated factors between different data types. The `StructuralModel` utilizes a binary adjacency matrix to define connections between different data views. It incorporates multiple Keras models, each corresponding to a specific data view, and integrates them to optimize associations using DLVs.

This model also depends on the custom layers `FactorLayer` and `ZCALayer`, these layers are called internally using this method. The layer that is used is determined by whether the user selects 'Moore-Penrose' or 'zca' for the orthogonalisztion parameter.

# Attributes

- **Path**: Binary adjacency matrix defining connections between data views.
- **model_list**: List of Keras models, one for each data view.
- **regularizer_list**: List of regularizers applied to projection layers in each data-view model.
- **tot_num**: Total number of features across all data batches.
- **ndims**: Number of orthogonal latent variables to be constructed.
- **orthogonalization (optional)**: Specifies the orthogonalization method ('Moore-Penrose' or 'ZCA'). Defaults to Moore-Penrose.
- **momentum (optional)**: The momentum defines how quickly global parameters such as means and correlation matrices are updated. Defaults to 0.95
- **epsilon (optional)**: epsilon is a small constant that is added for numeric stability during batch updates. Defaults to 1e-4.

# Callable Methods

StructuralModel inherits from tf.keras.Model. This means that the core functions used for model compilation (model.compile()), training (model.fit()) and testing (model.evaluate() and model.predict()) are the same. However, in some cases, these functions behave slightly differently. For example, model.compile() requires a list of optimizers as input arguments rather than the single optimizer that is standard when defining a keras/tensorflow model. More information about these functions, and the arguments they can take, can be found in the official tensorflow/keras documentation.

- **compile(optimizer_list)**: Configures the model for training with specified optimizers. This function requires a list of optimizers, one for each data-type/measurement model. As is usually the case, compile should be called when an instance of 'StrucutralModel' has been instantiated.

Example usage:

~~~

optimizer_list = [tf.keras.optimizers.Adam(learning_rate=1e-4),tf.keras.optimizers.Adam(learning_rate=1e-3),tf.keras.optimizers.Adam(learning_rate=1e-4)] ## Here, we have three optimizers, one for each measurement model 

struct_model.compile(optimizer_list)

~~~

- **fit**: In keras/tensorflow, the .fit method is used to train the model on the training data. In the present investigation, the model takes a data-list as input. The model can also take a data-generator that outputs a list. More information regarding additional arguments that can be taken by model.fit() can be found in the keras/tensorflow documentation.

Example usage:

~~~

X = [data1, data2, data3] ## data is entered as a list

struct_model.fit(X) ## fit the model

~~~

- **evaluate**: In keras/tensorflow, the .evaluate() function is used to evaluate model losses and metrics on some input data.

Example usage:

~~~

[losses, metrics] = struct_model.evaluate(X)

~~~

- **predict**: In keras/tensorflow, the .predict() function produces outputs for a particular model. In the case of DLVPM, .predict will produce DLVs from all measurement models. The tensor produced here will be of shape n x ndims x nviews where n is the total number of samples, ndim is the number of DLVs extracted and nviews is the number of data-types included in the analysis.

Example usage:

~~~

X = [data1, data2, data3] ## data is entered as a list

DLVs = struct_model.fit(X) ## fit the model

~~~

It is worth noting that we can also obtain DLVs from individual measurement models by calling predict on them, calling them from their list index in model_list. For example, if we would like to obtain results for the third data-type entered into the structural model, we could obtain DLVs from this model using:

~~~

DLVs_data_3 = struct_model.model_list[3].predict(X[3]) # This command returns DLVs from the third measurement model.

~~~

- **calculate_corrmat**: calculate_corrmat is a custom DLVPM function that is designed to calculate associations between DLVs that have been output by model.predict(). This function will output a list of correlation matrices of length equal to ndims, with one association matrix for each DLV. 


# Tracking Metrics

- **loss_tracker_total**: Tracks total loss during training.
- **corr_tracker**: Tracks correlation metrics during training.
- **loss_tracker_mse**: Tracks mean squared error loss during training.

# Internal Methods

These internal methods will generally not be called directly by the user. Rather, they are called by other methods. E.g. train_step will be called for each batch during model.fit.

- **__init__(...)**: Initializes the `StructuralModel` instance.
- **add_DLVPM_layer(...)**: Adds a `FactorLayer` or `ZCALayer` to a given model.
- **call(inputs)**: Runs data through each sub-model for the data views.
- **train_step(inputs)**: Performs a training step, updating model weights.
- **test_step(inputs)**: Evaluates the model on test data.
- **mse_loss(...)**: Calculates mean squared error loss.
- **corr_metric(...)**: Calculates the correlation metric.
- **get_config()**: Returns the configuration of the model.
- **from_config(config)**: Creates an instance of the model from a configuration.
- **get_compile_config()**: Retrieves the optimizer configurations.
- **compile_from_config(config)**: Compiles the model using a specified configuration.

# Input

- **inputs**: A list of input tensors, each representing a different data view.

# Example Usage

The `StructuralModel` is particularly useful in scenarios involving multi-view data analysis, where establishing connections and correlations between different data types is crucial. This model effectively integrates multiple sub-models, each tailored to a specific data view, and employs advanced techniques like orthogonalization to uncover the underlying relationships between these views.

~~~

import tensorflow as tf
from custom_models import StructuralModel

# Example usage of StructuralModel
model = StructuralModel(
    Path=adjacency_matrix,
    model_list=[model1, model2, ...],
    regularizer_list=[regularizer1, regularizer2, ...],
    tot_num=total_features,
    ndims=number_of_latent_variables,
    orthogonalization='Moore-Penrose'
)

model.compile(optimizer=tf.keras.optimizers.Adam())

~~~~

# FactorLayer

This document provides detailed information about the `FactorLayer`, a custom TensorFlow/Keras layer that is part of the DLVPM (Deep Latent Variable Projection Models) toolbox. The `FactorLayer` is designed to generate orthogonal factors that are highly correlated between different data views. This layer is called internally by StructuralModel. It is the option that is called by DLVPM when orthgonalisation = 'Moore-Penrose' is selected when the 'StructuralModel' is instantiated. This is also the default option of DLVPM.

# Overview

The `FactorLayer` performs three primary operations:

1. **Batch Normalization**: Normalizes inputs based on batch statistics.
2. **Orthogonalization**: Orthogonsalises Deep Latent Variable (DLV) outputs from the layer
3. **Linear Projection**: Projects the neural network outputs into a space where they correlate with outputs from other data views.

The layer behaves differently during training and testing, similar to other adaptive layers like batch normalization.

# Attributes

- **kernel_regularizer**: Regularizer function applied to the projection layer's kernel weights. Can be `None` or a TensorFlow/Keras regularizer object.
- **epsilon**: A small constant (default: 1e-6) added to variance in batch normalization to avoid division by zero.
- **momentum**: Momentum (default: 0.95) for moving average and moving variance in batch normalization.
- **tot_num**: Total number of samples used in training. Optimizes scaling of covariance matrices.
- **ndims**: Number of Deep Latent Variable (DLV) dimensions to extract.
- **run**: TensorFlow variable tracking the number of runs to initialize moving variables on the first call.

# Constructor Arguments

- **kernel_regularizer**: Optional. Regularizer for the projection layer's kernel weights.
- **epsilon**: Optional. Offset for batch normalization.
- **momentum**: Optional. Momentum for moving averages in batch normalization.
- **tot_num**: Optional. Total number of training samples.
- **ndims**: Optional. Number of DLV dimensions.
- **run**: Optional. Initial run tracker value, defaults to 0.
- **kwargs**: Additional keyword arguments inherited from `tf.keras.layers.Layer`.

# Methods

- **build(input_shape)**: Initializes layer weights and other necessary variables.
- **call(inputs, training=False)**: Forward pass of the `FactorLayer`.
- **get_config()**: Returns the configuration of the layer.
- **from_config(config)**: Creates a layer instance from its config.

# Input

- **inputs**: A single tensor used for projecting to other data views and identifying highly correlated factors between data views.

# Usage

The `FactorLayer` is designed to be used at the end of DLVPM models. It takes a single input tensor and processes it through batch normalization, orthogonalization, and linear projection. The layer is adaptable, functioning differently during training and testing to optimize its performance. The layer is called internally by the StructuralModel class.

# ZCA Layer

This document provides an overview of the `ZCALayer`, a custom TensorFlow/Keras layer in the DLVPM (Deep Latent Variable Projection Models) toolbox. The 'ZCALayer' is used internally by the StructuralModel class. This layer is selected when we use orthogonalization='zca'. Unlike the 'FactorLayer', which carries out orthgonalization within the layer, when orthogonalization='zca' is selected when instantiating 'StructuralModel', orthogonalization is carried out within the StructuralModel class. 

## Overview

The `ZCALayer` performs a series of operations to process input data:

- **Batch Normalization:** Normalizes the inputs based on batch statistics.
- **Linear Projection:** Projects the neural network outputs into a space correlating with outputs from other data views.

These operations follow the order: batch normalization > orthogonalization > linear projection. The layer functions differently during training and testing, akin to layers like batch normalization.

## Attributes

- **kernel_regularizer:** Regularizes the projection layer's kernel weights. Can be a TensorFlow/Keras regularizer object or None.
- **epsilon:** Small constant (default: 1e-6) added to variance in batch normalization to avoid division by zero.
- **momentum:** Momentum (default: 0.95) for moving average and moving variance in batch normalization.
- **diag_offset:** Small constant added to the diagonal of covariance matrices to ensure invertibility.
- **tot_num:** Total number of training samples. Used for optimal scaling of covariance matrices.
- **ndims:** Number of Deep Latent Variable (DLV) dimensions to extract.
- **run:** TensorFlow variable tracking the number of runs for initializing moving variables.

## Constructor Arguments

- **kernel_regularizer:** Regularizer function for the projection layer's kernel weights.
- **epsilon:** Offset value for batch normalization.
- **momentum:** Momentum for moving averages in batch normalization.
- **diag_offset:** Offset added to covariance matrix diagonal.
- **tot_num:** Total number of training samples.
- **ndims:** Number of DLV dimensions.
- **run:** Initial value for the run tracker.

## Methods

- **build(input_shape):** Initializes layer weights and necessary variables.
- **call(inputs, training=False):** Forward pass of the `ZCALayer`.
- **get_config():** Returns the configuration of the layer.
- **from_config(config):** Creates a layer instance from its configuration.

## Input

- **inputs:** A single tensor used for projecting to other data views and identifying factors that are highly correlated between them.

## Usage

The `ZCALayer` is intended to be positioned at the end of DLVPM models. The layer is called internally by the 'StructuralModel' class.

## Confound Layer

This document provides a comprehensive overview of the `ConfoundLayer`, a custom TensorFlow/Keras layer included in the DLVPM (Deep Latent Variable Projection Models) toolbox. The primary function of this layer is to orthogonalize data inputs with respect to a set of input confounds, making it a vital component for handling confounding variables in neural network models.

## Overview

The `ConfoundLayer` is designed to perform orthogonalization of one set of inputs relative to another, essentially removing the influence of confounding variables from the primary data inputs. This process is crucial in scenarios where the data might be influenced by external, non-relevant factors.

## Call Inputs

- **input[0]:** The primary data input that needs to be orthogonalized. This will usually be data that has been processed by earlier 
layers of the neural network
- **input[1]:** The set of confounding variables against which `input[0]` is orthogonalized.

## Attributes

- **tot_num:** Total number of samples over which training is conducted.
- **epsilon:** Small offset value used during batch normalization (default: 1e-4).
- **momentum:** Momentum for updating covariance matrices during training (default: 0.95).
- **diag_offset:** Offset added to the diagonal of the covariance matrix to ensure invertibility (default: 1e-6).
- **run:** TensorFlow variable tracking the number of runs for initialization.

## Constructor Arguments

- **tot_num (int):** Total number of training samples.
- **epsilon (float):** Offset for batch normalization.
- **momentum (float):** Momentum for covariance matrices.
- **diag_offset (float):** Offset added to the diagonal of the covariance matrix.
- **run (int):** Initial value for the run tracker.

## Methods

- **build(input_shape):** Initializes layer weights and necessary variables.
- **call(inputs, training=None):** Performs orthogonalization of inputs during training and testing.
- **get_config():** Returns the configuration of the layer.
- **from_config(config):** Creates a layer instance from its configuration.

## Usage

The `ConfoundLayer` is primarily used in models where it is crucial to adjust for confounding variables. By orthogonalizing the primary data input with respect to these confounds, the layer ensures that the subsequent layers in the model process data that is free from the influence of these external factors. This layer is particularly useful in complex models dealing with real-world data where confounding is a common issue.

~~~

# Example usage in a Keras model
model = tf.keras.models.Sequential([
    # ... (previous layers) ...
    ConfoundLayer(tot_num=1000, epsilon=1e-4, momentum=0.95, diag_offset=1e-6),
    # ... (next layers) ...
])

~~~

This layer is adaptable and can be easily integrated into various neural network architectures, particularly where data pre-processing and confound adjustment are necessary.