MNIST Tutorial
==============

This tutorial demonstrates how to build and train a simple DLVPM model that links greyscale images to dummy‑coded labels in the MNIST dataset.  The goal is to learn deep latent variables that are maximally correlated between the image and label views.

Prerequisites
-------------

Make sure you have installed :mod:`deep_lvpm` and its dependencies as described in the :doc:`installation` page.  This example uses only CPU and should run in a few minutes on a modern laptop.

1. Prepare the data
-------------------

We begin by loading the MNIST dataset using Keras.  Images are scaled to the range [0, 1], expanded to include a channel dimension, and the labels are converted to one‑hot encoded matrices.

.. code-block:: python

   import tensorflow as tf
   import numpy as np
   from tensorflow import keras
   from keras import layers

   num_classes = 10
   input_shape = (28, 28, 1)

   # Load the data and split it between train and test sets
   (x_train, y_train_cat), (x_test, y_test_cat) = keras.datasets.mnist.load_data()

   # Scale images to [0, 1] and add channel dimension
   x_train = x_train.astype("float32") / 255
   x_test  = x_test.astype("float32") / 255
   x_train = np.expand_dims(x_train, -1)
   x_test  = np.expand_dims(x_test,  -1)

   # Convert labels to one‑hot encoding
   y_train = keras.utils.to_categorical(y_train_cat, num_classes)
   y_test  = keras.utils.to_categorical(y_test_cat,  num_classes)

   # Assemble lists of data for DLVPM
   data_train_list = [x_train, y_train]
   data_test_list  = [x_test,  y_test]

2. Define measurement models
----------------------------

A DLVPM model requires one **measurement model** per data view.  For the MNIST images we use a small convolutional neural network.  For the dummy‑coded labels we simply define an input layer, because no further processing is required.

.. code-block:: python

   from deep_lvpm.models.StructuralModel import StructuralModel

   # Convolutional encoder for images
   MNIST_image_model = keras.Sequential(
       [
           keras.Input(shape=input_shape),
           layers.Conv2D(32, kernel_size=(3, 3), activation="relu",
                         kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-5)),
           layers.MaxPooling2D(pool_size=(2, 2)),
           layers.Conv2D(64, kernel_size=(3, 3), activation="relu",
                         kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-5)),
           layers.MaxPooling2D(pool_size=(2, 2)),
           layers.Flatten(),
           layers.Dense(100, activation="relu"),
           layers.Dropout(0.5),
       ],
       name="mnist_image_model",
   )

   # Identity mapping for labels
   data_input = keras.Input(shape = (10,))
   data_output = keras.layers.Activation('linear', name='identity')(data_input)
   MNIST_label_model=keras.Model(inputs=data_input,outputs=data_output)
  
   model_list = [MNIST_image_model, MNIST_label_model]

3. Define the structural path matrix
------------------------------------

DLVPM models use a binary **path matrix** to specify which latent factors are connected across data views.  For two views the matrix is trivial (each view connects to the other):

.. code-block:: python

   import tensorflow as tf

   # Adjacency matrix connecting the two data views
   Path = tf.constant([[0, 1],
                       [1, 0]], dtype="float32")

4. Build and compile the StructuralModel
---------------------------------------

We instantiate :class:`deep_lvpm.models.StructuralModel.StructuralModel` with the path matrix, the measurement models, and optional regularizers.  We must also specify the total number of samples and the dimensionality of the latent space (``ndims``).

.. code-block:: python

   ndims       = 9               # number of deep latent variables
   tot_num     = x_train.shape[0]   # total number of samples
   regularizer_list = [None, None]

   DLVPM_Model = StructuralModel(Path, model_list, regularizer_list, tot_num, ndims)

   # Compile the model with one optimiser per view
   optimizer_list = [
       keras.optimizers.Adam(learning_rate=1e-4),
       keras.optimizers.Adam(learning_rate=1e-4),
   ]
   DLVPM_Model.compile(optimizer=optimizer_list)

5. Train and evaluate the model
-------------------------------

Training uses the standard Keras ``fit`` interface with a list of data arrays.  After training we evaluate the model on the test set.  Two metrics are returned: the mean squared error and the mean Pearson correlation between latent factors.

.. code-block:: python

   batch_size = 256
   epochs     = 10

   # Train the model
   DLVPM_Model.fit(
       data_train_list,
       batch_size=batch_size,
       epochs=epochs,
       verbose=True,
       validation_split=0.1,
   )

   # Evaluate on the test set
   metrics = DLVPM_Model.evaluate(data_test_list)
   print(f"Test MSE: {metrics[0]}, mean correlation: {metrics[1]}")

6. Inspect the latent space
---------------------------

The ``predict`` method returns a three‑dimensional tensor of shape ``(n_samples, ndims, n_views)`` containing the learned deep latent variables (DLVs).  We can compute correlations between corresponding DLVs across views to verify that they are highly correlated, and we can project the DLVs to two dimensions using t‑SNE to visualise the latent structure.

.. code-block:: python

   import numpy as np
   from sklearn.manifold import TSNE
   import matplotlib.pyplot as plt

   # Predict the latent variables for the test data
   DLVs = DLVPM_Model.predict(data_test_list)

   # Correlation matrix between the first latent variable of each view
   Cmat = np.corrcoef(DLVs[:, 0, :].T)
   print("Correlation matrix for the first DLV:", Cmat)

   # Extract DLVs from a single view (images)
   image_DLVs = DLVPM_Model.model_list[0].predict(data_test_list[0])

   # Randomly select 100 samples for visualisation
   random_indices = np.random.choice(image_DLVs.shape[0], size=100, replace=False)
   image_DLVs_plot = image_DLVs[random_indices, :]
   y_test_plot     = y_test[random_indices, :]

   # Apply t‑SNE
   tsne = TSNE(n_components=2, random_state=42)
   tsne_results = tsne.fit_transform(image_DLVs_plot)

   # Plot the 2D projection coloured by digit label
   plt.figure(figsize=(12, 8))
   for i in range(y_test_plot.shape[1]):
       points = tsne_results[y_test_plot[:, i] == 1]
       plt.scatter(points[:, 0], points[:, 1], label=f"Digit {i}")
   plt.title("t‑SNE projection of MNIST image DLVs")
   plt.legend()
   plt.show()

7. Save the trained model
-------------------------

To reuse a trained model in the future, save it to disk using the Keras ``.save`` method.  Because DLVPM uses custom layers, save in the newer ``.keras`` format rather than the legacy ``.h5`` format.

.. code-block:: python

   DLVPM_Model.save("/path/to/output_folder/DLVPM_Model.keras")

This tutorial illustrates the core steps for defining, training and analysing a DLVPM model on a simple two‑view dataset.  More complex applications can extend this pattern by providing additional measurement models and specifying richer structural path matrices.