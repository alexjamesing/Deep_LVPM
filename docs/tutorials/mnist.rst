MNIST Tutorial (TensorFlow)
===========================

The MNIST tutorial walks through the complete workflow for a two‑view Deep LVPM model: we pair a convolutional encoder for the greyscale digits with a simple identity encoder for the one‑hot labels, learn highly correlated deep latent variables (DLVs), and visualise the latent space.  The code below mirrors ``deep_lvpm/tutorial/tutorial_mnist_tf.py`` line for line, with additional commentary to unpack each step.

Prerequisites
-------------

* Install :mod:`deep_lvpm` with TensorFlow support (see :doc:`../installation`).
* Set ``KERAS_BACKEND=tensorflow`` so Keras 3 selects the correct backend.
* Optional: run on GPU for faster training, although the script finishes quickly on CPU.

Step 1 – Load and prepare MNIST
-------------------------------

Normalise the pixel values, add a channel dimension so the convolutional layers see ``(28, 28, 1)`` inputs, and convert the labels to one‑hot encodings for the identity view.

.. code-block:: python

   import os
   import numpy as np
   import keras
   from keras import layers, regularizers

   from deep_lvpm.models.StructuralModel import StructuralModel

   os.environ.setdefault("KERAS_BACKEND", "tensorflow")

   num_classes = 10
   input_shape = (28, 28, 1)

   print("Loading MNIST data...")
   (x_train, y_train_cat), (x_test, y_test_cat) = keras.datasets.mnist.load_data()

   print("Preprocessing images and labels...")
   x_train = x_train.astype("float32") / 255.0
   x_test = x_test.astype("float32") / 255.0
   x_train = np.expand_dims(x_train, axis=-1)  # add channel dimension
   x_test = np.expand_dims(x_test, axis=-1)

   y_train = keras.utils.to_categorical(y_train_cat, num_classes)
   y_test = keras.utils.to_categorical(y_test_cat, num_classes)

   data_train = [x_train, y_train]
   data_test = [x_test, y_test]

Step 2 – Build the measurement models
-------------------------------------

Each view needs a measurement model.  The image encoder is a compact CNN; the label encoder is an identity mapping because the labels are already dummy‑coded.  The ``Sequential.add`` pattern in the script keeps the tutorial procedural and easy to follow.

.. code-block:: python

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

Step 3 – Configure the StructuralModel
--------------------------------------

The path matrix is a 2×2 symmetric grid so each view learns to correlate with the other.  ``tot_num`` should be the size of the full training set; FactorLayer uses it to scale running covariance estimates.

.. code-block:: python

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

   print("Compiling StructuralModel...")
   image_optimizer = keras.optimizers.Adam(learning_rate=1e-4)
   label_optimizer = keras.optimizers.Adam(learning_rate=1e-4)
   structural_model.compile(optimizer=[image_optimizer, label_optimizer])

Step 4 – Train and evaluate
---------------------------

Keras 3 still accepts a list of NumPy arrays for multi-view training.  After the ``fit`` call completes we convert the returned metrics dictionary to regular floats for printing.

.. code-block:: python

   print("Training...")
   history = structural_model.fit(
       data_train,
       batch_size=256,
       epochs=20,
       verbose=True,
       validation_split=0.1,
   )

   print("Evaluating on the test split...")
   metrics = structural_model.evaluate(data_test, verbose=False)
   metrics = {name: float(value) for name, value in metrics.items()}
   for metric_name, metric_value in metrics.items():
       print(f"{metric_name}: {metric_value:.6f}")

Step 5 – Inspect the latent space
---------------------------------

``predict`` returns the learned DLV tensor with shape ``(n_samples, ndims, n_views)``.  To visualise the latent structure we extract the image view, sample a subset, and run t‑SNE.

.. code-block:: python

   from sklearn.manifold import TSNE

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

Next steps
----------

* Save the trained model for reuse: ``structural_model.save("mnist_structural_model.keras")``.
* Swap in alternative encoders (e.g., deeper CNNs) to see how the latent correlations change.
* Run the PyTorch backend version (see :doc:`mnist_torch`) to compare outputs across backends.
