Siamese CIFAR-10 Tutorial
=========================

This tutorial explains every step of :mod:`deep_lvpm.tutorial.tutorial_siamese_tf`, which trains a
Siamese StructuralModel on CIFAR-10.  The code below is copied verbatim from the script so you can
follow along without editing anything.  A **GPU is required**: the batch size is 2048 and the encoder
trains for 500 epochs, which is impractical on CPU-only setups.  Apple Silicon (TensorFlow Metal) or
CUDA-enabled NVIDIA GPUs both work.
This tutorial shows how DLVPM can be used to construct a meaninful represenation of a single data type. 
More information on this can be found in the publication detailing this method.

Prerequisites
-------------

Install :mod:`deep_lvpm` as described on the :doc:`installation` page, ensure the TensorFlow backend
is available, and set ``KERAS_BACKEND=tensorflow`` before launching the tutorial.

0. Imports and runtime configuration
------------------------------------

The script pins the TensorFlow backend, enables deterministic ops, and prepares the GPU runtime before
any data processing happens.  All necessary libraries are imported here.

.. code-block:: python

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

   mixed_precision.set_global_policy("float32")

   for device in tf.config.list_physical_devices("GPU"):
       try:
           tf.config.experimental.set_memory_growth(device, True)
       except Exception:
           pass

   tf.config.run_functions_eagerly(False)


1. Load and preprocess CIFAR-10
-------------------------------

CIFAR-10 images are loaded, scaled to ``[0, 1]``, and labels are stored both as class IDs and
one-hot encodings for later evaluation.  Seeds are fixed to keep splits and augmentations repeatable.

.. code-block:: python

   NUM_CLASSES = 10
   INPUT_SHAPE = (32, 32, 3)

   (x_train, y_train_cat), (x_test, y_test_cat) = keras.datasets.cifar10.load_data()
   y_train_cat = y_train_cat.squeeze()
   y_test_cat = y_test_cat.squeeze()

   x_train = x_train.astype("float32") / 255.0
   x_test = x_test.astype("float32") / 255.0

   y_train = keras.utils.to_categorical(y_train_cat, NUM_CLASSES)
   y_test = keras.utils.to_categorical(y_test_cat, NUM_CLASSES)

   SEED = 1337
   random.seed(SEED)
   np.random.seed(SEED)
   keras.utils.set_random_seed(SEED)

   VAL_FRACTION = 0.1
   num_train = x_train.shape[0]
   indices = np.arange(num_train)
   rng = np.random.default_rng(SEED)
   rng.shuffle(indices)
   cutoff = int(num_train * (1 - VAL_FRACTION))
   x_tr, x_val = x_train[indices[:cutoff]], x_train[indices[cutoff:]]

2. Create siamese augmentations and datasets
-------------------------------------------

A ``Sequential`` augmentation model builds two independent views per image (random crops, resizing,
flips, occasional grayscale).  ``make_siamese_views_dataset`` wraps NumPy arrays into ``tf.data``
pipelines that emit ``([view_one, view_two],)`` batches for training and validation.

.. code-block:: python

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
       ds = tf.data.Dataset.from_tensor_slices(x)
       if shuffle:
           ds = ds.shuffle(len(x), seed=SEED, reshuffle_each_iteration=True)
       ds = ds.batch(int(batch_size), drop_remainder=training)

       def map_batch(batch):
           view_one = augment(batch, training=training)
           view_two = augment(batch, training=training)
           return ([view_one, view_two],)

       return ds.map(map_batch, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

   train_ds = make_siamese_views_dataset(
       x_tr, batch_size=BATCH_SIZE, shuffle=True, training=True
   )
   val_ds = make_siamese_views_dataset(
       x_val, batch_size=BATCH_SIZE, shuffle=False, training=True
   )

3. Define the shared encoder and StructuralModel
-----------------------------------------------

The Siamese branches share a single convolutional encoder (three Conv-BN/Dense blocks) that expands
to a 4096-dimensional latent space.  The same model instance is supplied twice in ``model_list`` so
weights stay tied.  A two-node adjacency matrix links the branches, and ``orthogonalization="zca"``
stabilises the high-dimensional projection head.

.. code-block:: python

   WEIGHT_DECAY = 0.0
   NDIMS = 4096
   CIFAR_image_model = keras.Sequential(
       [
           keras.Input(shape=INPUT_SHAPE),
           layers.Conv2D(64, 3, padding="same", activation="relu",
                         kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY)),
           layers.MaxPooling2D(2),
           layers.Conv2D(128, 3, padding="same", activation="relu",
                         kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY)),
           layers.MaxPooling2D(2),
           layers.Conv2D(256, 3, padding="same", activation="relu",
                         kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY)),
           layers.GlobalAveragePooling2D(),
           layers.Dense(512),
           layers.BatchNormalization(),
           layers.Dense(NDIMS),
           layers.BatchNormalization(),
           layers.ReLU(),
           layers.Dense(NDIMS),
           layers.BatchNormalization(),
       ],
       name="cifar_image_model",
   )

   model_list = [CIFAR_image_model, CIFAR_image_model]
   adjacency = tf.constant([[0, 1], [1, 0]], dtype="float32")
   regularizers = [None, None]

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

   optimizers = [
       keras.optimizers.Adam(learning_rate=1e-4),
       keras.optimizers.Adam(learning_rate=1e-4),
   ]
   dlvpm_model.compile(optimizers)

4. Train the Siamese StructuralModel
-----------------------------------

Training uses the standard ``fit`` call with the Siamese datasets.  Each epoch reports the extended
metrics introduced for StructuralModel—``total_loss``, ``cross_metric``, ``mse_loss``, and
``redundancy``—so you can monitor both cross-view alignment and within-view diversity.

.. code-block:: python

   EPOCHS = 500
   dlvpm_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=True)

5. Remove projection layers before evaluation
---------------------------------------------

Self-supervised Siamese methods typically discard the projection head before downstream evaluation
(as done in Barlow Twins and VICReg).  The helper below rebuilds a copy of the encoder without its final
``n`` layers, exposing the “trunk” features for linear probing.

.. code-block:: python

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
   image_model = remove_last_layers(dlvpm_model.model_list[0], n=7)

6. Export embeddings and run a linear probe
------------------------------------------

The truncated encoder generates embeddings for the train/test splits.  A scikit-learn pipeline applies
``StandardScaler`` and ``LinearSVC`` to measure how linearly separable the features are.  Accuracy,
full classification report, and confusion matrix act as the final evaluation.

.. code-block:: python

   train_dlvs = image_model.predict(x_train, batch_size=32, verbose=1)
   test_dlvs = image_model.predict(x_test, batch_size=32, verbose=1)

   print(f"Train DLVs shape: {train_dlvs.shape}")
   print(f"Test  DLVs shape: {test_dlvs.shape}")

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

Running the tutorial
--------------------

Launch the script directly:

.. code-block:: bash

   export KERAS_BACKEND=tensorflow
   python -m deep_lvpm.tutorial.tutorial_siamese_tf

Expect ~5 seconds per training step on Apple Silicon, with higher throughput on modern CUDA GPUs.
The downstream linear SVM typically achieves >0.6 accuracy on CIFAR-10, far higher than chance, 
and comparable with similar methods such as BarlowTwins (ref), trained for the same amount of time 
on the same dataset.

If this deep dive was useful, please `star <https://github.com/alexjamesing/Deep_LVPM>`_ the repository—community
support signals that DLVPM matters and helps us justify the time invested in future improvements.
