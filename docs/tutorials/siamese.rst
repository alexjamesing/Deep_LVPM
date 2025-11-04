Siamese CIFAR-10 Tutorial (TensorFlow)
======================================

This tutorial expands ``deep_lvpm/tutorial/tutorial_siamese_tf.py`` with step-by-step commentary.  We build a Siamese Deep LVPM model that learns correlated representations from two augmented views of each CIFAR-10 image and validate the embeddings with a linear probe.

Prerequisites
-------------

* Install :mod:`deep_lvpm` with TensorFlow support and set ``KERAS_BACKEND=tensorflow``.
* A GPU is recommended—the tutorial runs for 500 epochs—but the sample code still executes on CPU with longer training times.

Step 1 – Environment configuration
----------------------------------

We lock the global precision policy to float32 and enable memory growth on detected GPUs to avoid allocation spikes.  This mirrors the initial setup in the script.

.. code-block:: python

   import os
   import numpy as np
   import tensorflow as tf
   import keras
   from keras import layers
   from keras.optimizers import Adam
   from keras.mixed_precision import set_global_policy

   from deep_lvpm.models.StructuralModel import StructuralModel

   os.environ.setdefault("KERAS_BACKEND", "tensorflow")

   set_global_policy("float32")
   tf.config.run_functions_eagerly(False)

   for device in tf.config.list_physical_devices("GPU"):
       try:
           tf.config.experimental.set_memory_growth(device, True)
       except Exception:
           pass

Step 2 – Load CIFAR-10 and prepare one-hot labels
-------------------------------------------------

The loader returns labels shaped ``(n, 1)``; squeeze them before creating one-hot matrices.

.. code-block:: python

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

Step 3 – Build the augmentation pipeline and datasets
-----------------------------------------------------

Two augmented views are required for Siamese training.  We re-use the augmentation model from the script and wrap the arrays in ``tf.data`` pipelines for efficient batching.

.. code-block:: python

   def make_augmenter() -> keras.Sequential:
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

   def make_dataset(images, batch_size, seed, augment, training):
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

   augment = make_augmenter()
   train_ds = make_dataset(x_tr, batch_size=batch_size, seed=seed, augment=augment, training=True)
   val_ds = make_dataset(x_val, batch_size=batch_size, seed=seed, augment=augment, training=False)

Step 4 – Construct the shared encoder and StructuralModel
---------------------------------------------------------

Both branches share the same CNN encoder.  Setting ``is_siamese=True`` in the StructuralModel constructor ensures weights are shared when the FactorLayer is appended.

.. code-block:: python

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

   siamese_model = StructuralModel(
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
   siamese_model.compile(optimizer=optimisers)

Step 5 – Train and track metrics
--------------------------------

The ``evaluate`` method returns the familiar metrics dictionary (``total_loss``, ``cross_metric``, ``mse_loss``, ``redundancy``).  Printing the dictionary helps monitor how redundancy drops over time.

.. code-block:: python

   history = siamese_model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=True)
   val_metrics = siamese_model.evaluate(val_ds, verbose=False)
   print("Validation metrics:", {name: float(value) for name, value in val_metrics.items()})
   print("Training history keys:", list(history.history))

Step 6 – Evaluate the representation with a linear probe
--------------------------------------------------------

Remove the final normalisation layers from the encoder to expose a compact embedding, then train a small softmax classifier on top.  This is identical to the evaluation block in the tutorial script.

.. code-block:: python

   def remove_last_layers(model: keras.Model, n: int) -> keras.Model:
       if n == 0:
           return model
       if n >= len(model.layers):
           raise ValueError(f"Cannot remove {n} layers from model with only {len(model.layers)} layers.")
       cutoff = model.layers[-(n + 1)].output
       return keras.Model(inputs=model.inputs, outputs=cutoff, name=f"{model.name}_truncated")

   truncated_encoder = remove_last_layers(siamese_model.model_list[0], n=3)
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

Next steps
----------

* Experiment with different augmentations or encoder depths to see how redundancy and probe accuracy change.
* Reduce ``epochs`` when prototyping, then scale back up for final experiments.
* Switch to the PyTorch backend tutorial (see :doc:`mnist_torch`) to explore the alternative implementation style.
