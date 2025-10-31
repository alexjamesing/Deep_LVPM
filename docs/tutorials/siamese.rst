Siamese CIFAR-10 Tutorial
=========================

This tutorial shows how to train a Siamese Deep LVPM model on the CIFAR‑10 dataset using the TensorFlow backend.  Two augmented views of each image are passed through identical convolutional encoders, and the resulting deep latent variables are evaluated with a linear probe.

Prerequisites
-------------

Install :mod:`deep_lvpm` with one of the TensorFlow extras and set ``KERAS_BACKEND=tensorflow``.  The tutorial benefits from a GPU but can run on CPU with longer training time.

1. Prepare the data and augmentation pipeline
---------------------------------------------

.. code-block:: python

   import os
   import numpy as np
   import tensorflow as tf
   import keras
   from keras import layers

   os.environ.setdefault("KERAS_BACKEND", "tensorflow")
   keras.config.run_eagerly(False)

   (x_train, y_train_cat), (x_test, y_test_cat) = keras.datasets.cifar10.load_data()
   x_train = x_train.astype("float32") / 255.0
   x_test = x_test.astype("float32") / 255.0
   y_train_cat = y_train_cat.squeeze()
   y_test_cat = y_test_cat.squeeze()

   y_train = keras.utils.to_categorical(y_train_cat, 10)
   y_test = keras.utils.to_categorical(y_test_cat, 10)

   augment = keras.Sequential(
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

   def make_dataset(images, batch_size, seed, training):
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

2. Build the shared encoder and Siamese StructuralModel
-------------------------------------------------------

.. code-block:: python

   from keras.optimizers import Adam
   from keras.mixed_precision import set_global_policy
   from deep_lvpm.models.StructuralModel import StructuralModel

   set_global_policy("float32")
   for device in tf.config.list_physical_devices("GPU"):
       try:
           tf.config.experimental.set_memory_growth(device, True)
       except Exception:
           pass

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

3. Train and monitor metrics
----------------------------

.. code-block:: python

   seed = 1337
   batch_size = 512
   epochs = 500

   rng = np.random.default_rng(seed)
   indices = rng.permutation(len(x_train))
   cutoff = int(len(x_train) * 0.9)
   x_tr, x_val = x_train[indices[:cutoff]], x_train[indices[cutoff:]]

   train_ds = make_dataset(x_tr, batch_size=batch_size, seed=seed, training=True)
   val_ds = make_dataset(x_val, batch_size=batch_size, seed=seed, training=False)

   history = siamese_model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=True)
   metrics = siamese_model.evaluate(val_ds, verbose=False)
   print({name: float(value) for name, value in metrics.items()})

4. Evaluate the learned representation with a linear probe
----------------------------------------------------------

.. code-block:: python

   def remove_last_layers(model, n):
       if n == 0:
           return model
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

With the default hyperparameters and 500 training epochs, you should expect a linear probe accuracy in the region of **60 %**. Small variations are normal depending on hardware, random seeds, and augmentation randomness.

This workflow highlights the additional ``redundancy`` metric exposed by :meth:`StructuralModel.evaluate` and demonstrates how to reuse the learned DLVPM encoder for downstream tasks.
