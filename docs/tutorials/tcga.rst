TCGA Lung Cancer Tutorial (TensorFlow)
======================================

This tutorial reproduces ``deep_lvpm/tutorial/tutorial_tcga_tf.py`` in a prose format.  We integrate five TCGA modalities—histology image features, RNA‑seq, DNA methylation, miRNA, and somatic mutations—using shallow residual encoders and a five-factor StructuralModel.  Along the way we highlight why each step matters.

Prerequisites
-------------

* Install :mod:`deep_lvpm` with TensorFlow extras.
* Set ``KERAS_BACKEND=tensorflow`` before importing Keras so the backend is locked in.
* The tutorial uses the sample NPZ files shipped in :mod:`deep_lvpm.data`, so no extra downloads are required.

Step 1 – Load the packaged multi-omics views
--------------------------------------------

The helper below mirrors ``_load_tcga_sample`` from the script.  Keeping the order of the views consistent is crucial because the encoders, optimisers, and adjacency entries rely on it.

.. code-block:: python

   import os
   from importlib import resources

   import numpy as np
   import tensorflow as tf
   import keras
   from keras import layers, regularizers
   from keras.optimizers import Adam, schedules

   from deep_lvpm.models.StructuralModel import StructuralModel

   os.environ.setdefault("KERAS_BACKEND", "tensorflow")

   def load_tcga_sample() -> list[np.ndarray]:
       """Load the five training modalities bundled with deep_lvpm."""

       with resources.as_file(resources.files("deep_lvpm.data") / "Lung_multiomics_sample_train.npz") as handle:
           arrays = np.load(handle)
           rnaseq = arrays["rnaseq"]
           snv = arrays["snv"]
           methylation = arrays["methylation"]
           mirna = arrays["mirna"]
           histo20 = arrays["histo20"]
       return [histo20, rnaseq, methylation, mirna, snv]

   views = load_tcga_sample()
   view_names = ["histo20", "rnaseq", "methylation", "mirna", "snv"]

Step 2 – Build residual measurement encoders
--------------------------------------------

Each modality feeds into a shallow residual MLP.  The residual connection preserves the original features while allowing the encoder to learn corrections, which stabilises the downstream FactorLayer.

.. code-block:: python

   def residual_block(input_dim: int, name: str) -> keras.Model:
       """Return a one-hidden-layer residual encoder used for tabular views."""

       inputs = keras.Input(shape=(input_dim,), name=f"{name}_in")
       x = layers.Dense(
           input_dim,
           activation="linear",
           kernel_initializer=keras.initializers.Identity(),
           kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
           name=f"{name}_dense1",
       )(inputs)
       x = layers.BatchNormalization(momentum=0.9, name=f"{name}_bn1")(x)
       x = layers.ReLU(name=f"{name}_relu")(x)
       x = layers.Dense(
           input_dim,
           activation="linear",
           kernel_initializer=keras.initializers.Identity(),
           kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
           name=f"{name}_dense2",
       )(x)
       x = layers.Add(name=f"{name}_add")([inputs, x])
       x = layers.Dropout(0.5, name=f"{name}_drop")(x)
       return keras.Model(inputs=inputs, outputs=x, name=f"{name}_encoder")

   encoders = [residual_block(view.shape[1], name) for view, name in zip(views, view_names)]

Step 3 – Configure the structural graph and learning schedule
-------------------------------------------------------------

Factor 2 acts as the central hub in this example.  We also create an exponential decay schedule for the Adam optimisers so the learning rate tapers over 300 epochs.

.. code-block:: python

   adjacency = np.array(
       [
           [0, 1, 0, 0, 0],
           [1, 0, 1, 1, 1],
           [0, 1, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 1, 0, 0, 0],
       ],
       dtype="float32",
   )

   batch_size = 256
   epochs = 300
   total_steps = max(1, (views[0].shape[0] // batch_size) * epochs)

   lr_schedule = schedules.ExponentialDecay(
       initial_learning_rate=1e-4,
       decay_steps=total_steps,
       decay_rate=1e-5 / 1e-4,
       staircase=False,
   )

   regulariser_list = [regularizers.L1L2(l1=1e-2, l2=1e-2) for _ in encoders]

Step 4 – Instantiate and fit the StructuralModel
------------------------------------------------

``tot_num`` should match the number of samples in each view (they are aligned).  Giving each measurement model its own optimiser keeps the optimisation statistics independent across modalities.

.. code-block:: python

   structural_model = StructuralModel(
       Path=adjacency,
       model_list=encoders,
       regularizer_list=regulariser_list,
       tot_num=views[0].shape[0],
       ndims=5,
       orthogonalization="Moore-Penrose",
       momentum=0.95,
       epsilon=1e-3,
       train_DLV=True,
   )

   optimiser_list = [Adam(learning_rate=lr_schedule) for _ in encoders]
   structural_model.compile(optimizer=optimiser_list)

   history = structural_model.fit(
       views,
       batch_size=batch_size,
       epochs=epochs,
       verbose=True,
   )

Step 5 – Inspect metrics and latent variables
---------------------------------------------

The TensorFlow backend returns a metrics dictionary.  You can also call ``predict`` to obtain the raw DLV tensor for further analysis or visualisation.

.. code-block:: python

   metrics = structural_model.evaluate(views, verbose=False)
   metrics = {name: float(value) for name, value in metrics.items()}
   print("Training metrics:", metrics)
   print("Training history keys:", list(history.history))

   latent = structural_model.predict(views, verbose=False)
   print("Latent tensor shape:", latent.shape)

Optional – Evaluate on the bundled test set
-------------------------------------------

The repository also ships a held-out sample.  Load it with the same helper, evaluate the metrics, and compare them with the training run.

.. code-block:: python

   def load_tcga_sample_test() -> list[np.ndarray]:
       with resources.as_file(resources.files("deep_lvpm.data") / "Lung_multiomics_sample_test.npz") as handle:
           arrays = np.load(handle)
           rnaseq = arrays["rnaseq"]
           snv = arrays["snv"]
           methylation = arrays["methylation"]
           mirna = arrays["mirna"]
           histo20 = arrays["histo20"]
       return [histo20, rnaseq, methylation, mirna, snv]

   test_views = load_tcga_sample_test()
   test_metrics = structural_model.evaluate(test_views, verbose=False)
   print("Test metrics:", {name: float(value) for name, value in test_metrics.items()})

Next steps
----------

* Adjust the adjacency matrix to express different hypotheses about factor connectivity.
* Swap in deeper encoders for modalities that benefit from additional capacity.
* Enable GPU execution for faster experimentation; the sample dataset is small enough for CPU but larger cohorts will benefit from acceleration.
