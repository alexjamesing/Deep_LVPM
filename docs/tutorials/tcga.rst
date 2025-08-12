TCGA Lung Cancer Tutorial
=========================

This tutorial demonstrates how to use DLVPM to integrate multiple data types in a lung cancer dataset derived from The Cancer Genome Atlas (TCGA).  In this example we link five different modalities: histological image features, RNA‑seq, methylation, miRNA and somatic mutation data.  Each view is encoded by a small residual fully connected network and connected via a five‑factor structural path model.

Prerequisites
-------------

Ensure that :mod:`deep_lvpm` is installed as described on the :doc:`installation` page.  The tutorial uses the small sample datasets bundled with the package under ``deep_lvpm.data`` and can be run on CPU.

1. Load the multi‑omics dataset
-------------------------------

We start by importing the necessary dependencies and loading the training data.  The data files are packaged as NumPy archives inside the ``deep_lvpm.data`` module.

.. code-block:: python

   import numpy as np
   import tensorflow as tf
   from tensorflow import keras
   from tensorflow.keras import layers, regularizers, optimizers
   from importlib import resources

   import deep_lvpm as DLVPM
   from deep_lvpm.models.StructuralModel import StructuralModel

   # Disable eager mode for improved performance
   tf.config.run_functions_eagerly(False)

   # Load the training arrays (preserve order!)
   with resources.as_file(resources.files("deep_lvpm.data") / "Lung_multiomics_sample_train.npz") as f:
       arrays = np.load(f)
       rnaseq      = arrays["rnaseq"]
       snv         = arrays["snv"]
       methylation = arrays["methylation"]
       mirna       = arrays["mirna"]
       histo20     = arrays["histo20"]

   # Assemble the list of data arrays in the same order
   X_arr = [histo20, rnaseq, methylation, mirna, snv]

2. Define measurement models
----------------------------

Each data type is processed by a simple fully connected residual block.  You can experiment with different architectures or hyperparameters.

.. code-block:: python

   def residual_block(
       input_dim: int,
       kernel_reg_l1: float = 0.01,
       kernel_reg_l2: float = 0.01,
       dropout_rate: float = 0.5,
       name: str = "residual_block",
   ) -> tf.keras.Model:
       """
       Builds a simple fully connected residual block.

       Parameters
       ----------
       input_dim : int
           Number of features in the (flat) input vector.
       kernel_reg_l1 : float
           L1 regularisation factor for dense layers.
       kernel_reg_l2 : float
           L2 regularisation factor for dense layers.
       dropout_rate : float
           Dropout rate applied after the residual connection.
       name : str
           Name for the returned Keras model.
       """
       inputs = keras.Input(shape=(input_dim,), name=f"{name}_in")

       x = layers.Dense(
           input_dim,
           activation="linear",
           kernel_initializer=keras.initializers.Identity(),
           kernel_regularizer=regularizers.l1_l2(l1=kernel_reg_l1, l2=kernel_reg_l2),
           name=f"{name}_dense1",
       )(inputs)
       x = layers.BatchNormalization(name=f"{name}_bn")(x)
       x = layers.ReLU(name=f"{name}_relu")(x)
       x = layers.Dense(
           input_dim,
           activation="linear",
           kernel_initializer=keras.initializers.Identity(),
           kernel_regularizer=regularizers.l1_l2(l1=kernel_reg_l1, l2=kernel_reg_l2),
           name=f"{name}_dense2",
       )(x)
       x = layers.Add(name=f"{name}_add")([inputs, x])
       x = layers.Dropout(dropout_rate, name=f"{name}_drop")(x)

       return keras.Model(inputs=inputs, outputs=x, name=name)

   # Create an encoder for each modality
   model_list = [
       residual_block(histo20.shape[1], name="histo20_enc"),
       residual_block(rnaseq.shape[1],  name="rnaseq_enc"),
       residual_block(methylation.shape[1], name="meth_enc"),
       residual_block(mirna.shape[1],   name="mirna_enc"),
       residual_block(snv.shape[1],     name="snv_enc"),
   ]

3. Specify the structural path matrix
-------------------------------------

For this example we use a five‑factor model with asymmetric paths.  The matrix below defines which latent factors influence each other.

.. code-block:: python

   import numpy as np

   ndims = 5  # number of latent factors

   Path = np.array([
       # F1 F2 F3 F4 F5
       [0, 1, 0, 0, 0],  # F1 ← F2
       [1, 0, 1, 1, 1],  # F2 ← F1,F3,F4,F5
       [0, 1, 0, 0, 0],  # F3 ← F2
       [0, 1, 0, 0, 0],  # F4 ← F2
       [0, 1, 0, 0, 0],  # F5 ← F2
   ], dtype="float32")

   batch_size  = 256
   epochs      = 300
   total_steps = int(rnaseq.shape[0] / batch_size) * epochs

   # Exponential learning rate decay
   init_lr, final_lr = 1e-4, 1e-5
   lr_schedule = optimizers.schedules.ExponentialDecay(
       initial_learning_rate=init_lr,
       decay_steps=total_steps,
       decay_rate=final_lr / init_lr,
       staircase=False,
   )

   # Total number of samples (needed by DLVPM for normalisation)
   tot_num = rnaseq.shape[0]

4. Build and compile the model
-------------------------------

We create a :class:`StructuralModel` instance and provide regularisers for the projection layers.  We then compile it with a list of optimisers, one per view.

.. code-block:: python

   from tensorflow.keras import regularizers

   # Regularisers applied to each projection layer
   regularizer_list = [
       regularizers.L1L2(l1=0.01, l2=0.01),
       regularizers.L1L2(l1=0.01, l2=0.01),
       regularizers.L1L2(l1=0.01, l2=0.01),
       regularizers.L1L2(l1=0.01, l2=0.01),
       regularizers.L1L2(l1=0.01, l2=0.01),
   ]

   # Build the structural model
   DLVPM_Structural_instance = StructuralModel(
       Path,
       model_list,
       regularizer_list,
       tot_num,
       ndims,
       momentum=0.95,
       epsilon=0.001,
       orthogonalization="Moore-Penrose",
   )

   # One optimizer per measurement model using the decaying learning rate
   opt_list = [
       optimizers.Adam(learning_rate=lr_schedule) for _ in model_list
   ]

   # Compile the model
   DLVPM_Structural_instance.compile(optimizer=opt_list)

5. Train and evaluate
---------------------

Training proceeds with the standard Keras ``fit`` interface.  The ``evaluate`` method returns both the mean squared error and the mean correlation between connected data types.

.. code-block:: python

   # Train the model on the training data
   DLVPM_Structural_instance.fit(
       X_arr,
       batch_size=batch_size,
       epochs=epochs,
       verbose=True,
   )

   # Evaluate on the training data
   mean_corr = DLVPM_Structural_instance.evaluate(X_arr)
   print(f"Mean correlation on training data: r={mean_corr[1]:.3f}")

6. Evaluate on the test set
---------------------------

We load the separate test dataset and compute the mean correlation of the learned DLVs.

.. code-block:: python

   # Load the independent test dataset
   with resources.as_file(resources.files("deep_lvpm.data") / "Lung_multiomics_sample_test.npz") as f:
       arrays = np.load(f)
       rnaseq_test      = arrays["rnaseq"]
       snv_test         = arrays["snv"]
       methylation_test = arrays["methylation"]
       mirna_test       = arrays["mirna"]
       histo20_test     = arrays["histo20"]

   X_arr_test = [histo20_test, rnaseq_test, methylation_test, mirna_test, snv_test]

   mean_corr_test = DLVPM_Structural_instance.evaluate(X_arr_test)
   print(f"Mean correlation on test data: r={mean_corr_test[1]:.3f}")

7. Inspect the learned latent variables
--------------------------------------

To extract the latent factors for each view, call ``predict``.  This returns a tensor with shape ``(n_samples, ndims, n_views)``.

.. code-block:: python

   test_DLVs = DLVPM_Structural_instance.predict(X_arr_test)

   # Correlation matrix of the first latent factor across views
   corr_first = np.corrcoef(test_DLVs[:, 0, :].T)
   print("Correlation matrix for latent factor 1:", corr_first)

   # Correlation matrix of the second latent factor
   corr_second = np.corrcoef(test_DLVs[:, 1, :].T)
   print("Correlation matrix for latent factor 2:", corr_second)

8. Save the model
-----------------

Finally, save your trained model to disk in the ``.keras`` format:

.. code-block:: python

   DLVPM_Structural_instance.save("/path/to/output_folder/DLVPM_Model.keras")

This tutorial illustrates how DLVPM can be applied to real multi‑omics data.  You can extend this example by changing the measurement models, experimenting with different regularisation schemes, or altering the structural path matrix to test different hypotheses about cross‑modal relationships.