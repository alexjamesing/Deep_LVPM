ConfoundLayer
=============

The :class:`deep_lvpm.layers.ConfoundLayer` removes the influence of confounding variables from a primary data input by orthogonalising one tensor with respect to another.  This layer can be used within custom models where you need to adjust for known confounders before computing latent representations.

Overview
--------

Given two inputs—``input[0]`` (the primary features) and ``input[1]`` (the confounds)—the :class:`ConfoundLayer` performs:

1. Batch normalisation of both inputs.
2. Estimation of the covariance between the primary features and the confounds.
3. Projection of the primary features into the subspace orthogonal to the confounds.
4. Optional dropout.

The result is a cleaned feature representation with reduced dependence on the confounding variables.

Constructor arguments
---------------------

.. code-block:: python

   ConfoundLayer(
       tot_num,
       epsilon=1e-4,
       momentum=0.95,
       diag_offset=1e-6,
       run=0,
       **kwargs,
   )

where:

* **tot_num** (*int*) – Total number of samples used during training.  Necessary for scaling covariance estimates.
* **epsilon** (*float*) – Small constant for batch normalisation (default: 1e-4).
* **momentum** (*float*) – Momentum for updating covariance matrices (default: 0.95).
* **diag_offset** (*float*) – Constant added to the diagonal of covariance matrices to ensure invertibility (default: 1e-6).
* **run** (*int*) – Internal run counter.

Methods
-------

* **build(input_shape)** – Creates the variables used for normalisation and projection.
* **call(inputs, training=False)** – Takes a list ``[primary, confounds]`` and returns the orthogonalised primary features.
* **get_config()** / **from_config()** – Keras serialisation functions.

Usage
-----

Use :class:`ConfoundLayer` within your own Keras models to regress out known confounders.  For example, if you want to remove batch effects from gene expression data:

.. code-block:: python

   import tensorflow as tf
   from deep_lvpm.layers import ConfoundLayer

   primary_input = tf.keras.Input(shape=(n_features,))
   confounds = tf.keras.Input(shape=(n_confounds,))

   cleaned = ConfoundLayer(tot_num=n_samples)([primary_input, confounds])
   # Pass cleaned features through subsequent layers
   ...