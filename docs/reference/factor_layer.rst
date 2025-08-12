FactorLayer
===========

The :class:`deep_lvpm.layers.FactorLayer` is a custom Keras layer used by DLVPM to generate orthogonal latent factors that are highly correlated between different data views.  When you instantiate a :class:`StructuralModel` with ``orthogonalization="Moore-Penrose"`` (the default), a :class:`FactorLayer` is appended automatically to each measurement model.

Overview
--------

The layer performs three operations:

1. **Batch Normalisation** – each batch of inputs is normalised to have zero mean and unit variance.
2. **Orthogonalisation** – outputs are orthogonalised to produce uncorrelated DLVs.
3. **Linear projection** – a final dense layer projects the network output into the latent space used for correlation with other views.

These operations are adaptive: the behaviour during training and inference differs similarly to standard normalisation layers.

Constructor arguments
---------------------

.. code-block:: python

   FactorLayer(
       kernel_regularizer=None,
       epsilon=1e-6,
       momentum=0.95,
       tot_num=None,
       ndims=1,
       run=0,
       **kwargs,
   )

The key arguments are:

* **kernel_regularizer** – Regulariser applied to the projection layer's weights (e.g., ``tf.keras.regularizers.L1L2``).  If ``None``, no regularisation is used.
* **epsilon** – Small constant added to the variance during normalisation for numerical stability (default: 1e-6).
* **momentum** – Momentum for the moving mean and variance used in batch normalisation (default: 0.95).
* **tot_num** – Total number of samples seen during training.  Used internally for scaling covariance estimates.
* **ndims** – Number of latent factors to extract.
* **run** – Internal counter used to initialise moving statistics.

Methods
-------

The main methods are:

* **build(input_shape)** – Allocates layer weights and internal variables.
* **call(inputs, training=False)** – Performs the forward pass.  Returns a tensor of shape ``(batch_size, ndims)`` representing the latent projections.
* **get_config()** / **from_config()** – Support serialisation/deserialisation with Keras.

Usage
-----

Users typically do not instantiate :class:`FactorLayer` directly; it is added to each measurement model when you create a :class:`StructuralModel` with the default orthogonalisation method.  However, it can also be used manually to orthogonalise arbitrary tensors.  For example:

.. code-block:: python

   import tensorflow as tf
   from deep_lvpm.layers import FactorLayer

   x = tf.keras.Input(shape=(100,))
   f = FactorLayer(ndims=5)(x)
   model = tf.keras.Model(inputs=x, outputs=f)