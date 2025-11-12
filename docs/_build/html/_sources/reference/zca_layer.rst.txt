ZCALayer
========

The :class:`deep_lvpm.layers.ZCALayer` provides an alternative orthogonalisation mechanism for DLVPM.  It is selected automatically when you instantiate a :class:`StructuralModel` with ``orthogonalization="zca"``.  Unlike :class:`FactorLayer`, which performs orthogonalisation within the layer, the ZCA whitening transform is applied inside the structural model itself, and the :class:`ZCALayer` only performs batch normalisation and linear projection.

Overview
--------

After normalising inputs, the :class:`ZCALayer` projects them into a latent space.  The structural model then applies a ZCA (zero‑phase component analysis) whitening to the latent outputs across views.  This can improve stability and interpretability in some applications.

Constructor arguments
---------------------

.. code-block:: python

   ZCALayer(
       kernel_regularizer=None,
       epsilon=1e-6,
       momentum=0.95,
       diag_offset=1e-6,
       tot_num=None,
       ndims=1,
       run=0,
       **kwargs,
   )

Important arguments include:

* **kernel_regularizer** – Regulariser applied to the projection layer's weights.
* **epsilon** – Small constant for batch normalisation (default: 1e-6).
* **momentum** – Momentum for moving statistics (default: 0.95).
* **diag_offset** – Small value added to the diagonal of covariance matrices to ensure invertibility (default: 1e-6).
* **tot_num** – Total number of samples seen during training.
* **ndims** – Number of latent factors to extract.

Methods
-------

* **build(input_shape)** – Initialises weights and variables.
* **call(inputs, training=False)** – Performs batch normalisation and linear projection.
* **get_config()** / **from_config()** – Standard Keras serialisation methods.

Usage
-----

You do not normally instantiate :class:`ZCALayer` directly.  It is added internally by :class:`StructuralModel` when ``orthogonalization="zca"``.  See the :doc:`structural_model` page for details on switching between orthogonalisation strategies.