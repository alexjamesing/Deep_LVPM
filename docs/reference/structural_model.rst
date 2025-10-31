StructuralModel
===============

The :class:`deep_lvpm.models.StructuralModel.StructuralModel` class is the core of the DLVPM toolbox.  It inherits from :class:`keras.Model` (Keras 3) and coordinates multiple **measurement models** (one per data view) together with a binary **path model** that specifies how latent factors are connected across views.  During training, the model learns sets of orthogonal deep latent variables (DLVs) that maximise correlation between the outputs of the measurement models while respecting the path structure.  The implementation is backend-agnostic: you can train the same model on TensorFlow or PyTorch by switching ``KERAS_BACKEND``.

Parameters
----------

The constructor has the following signature:

.. code-block:: python

   StructuralModel(
       Path,
       model_list,
       regularizer_list,
       tot_num,
       ndims,
       orthogonalization="Moore-Penrose",
       momentum=0.95,
       epsilon=1e-4
   )

where:

* **Path** (*array-like*): Binary adjacency matrix defining which latent factors are connected between data views.  The shape of ``Path`` determines the number of latent factors.
* **model_list** (*list of keras.Model*): One measurement model per data view.  Each model should accept a single input tensor and output a latent representation of arbitrary dimension.
* **regularizer_list** (*list*): List of regulariser objects applied to the final projection layer of each measurement model.  May be ``None`` for no regularisation.
* **tot_num** (*int*): Total number of samples used in training.  This is used internally for scaling covariance matrices.
* **ndims** (*int*): Number of orthogonal latent variables (DLVs) to extract.
* **orthogonalization** (*str, optional*): Method for orthogonalising latent factors.  Either ``"Moore-Penrose"`` (default) or ``"zca"``.  When set to ``"zca"``, the :class:`ZCALayer` is used instead of :class:`FactorLayer` for orthogonalisation.
* **momentum** (*float, optional*): Momentum parameter for updating global statistics (default: 0.95).
* **epsilon** (*float, optional*): Small constant added for numerical stability (default: 1e-4).

Attributes
----------

A :class:`StructuralModel` instance exposes several public attributes:

* **Path** – the binary adjacency matrix.
* **model_list** – list of measurement models.
* **regularizer_list** – list of regularisers for projection layers.
* **tot_num** – total number of samples.
* **ndims** – number of latent variables.
* **loss_tracker_total** – average total loss across views.
* **corr_tracker** – mean Pearson correlation between connected views (``cross_metric``).
* **loss_tracker_mse** – reconstruction mean squared error.
* **loss_tracker_redundancy** – redundancy score that measures within-view correlation between discovered DLVs.

Common methods
--------------

Because :class:`StructuralModel` subclasses :class:`keras.Model`, it supports the standard Keras interface for compilation, training and evaluation:

``compile(optimizer_list)``
    Configures the model for training.  Unlike standard Keras models, this method requires a **list of optimisers**, one per measurement model.  Example:

    .. code-block:: python

       import keras

       optimizer_list = [
           keras.optimizers.Adam(learning_rate=1e-4),
           keras.optimizers.Adam(learning_rate=1e-3),
           keras.optimizers.Adam(learning_rate=1e-4),
       ]
       struct_model.compile(optimizer_list)

``fit(data, batch_size=None, epochs=1, ...)``
    Trains the model on a list or generator of data arrays.  The input ``data`` should be a list of arrays, one per view.  Additional arguments (``batch_size``, ``epochs``, callbacks, etc.) behave as in Keras.

``evaluate(data)``
    Evaluates the model on input data and returns a dictionary with four entries: ``total_loss``, ``cross_metric`` (mean correlation), ``mse_loss``, and ``redundancy``.  All values are per-view averages computed over the evaluation dataset.

``predict(data)``
    Computes the deep latent variables for each view, returning a tensor of shape ``(n_samples, ndims, n_views)``.  To extract the latent variables for an individual view use ``struct_model.model_list[i].predict(data[i])``.

``calculate_corrmat(DLVs)``
    Calculates correlation matrices for the latent variables produced by ``predict``.  Returns a list of correlation matrices with length ``ndims``.

Internal methods
----------------

The following methods are used internally by the implementation and are generally not called directly:

* **add_DLVPM_layer** – Adds a :class:`FactorLayer` or :class:`ZCALayer` to each measurement model, depending on the orthogonalisation method.
* **call(inputs)** – Forwards inputs through each measurement model and applies the latent projection.
* **train_step(inputs)** / **test_step(inputs)** – Custom training and testing routines.
* **mse_loss(...)** – Calculates the mean squared error loss.
* **corr_metric(...)** – Calculates the average correlation metric.
* **get_config()** and **from_config()** – Used for model serialisation.
* **get_compile_config()** and **compile_from_config()** – Used by Keras to serialise optimizer configurations.
