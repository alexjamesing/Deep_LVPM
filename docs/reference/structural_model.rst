StructuralModel
===============

The :class:`deep_lvpm.models.StructuralModel.StructuralModel` class is the core of the DLVPM toolbox.  It inherits from ``keras.Model`` (Keras 3, multi-backend) and coordinates multiple sub-models, termed **measurement models** (one per data view), together with a binary **path model**, which specifies how latent factors are connected across views.  During training, the model learns sets of orthogonal deep latent variables (DLVs) that maximise correlation between the outputs of the measurement models while respecting the path structure.  The implementation runs unchanged on either the TensorFlow or PyTorch backends as long as the appropriate backend is selected when importing Keras.

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
       epsilon=1e-4,
       train_DLV=True,
       is_siamese=False,
       diag_offset=1e-3,
       **kwargs
   )

where:

* **Path** (*array-like*): Binary adjacency matrix defining which latent factors are connected between data views.  The shape of ``Path`` determines the number of latent factors.
* **model_list** (*list of keras.Model*): One measurement model per data view.  Each model should accept a single input tensor and output a latent representation of arbitrary dimension.
* **regularizer_list** (*list*): List of regulariser objects applied to the final projection layer of each measurement model.  May be ``None`` for no regularisation.
* **tot_num** (*int*): Total number of samples used in training.  This is used internally for scaling covariance matrices.
* **ndims** (*int*): Number of orthogonal latent variables (DLVs) to extract.
* **orthogonalization** (*str, optional*): Method for orthogonalising latent factors.  Either ``"Moore-Penrose"`` (default) or ``"zca"``.  The StructuralModel automatically appends the appropriate projection head for the requested method; no user action is required besides choosing the mode.
* **momentum** (*float, optional*): Momentum parameter for updating global statistics (default: 0.95).
* **epsilon** (*float, optional*): Small constant added for numerical stability (default: 1e-4).
* **train_DLV** (*bool, optional*): If ``True`` (default) the shared latent variables are constructed in training mode using batch parameters in the projection layer (rather than running parameters)
* **is_siamese** (*bool, optional*): Indicates whether the model is being used in a Siamese configuration where all views share weights (default: ``False``).
* **diag_offset** (*float, optional*): Additional diagonal jitter applied when using ZCA orthogonalisation to keep covariance matrices well-conditioned.
* **kwargs**: Forwarded to ``keras.Model`` (e.g., ``name`` or ``dtype``).

Attributes
----------

A :class:`StructuralModel` instance exposes several public attributes:

* **Path** – the binary adjacency matrix.
* **model_list** – list of measurement models.
* **regularizer_list** – list of regularisers for projection layers.
* **tot_num** – total number of samples.
* **ndims** – number of latent variables.
* **loss_tracker_total** – Keras metric tracking total loss during training.
* **corr_tracker** – Keras metric tracking the average correlation between connected views.
* **loss_tracker_mse** – Keras metric tracking mean squared error.
* **loss_tracker_redundancy** – Keras metric tracking the redundancy penalty (average within-view correlation of latent factors).

Common methods
--------------

Because :class:`StructuralModel` subclasses ``keras.Model`` you can use the standard ``compile``/``fit``/``evaluate``/``predict`` APIs regardless of backend:

``compile(optimizer_list)``
    Configures the model for training.  Pass either a **list of optimisers** (one per measurement model) or a single optimiser object (useful for Siamese setups where the measurement models share weights).  Example:

    .. code-block:: python

       optimizer_list = [
           tf.keras.optimizers.Adam(learning_rate=1e-4),
           tf.keras.optimizers.Adam(learning_rate=1e-3),
           tf.keras.optimizers.Adam(learning_rate=1e-4),
       ]
       struct_model.compile(optimizer_list)

``fit(data, batch_size=None, epochs=1, ...)``
    Trains the model on a list or generator of data arrays.  The input ``data`` should be a list of arrays, one per view (or a ``tf.data``/``torch.utils.data`` dataset yielding such lists).  Additional arguments (``batch_size``, ``epochs``, callbacks, etc.) behave as in Keras.

``evaluate(data)``
    Evaluates the model on input data and returns a dictionary with the tracked metrics: ``{"total_loss", "cross_metric", "mse_loss", "redundancy"}``.  Briefly:

    * ``total_loss`` – the overall objective combining masked reconstruction error, regularisation terms, and redundancy penalties for every view.
    * ``cross_metric`` – the average Pearson correlation between latent factors in views that are connected in ``Path`` (higher is better).
    * ``mse_loss`` – the mean squared error between each view and its connected partners, masked by the adjacency matrix.
    * ``redundancy`` – the within-view correlation penalty; low values indicate that latent factors remain orthogonal/non-redundant inside each measurement model.

``predict(data)``
    Computes the deep latent variables for each view, returning a tensor of shape ``(n_samples, ndims, n_views)``.  To extract the latent variables for an individual view use ``struct_model.model_list[i].predict(data[i])``.

``calculate_corrmat(DLVs)``
    Calculates correlation matrices for the latent variables produced by ``predict``.  Returns a list of correlation matrices with length ``ndims``.  The redundancy metric reported during training/evaluation is derived from these per-view correlations.


Minimal working example
-----------------------

The script below shows the minimum wiring required to instantiate and run :class:`StructuralModel`.
It synthesises two NumPy views, builds tiny measurement models, defines a path matrix, trains the
model for a few epochs, evaluates, and finally inspects the learned latent variables.

.. code-block:: python

   """
   Minimal StructuralModel demo with explanatory comments.

   Steps covered:
   1. Build toy numpy arrays for two data views.
   2. Define simple measurement models (dense networks).
   3. Specify the structural path matrix linking the views.
   4. Train/evaluate StructuralModel end to end and inspect the outputs.
   """

   import os

   # Force a specific backend before importing Keras. Choose "torch" to run on PyTorch.
   os.environ.setdefault("KERAS_BACKEND", "tensorflow")

   import numpy as np
   import keras
   from keras import layers
   from deep_lvpm.model import StructuralModel

   # 1. Toy data -------------------------------------------------------------
   # Generate two independent Gaussian views so the example stays self-contained.
   rng = np.random.default_rng(42)
   n_samples = 512
   view_a = rng.normal(size=(n_samples, 8)).astype("float32")  # first modality (8 features)
   view_b = rng.normal(size=(n_samples, 6)).astype("float32")  # second modality (6 features)

   # 2. Measurement models --------------------------------------------------
   # Each measurement model is just a two-layer MLP that outputs latent features.
   def make_measurement(input_dim, name):
       inputs = keras.Input(shape=(input_dim,), name=f"{name}_in")
       x = layers.Dense(16, activation="relu")(inputs)
       x = layers.Dense(16, activation="relu")(x)
       outputs = layers.Dense(8, name=f"{name}_proj")(x)  # StructuralModel adds its projection head next
       return keras.Model(inputs, outputs, name=name)

   model_a = make_measurement(view_a.shape[1], "view_a_encoder")
   model_b = make_measurement(view_b.shape[1], "view_b_encoder")

   # 3. Structural path -----------------------------------------------------
   # Two views, one latent factor connecting them (symmetric adjacency).
   Path = np.array([[0, 1],
                    [1, 0]], dtype="float32")
   ndims = 4  # number of DLVs to learn per view

   # 4. Build StructuralModel -----------------------------------------------
   regularizers = [None, None]  # no extra projection regularization in this toy example
   tot_num = n_samples          # required for internal covariance scaling

   struct_model = StructuralModel(
       Path=Path,
       model_list=[model_a, model_b],
       regularizer_list=regularizers,
       tot_num=tot_num,
       ndims=ndims,
       orthogonalization="Moore-Penrose",  # stay with the default FactorLayer
       momentum=0.95,
       epsilon=1e-4,
   )

   # Compile with one optimizer per measurement model.
   optimizers = [
       keras.optimizers.Adam(learning_rate=1e-3),
       keras.optimizers.Adam(learning_rate=1e-3),
   ]
   struct_model.compile(optimizers)

   # Training loop (short run just to demonstrate API).
   history = struct_model.fit(
       [view_a, view_b],   # inputs must be passed as a list matching model_list order
       batch_size=64,
       epochs=5,
       verbose=True,
   )

   # Evaluate returns the tracked metrics: total_loss, cross_metric, mse_loss, redundancy.
   metrics = struct_model.evaluate([view_a, view_b], verbose=False)
   print("Evaluation metrics:", metrics)

   # Predict gives the latent tensor with shape (samples, ndims, n_views).
   dlvs = struct_model.predict([view_a, view_b], verbose=False)
   print("Latent tensor shape:", dlvs.shape)
