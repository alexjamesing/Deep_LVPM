StructuralModel
===============

The :class:`deep_lvpm.model.StructuralModel` class is the core DLVPM model.
It subclasses ``keras.Model`` and coordinates one measurement model per data
view with a structural path matrix that defines which views should share latent
information.

During training, each measurement model produces deep latent variables (DLVs).
``StructuralModel`` appends either a Moore-Penrose ``FactorLayer`` or a
``ZCALayer`` to each measurement model, normalizes the latent variables, and
optimizes the views so that connected views agree while each view keeps
non-redundant latent dimensions.

The implementation uses Keras 3 and supports the TensorFlow and PyTorch
backends in the custom training loop.


Constructor
-----------

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
       run_from_config=False,
       is_siamese=False,
       diag_offset=1e-3,
       sparse_l1_list=0.0,
       attention_mse=False,
       attention_gate=0.3,
       order=False,
       order_association_cutoff=None,
       **kwargs,
   )

Parameters
~~~~~~~~~~

``Path``
    Binary or weighted adjacency matrix describing which views are connected.
    The usual shape is ``(n_views, n_views)``. A non-zero entry
    ``Path[i, j]`` means view ``i`` is trained against view ``j``.

``model_list``
    List of Keras measurement models, one per data view. ``StructuralModel``
    appends the DLVPM projection layer to these models unless loading from a
    serialized config.

``regularizer_list``
    List of Keras regularizers applied to the appended projection layer for
    each view. Use ``None`` for views without projection-layer regularization.

``tot_num``
    Total number of training samples. This is used by the appended projection
    layers when updating global covariance statistics.

``ndims``
    Number of DLV dimensions to learn per view.

``orthogonalization``
    Projection method used for the appended DLVPM layer. Supported values are
    ``"Moore-Penrose"`` and ``"zca"``. Moore-Penrose uses ``FactorLayer``.
    ZCA uses ``ZCALayer`` and enables the ordering options below.

``momentum``
    Momentum used when updating moving statistics such as covariance and
    ordering matrices.

``epsilon``
    Numerical stability constant used in normalization, covariance, and
    correlation calculations.

``train_DLV``
    If ``True``, target DLVs are computed with measurement models in training
    mode during training. If ``False``, target DLVs are computed in inference
    mode.

``run_from_config``
    Internal flag used during deserialization. Users normally leave this at
    ``False``.

``is_siamese``
    If ``True``, the first measurement model is wrapped once and reused for all
    views. This is useful when all views should share weights. In Siamese mode,
    all ``sparse_l1_list`` values must be identical.

``diag_offset``
    Diagonal jitter used by ``ZCALayer`` to keep covariance matrices
    well-conditioned. This only affects ``orthogonalization="zca"``.

``sparse_l1_list``
    Proximal L1 soft-thresholding strength for the appended projection weights.
    Use a scalar to apply the same threshold to all views, a list with one
    value per view, or ``0.0`` for no sparsity.

``attention_mse``
    If ``True``, reconstruction loss uses attention-weighted MSE instead of
    the standard masked MSE. Attention weights are based on per-dimension
    correlations to connected target views.

``attention_gate``
    Minimum correlation required for a connected target view to contribute to
    the attention-weighted MSE. Must lie in ``[-1, 1]``.

``order``
    If ``True``, enables structural ordering for ZCA DLVs. Ordering rotates the
    learned ZCA basis at the end of training so earlier dimensions capture more
    structural association. This option is only valid with
    ``orthogonalization="zca"``.

``order_association_cutoff``
    Optional cumulative association-mass cutoff used with ordered ZCA. Must lie
    in ``(0, 1]`` and requires ``order=True`` and
    ``orthogonalization="zca"``. When set, the model keeps the smallest number
    of ordered dimensions whose cumulative association strength reaches the
    cutoff, then rebuilds the appended ZCA layers with the reduced dimension.

``**kwargs``
    Extra keyword arguments forwarded to ``keras.Model``.


Training API
------------

``StructuralModel`` uses standard Keras entry points, but the loss is built
into the custom training loop. Compile the model with an optimizer object or a
list of optimizers.

.. code-block:: python

   struct_model.compile(
       [
           keras.optimizers.Adam(learning_rate=1e-3),
           keras.optimizers.Adam(learning_rate=1e-3),
       ]
   )

Use one optimizer per measurement model for ordinary multi-view models. For a
Siamese model with shared weights, a single optimizer is usually sufficient.

.. code-block:: python

   history = struct_model.fit(
       [view_a, view_b],
       batch_size=64,
       epochs=20,
   )

   metrics = struct_model.evaluate([view_a, view_b], verbose=False)
   dlvs = struct_model.predict([view_a, view_b], verbose=False)

``predict`` returns a tensor with shape ``(n_samples, ndims, n_views)``. If
ordered ZCA dimension pruning is enabled, ``ndims`` may be smaller after
training and will match the retained ordered dimensions.


Tracked Metrics
---------------

``fit`` and ``evaluate`` report the following metrics:

``total_loss``
    Mean training objective across views, including reconstruction loss and any
    projection-layer regularization.

``cross_metric``
    Mean Pearson correlation between connected views. Higher values indicate
    stronger agreement between structurally connected DLVs.

``mse_loss``
    Reconstruction loss between each source view and its connected target
    views. Missing rows and unconnected views are masked out.

``redundancy``
    Mean absolute off-diagonal within-view latent correlation. Lower values
    indicate more orthogonal, less redundant DLV dimensions.

``order_strength``
    Ordering diagnostic computed from the current structural association
    matrix. Values near ``1`` mean earlier DLV dimensions have stronger
    association than later dimensions more consistently. This metric is still
    reported when ``order=False``, but it is mainly useful when ordered ZCA is
    enabled.


Ordered ZCA
-----------

Set ``orthogonalization="zca"`` to use ZCA-orthogonalized DLVs. Set
``order=True`` to order the ZCA basis by structural association.

.. code-block:: python

   struct_model = StructuralModel(
       Path=Path,
       model_list=[model_a, model_b],
       regularizer_list=[None, None],
       tot_num=n_samples,
       ndims=16,
       orthogonalization="zca",
       order=True,
   )

During training, the model accumulates an association matrix from connected
views. At the end of ``fit``, a callback rotates the ZCA basis so the strongest
structural directions appear first.

To automatically keep only the most structurally associated ordered
dimensions, set ``order_association_cutoff``:

.. code-block:: python

   struct_model = StructuralModel(
       Path=Path,
       model_list=[model_a, model_b],
       regularizer_list=[None, None],
       tot_num=n_samples,
       ndims=32,
       orthogonalization="zca",
       order=True,
       order_association_cutoff=0.95,
   )

After training, inspect ``struct_model.ndims`` or
``struct_model.retained_order_dims`` to see how many ordered dimensions were
kept.

Important constraints:

* ``order=True`` requires ``orthogonalization="zca"``.
* ``order_association_cutoff`` requires both ``order=True`` and
  ``orthogonalization="zca"``.
* ``order_association_cutoff`` must be greater than ``0`` and less than or
  equal to ``1``.


Missing-View Data
-----------------

``StructuralModel`` can train and evaluate with samples that are missing entire
views. Mark a missing view by setting the whole row for that view to ``NaN``.
Rows with partial ``NaN`` values inside a view are rejected.

.. code-block:: python

   view_a = np.random.normal(size=(100, 20)).astype("float32")
   view_b = np.random.normal(size=(100, 15)).astype("float32")

   # Samples 10 through 19 are missing view B.
   view_b[10:20, :] = np.nan

   struct_model.fit([view_a, view_b], batch_size=32, epochs=10)

Missing rows are skipped when a view is encoded, scattered back as zero latent
rows internally, and then masked out of losses and correlation metrics. This
means a sample can still contribute through the views that are present.

Rules for missing data:

* Only all-``NaN`` rows are treated as missing views.
* Partial ``NaN`` rows raise an error.
* Non-floating inputs cannot contain ``NaN`` and are treated as fully present.
* For a multi-input measurement model, every tensor for that view must mark the
  same rows as missing.
* At least two present samples are needed for a connected pair to contribute to
  correlation metrics.


Attention-Weighted MSE
----------------------

The default reconstruction loss averages squared error over connected target
views according to ``Path``. With ``attention_mse=True``, the loss instead
weights connected target views by their per-dimension correlations to the
source view.

.. code-block:: python

   struct_model = StructuralModel(
       Path=Path,
       model_list=[model_a, model_b, model_c],
       regularizer_list=[None, None, None],
       tot_num=n_samples,
       ndims=8,
       attention_mse=True,
       attention_gate=0.25,
   )

Targets whose correlation is below ``attention_gate`` are gated out for that
dimension. The attention weights are detached from the gradient calculation, so
they reweight the MSE without directly optimizing the attention scores.


Genuine Sparsity
----------------

Setting ``sparse_l1_list`` enables proximal soft-thresholding on the projection
weights appended by ``StructuralModel``. Unlike ordinary L1 regularization,
soft-thresholding can produce exact zeros in the projection matrix.

.. code-block:: python

   struct_model = StructuralModel(
       Path=Path,
       model_list=[model_a, model_b],
       regularizer_list=[None, None],
       tot_num=n_samples,
       ndims=8,
       sparse_l1_list=[0.0, 5e-6],
   )

Notes:

* ``sparse_l1_list=0.0`` leaves behavior unchanged.
* A scalar applies the same threshold to every view.
* A list gives per-view thresholds and must match ``len(model_list)``.
* In Siamese mode, all sparse thresholds must be identical.


Minimal Example
---------------

.. code-block:: python

   import os

   os.environ.setdefault("KERAS_BACKEND", "tensorflow")

   import numpy as np
   import keras
   from keras import layers
   from deep_lvpm.model import StructuralModel

   n_samples = 512
   rng = np.random.default_rng(42)
   view_a = rng.normal(size=(n_samples, 8)).astype("float32")
   view_b = rng.normal(size=(n_samples, 6)).astype("float32")

   def make_measurement_model(input_dim, name):
       inputs = keras.Input(shape=(input_dim,), name=f"{name}_input")
       x = layers.Dense(16, activation="relu")(inputs)
       x = layers.Dense(16, activation="relu")(x)
       outputs = layers.Dense(8, name=f"{name}_features")(x)
       return keras.Model(inputs, outputs, name=name)

   model_a = make_measurement_model(view_a.shape[1], "view_a_encoder")
   model_b = make_measurement_model(view_b.shape[1], "view_b_encoder")

   Path = np.array(
       [
           [0, 1],
           [1, 0],
       ],
       dtype="float32",
   )

   struct_model = StructuralModel(
       Path=Path,
       model_list=[model_a, model_b],
       regularizer_list=[None, None],
       tot_num=n_samples,
       ndims=4,
       orthogonalization="zca",
       order=True,
   )

   optimizers = [
       keras.optimizers.Adam(learning_rate=1e-3),
       keras.optimizers.Adam(learning_rate=1e-3),
   ]
   struct_model.compile(optimizers)

   struct_model.fit(
       [view_a, view_b],
       batch_size=64,
       epochs=5,
       verbose=True,
   )

   metrics = struct_model.evaluate([view_a, view_b], verbose=False)
   print("Evaluation metrics:", metrics)

   dlvs = struct_model.predict([view_a, view_b], verbose=False)
   print("Latent tensor shape:", dlvs.shape)


Correlation Matrices
--------------------

Use ``calculate_corrmat`` to inspect cross-view correlations for each latent
dimension after prediction.

.. code-block:: python

   dlvs = struct_model.predict([view_a, view_b], verbose=False)
   corr_matrices = struct_model.calculate_corrmat(dlvs)

   for dim_index, corr_matrix in enumerate(corr_matrices):
       print("DLV dimension:", dim_index)
       print(corr_matrix)

``calculate_corrmat`` expects a 3D tensor with shape
``(n_samples, ndims, n_views)`` and returns one ``(n_views, n_views)``
correlation matrix per DLV dimension.
