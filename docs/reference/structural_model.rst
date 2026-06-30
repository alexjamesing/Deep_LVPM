StructuralModel
===============

The :class:`deep_lvpm.model.StructuralModel` class is the core DLVPM model.  It
subclasses ``torch.nn.Module`` and coordinates one PyTorch measurement model
per data view.  Each measurement model produces features for one view, and
``StructuralModel`` appends the final DLVPM projection layer that turns those
features into deep latent variables (DLVs).

The public methods ``compile``, ``fit``, ``evaluate``, and ``predict`` are
Deep LVPM convenience methods.  Internally they use ordinary PyTorch concepts:
``model.train()``, forward passes, loss calculation, ``loss.backward()``,
``optimizer.step()``, ``model.eval()``, and ``torch.no_grad()``.

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
       order=False,
       order_association_cutoff=None,
       missing_strategy="impute",
       latent_imputation_rank=2,
       latent_imputation_iterations=3,
       device=None,
       **kwargs,
   )

Parameters
~~~~~~~~~~

``Path``
    Binary or weighted adjacency matrix with shape ``(n_views, n_views)``.
    A non-zero entry ``Path[i, j]`` means view ``i`` is trained to agree with
    view ``j``.

``model_list``
    List of ``torch.nn.Module`` measurement models.  There is normally one
    module per data view.  The modules should return feature tensors with shape
    ``(batch, features)``; higher-dimensional outputs are flattened before the
    DLVPM projection layer.

``regularizer_list``
    One regularizer entry per view.  Use ``None`` for views without projection
    regularization.  Regularizers come from :mod:`deep_lvpm.regularizers`.

``tot_num``
    Total number of training samples.  The DLVPM projection layers use this for
    internal covariance and normalization statistics.

``ndims``
    Number of DLV dimensions to learn per view.

``orthogonalization``
    Projection method for the appended DLVPM layer.  ``"Moore-Penrose"`` uses
    :class:`deep_lvpm.layers.FactorLayer.FactorLayer`; ``"zca"`` uses
    :class:`deep_lvpm.layers.ZCALayer.ZCALayer`.

``momentum`` and ``epsilon``
    Numerical settings used by the projection layers and correlation
    calculations.  ``momentum`` controls moving statistics; ``epsilon`` avoids
    division by zero.

``train_DLV``
    If ``True``, target DLVs are computed with measurement models in training
    mode during training.  If ``False``, target DLVs are computed in evaluation
    mode.

``is_siamese``
    If ``True``, the first measurement model is wrapped once and reused for all
    views.  This is used when two or more views should share the same encoder
    weights, as in the CIFAR-10 Siamese tutorial.

``diag_offset``
    Diagonal jitter used by ``ZCALayer`` to keep covariance matrices
    well-conditioned.

``sparse_l1_list``
    Proximal L1 soft-thresholding strength for the appended projection weights.
    Use a scalar for all views, a list with one value per view, or ``0.0`` for
    no soft-thresholding.

``order`` and ``order_association_cutoff``
    Ordered ZCA options.  ``order=True`` accumulates a structural association
    matrix during training and orders DLV dimensions by shared structure at the
    end of ``fit``.  ``order_association_cutoff`` optionally keeps the smallest
    number of ordered dimensions needed to reach the requested cumulative
    association mass.

``missing_strategy``
    Missing-view strategy used during training.  ``"impute"`` is the default
    and completes missing DLVs, not raw input features, before the structural
    losses and training metrics are calculated.  ``"project"`` keeps the
    previous behavior, where missing views are zero-scattered and masked out.
    ``evaluate()`` and ``predict()`` always use projection behavior.

``latent_imputation_rank`` and ``latent_imputation_iterations``
    Settings for ``missing_strategy="impute"``.  DLVPM completes missing
    latent view blocks with batch-local low-rank SVD completion using the given
    rank and number of refinement iterations.

``device``
    Optional PyTorch device.  If omitted, DLVPM uses CUDA when available and
    otherwise CPU.

Minimal example
---------------

This example creates two tabular views, defines one encoder per view, connects
the views with a symmetric path matrix, and trains a ZCA-orthogonalized DLVPM
model.

.. code-block:: python

   import numpy as np
   import torch
   import torch.nn as nn

   from deep_lvpm.model import StructuralModel

   n_samples = 512
   rng = np.random.default_rng(42)
   view_a = rng.normal(size=(n_samples, 8)).astype("float32")
   view_b = rng.normal(size=(n_samples, 6)).astype("float32")

   class Encoder(nn.Module):
       def __init__(self, input_dim):
           super().__init__()
           self.net = nn.Sequential(
               nn.Linear(input_dim, 32),
               nn.ReLU(),
               nn.Linear(32, 16),
               nn.ReLU(),
           )
           self.n_inputs = 1

       def forward(self, inputs):
           return self.net(inputs)

   model_a = Encoder(view_a.shape[1])
   model_b = Encoder(view_b.shape[1])

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
       torch.optim.Adam(view_model.parameters(), lr=1e-3)
       for view_model in struct_model.model_list
   ]
   struct_model.compile(optimizer=optimizers)

   history = struct_model.fit(
       [view_a, view_b],
       batch_size=64,
       epochs=5,
       verbose=True,
   )

   metrics = struct_model.evaluate([view_a, view_b], verbose=False)
   dlvs = struct_model.predict([view_a, view_b], verbose=False)

   print(metrics)
   print(dlvs.shape)

Training API
------------

``compile`` accepts either a single PyTorch optimizer or a list of optimizers.
For ordinary multi-view models, use one optimizer per wrapped view model:

.. code-block:: python

   optimizers = [
       torch.optim.Adam(view_model.parameters(), lr=1e-4)
       for view_model in struct_model.model_list
   ]
   struct_model.compile(optimizer=optimizers)

For Siamese models with shared weights, it is valid to provide the same
optimizer object for each shared view, as shown in the Siamese tutorial.

``fit`` accepts NumPy arrays, tensors, or a PyTorch ``DataLoader``.  For array
or tensor inputs, pass one item per model input:

.. code-block:: python

   history = struct_model.fit(
       [view_a, view_b],
       batch_size=64,
       epochs=20,
       validation_split=0.1,
       shuffle=True,
   )

When using schedulers, pass a list through ``schedulers``.  Each scheduler is
stepped once per epoch after the training batches finish.

.. code-block:: python

   schedulers = [
       torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
       for optimizer in optimizers
   ]
   struct_model.fit([view_a, view_b], schedulers=schedulers)

Metrics
-------

``fit`` and ``evaluate`` report a dictionary with these keys:

``total_loss``
    Mean DLVPM objective across views, including reconstruction loss and any
    projection-layer regularization.

``cross_metric``
    Mean Pearson correlation between structurally connected views.  Higher
    values indicate stronger cross-view agreement.

``mse_loss``
    Reconstruction loss between each source view and its connected target
    views.

``redundancy``
    Mean absolute off-diagonal within-view latent correlation.  Lower values
    indicate less redundant DLV dimensions.

``order_strength``
    Present only when ``order=True``.  This diagnostic summarizes whether
    earlier ordered dimensions carry stronger structural association than later
    dimensions.

Prediction and correlation matrices
-----------------------------------

``predict`` returns a NumPy array with shape ``(n_samples, ndims, n_views)``.
If ordered ZCA dimension pruning is enabled, the second dimension may be
smaller than the original ``ndims`` after training.

Use ``calculate_corrmat`` to inspect one cross-view correlation matrix per DLV
dimension:

.. code-block:: python

   dlvs = struct_model.predict([view_a, view_b], verbose=False)
   corr_matrices = struct_model.calculate_corrmat(dlvs)

   for dim_index, corr_matrix in enumerate(corr_matrices):
       print("DLV dimension:", dim_index)
       print(corr_matrix)

Missing views
-------------

``StructuralModel`` supports samples where an entire view is missing.  Mark a
missing view by setting the whole row for that view to ``NaN``.  Rows with
partial ``NaN`` values inside a view are rejected.

.. code-block:: python

   view_a = np.random.normal(size=(100, 20)).astype("float32")
   view_b = np.random.normal(size=(100, 15)).astype("float32")

   # Samples 10 through 19 are missing view B.
   view_b[10:20, :] = np.nan

   struct_model.fit([view_a, view_b], batch_size=32, epochs=10)

During training, the default ``missing_strategy="impute"`` skips missing raw
views when the corresponding encoder is run, then fills the missing DLVs with
detached batch-local low-rank latent imputation.  The completed DLVs are used
for normalization, structural losses, redundancy, and training metrics, while
encoder updates still come only from rows where that source view has real raw
input.

Set ``missing_strategy="project"`` to use the older projection behavior:
missing view rows are zero-scattered, and only observed views contribute to
losses and metrics.  Regardless of the training strategy, ``evaluate()`` and
``predict()`` do not impute.  ``predict()`` returns zeros for missing views.
Rows where every view is missing are rejected because there is no observed
information to condition on.

Multi-input views
-----------------

Some measurement models need more than one tensor.  For example, a text encoder
may need token IDs and an attention mask.  Set ``n_inputs`` on that encoder so
``StructuralModel`` knows how many consecutive tensors belong to that view.

.. code-block:: python

   class TextEncoder(nn.Module):
       def __init__(self):
           super().__init__()
           self.n_inputs = 2

       def forward(self, inputs):
           input_ids, attention_mask = inputs
           ...

   model_list = [image_encoder, text_encoder]
   struct_model = StructuralModel(
       Path=Path,
       model_list=model_list,
       regularizer_list=[None, None],
       tot_num=n_samples,
       ndims=128,
   )

   # One image tensor, then the two tensors consumed by text_encoder.
   struct_model.fit([images, token_ids, attention_masks])

Saving
------

Use PyTorch checkpoints.  ``save`` stores both the model configuration and the
``state_dict``.  If you only need raw weights, use ``save_state_dict`` or
``torch.save(model.state_dict(), path)``.

.. code-block:: python

   struct_model.save("/path/to/DLVPM_Model.pt")
   struct_model.save_state_dict("/path/to/DLVPM_Model_state_dict.pt")

   restored = StructuralModel(
       Path=Path,
       model_list=[model_a, model_b],
       regularizer_list=[None, None],
       tot_num=n_samples,
       ndims=4,
       orthogonalization="zca",
   )
   restored.load_state_dict_from_path("/path/to/DLVPM_Model.pt")

Plotting the structural model
-----------------------------

``plot_structural_model`` writes the path graph to a PNG file using ``pydot``.

.. code-block:: python

   struct_model.plot_structural_model("path_model.png")
