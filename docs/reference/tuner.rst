Tuner
=====

``deep_lvpm.tuner`` provides a coordinate-descent hyperparameter tuner that
leverages :mod:`keras_tuner` for sampling while respecting the multi-view
structure of the DLVPM model.  Each measurement model (one per view) is
optimised in turn: the tuner freezes the best-known configuration for all other
views, samples a fresh set of hyperparameters for the current view (including
its projection-head sparsity and regularisation levels), trains the full
StructuralModel, measures the resulting inter-view correlations, and keeps the
configuration if the score improves.  This block-coordinate loop is repeated a
user-specified number of times, yielding a practical global architecture search
routine for DLVPM measurement models.

Class summary
-------------

.. py:class:: deep_lvpm.tuner.Tuner(view_builders, structural_kwargs, **options)

   * ``view_builders`` – list of callables ``builder(hp, view_index)`` that
     construct the measurement sub-models.  Each builder receives a
     :class:`keras_tuner.HyperParameters` instance and **must namespace its
     parameter names** (e.g., ``hp.Int(f"view{view_index}_units", ...)``) so
     that the tuner can cache/replay the best configuration for that view.
   * ``structural_kwargs`` – dictionary of arguments forwarded to
     :class:`deep_lvpm.model.StructuralModel`.  It must include ``Path``,
     ``tot_num``, ``ndims``, and optional defaults such as ``orthogonalization``
     or ``diag_offset``.  The sparse L1 and regulariser lists supplied here act
     as starting points for the search.
   * ``n_loops`` / ``max_trials_per_view`` – control how many global sweeps and
     per-view samples are evaluated.
   * ``sparse_config`` / ``regularizer_config`` – optional per-view search space
     definitions that are passed to :func:`sample_structural_hparams` to sample
     new ``sparse_l1_list``/``regularizer_list`` entries for the target view.

Usage example
-------------

.. code-block:: python

   import numpy as np
   import keras
   import keras_tuner as kt

   from deep_lvpm.tuner import Tuner

   def build_view(hp, view_index):
       units = hp.Int(f"view{view_index}_units", min_value=64, max_value=256, step=64)
       dropout = hp.Float(f"view{view_index}_dropout", 0.0, 0.5, step=0.1)

       inputs = keras.Input(shape=(128,))
       x = keras.layers.Dense(units, activation="relu")(inputs)
       x = keras.layers.Dropout(dropout)(x)
       outputs = keras.layers.Dense(32)(x)
       return keras.Model(inputs, outputs)

   Path = np.array([[0, 1],
                    [1, 0]], dtype="float32")

   structural_kwargs = dict(
       Path=Path,
       tot_num=2048,
       ndims=4,
       orthogonalization="Moore-Penrose",
       sparse_l1_list=[0.0, 0.0],
       regularizer_list=[None, None],
   )

   tuner = Tuner(
       view_builders=[build_view, build_view],
       structural_kwargs=structural_kwargs,
       n_loops=3,
       max_trials_per_view=5,
       sparse_config={"values": [0.0, 1e-6, 5e-6, 1e-5]},
       regularizer_config={
           "choices": ["none", "l2"],
           "l2_range": {"min": 1e-6, "max": 1e-3, "sampling": "log"},
       },
   )

   optimizers = [keras.optimizers.Adam(1e-3) for _ in range(2)]

   tuner.search(
       train_data=[view_a_train, view_b_train],
       optimizers=optimizers,
       validation_data=[view_a_val, view_b_val],
       fit_kwargs={"epochs": 5, "batch_size": 64, "verbose": 0},
   )

   best_model = tuner.build_best_model(optimizers)

The ``search`` method trains many short-lived StructuralModel instances.  Each
call follows this cycle:

1. Loop over the views; for the current view, sample architectural, sparsity,
   and regularisation hyperparameters with :mod:`keras_tuner`.
2. Build measurement models for all views (the sampled view plus cached best
   versions for the others) and instantiate a :class:`StructuralModel` with the
   updated ``sparse_l1_list``/``regularizer_list``.
3. Train the StructuralModel with the provided data and ``fit_kwargs``.
4. Predict latent factors on the validation data and compute the mean Pearson
   correlation between the tuned view and its connected partners.
5. If the correlation improved, cache the new hyperparameter set for that view
   (and its per-view sparsity/regulariser values).

After all loops finish, :meth:`Tuner.build_best_model` reconstructs a fresh
StructuralModel that can be trained for longer using the best combination of
measurement models and structural hyperparameters discovered during tuning.

Defining structural search spaces
---------------------------------

:func:`deep_lvpm.tuner.sample_structural_hparams` exposes per-view ranges for
genuine sparsity and regularisation.  ``sparse_config`` and
``regularizer_config`` can be either a single dict (broadcast to all views) or a
list with one dict per view.  Each dictionary may specify ``values`` (for
categorical sampling) or ``min``/``max``/``sampling`` for continuous sampling.
Regulariser configs support ``choices`` among ``"none"``, ``"l2"``, ``"l1"``,
and ``"l1l2"`` with independent ranges for the L1/L2 coefficients.  The sampled
values are inserted into the ``sparse_l1_list`` and ``regularizer_list`` that
feed :class:`StructuralModel`, ensuring that tuning per view also covers the
soft-thresholding strength and projection-layer penalties.

Because the tuner repeatedly builds complete StructuralModel instances, the
process can be computationally intensive.  Keep the number of loops, trials, and
training epochs modest for exploratory searches, then retrain the best model
configuration with a larger budget once convergence has been reached.
