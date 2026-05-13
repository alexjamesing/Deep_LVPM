Multimodal Models
=================

In addition to :class:`deep_lvpm.model.StructuralModel`, the package now
includes several alternative multi-view and multimodal representation-learning
models in :mod:`deep_lvpm.multi_model`. These are useful when you want to
compare DLVPM against other modern objectives on the same encoders and data.

The current models are:

- :class:`deep_lvpm.multi_model.CLIP`
- :class:`deep_lvpm.multi_model.DGCCA`
- :class:`deep_lvpm.multi_model.VICReg`
- :class:`deep_lvpm.multi_model.LeJEPA`

All four classes inherit from ``keras.Model`` and follow the same high-level
pattern as ``StructuralModel``:

- you provide one measurement model per data view through ``model_list``
- the class appends its own final projection layer of width ``ndims``
- you compile with one optimizer per view, or a single optimizer
- you train with the usual ``fit`` / ``evaluate`` / ``predict`` methods

Unlike DLVPM, these classes do **not** use a structural path matrix. They are
global multiview objectives rather than path-model-based latent variable models.


Common constructor pattern
--------------------------

The multimodal classes use a shared core constructor pattern:

.. code-block:: python

   SomeModel(
       model_list,
       regularizer_list,
       ndims,
       is_siamese=False,
       **method_specific_kwargs
   )

where:

* **model_list** (*list of keras.Model*): One measurement model per view.
* **regularizer_list** (*list*): Regularisers applied to the final projection
  head that each multimodal class adds internally.
* **ndims** (*int*): Width of the final shared embedding space.
* **is_siamese** (*bool, optional*): If ``True``, the first model is wrapped
  once and then shared across all views.

The standard workflow is therefore:

.. code-block:: python

   import keras
   from deep_lvpm.multi_model import CLIP

   model = CLIP(
       model_list=view_models,
       regularizer_list=[None for _ in view_models],
       ndims=512,
   )

   optimizers = [keras.optimizers.Adam(1e-4) for _ in view_models]
   model.compile(optimizers)
   model.fit(train_data, epochs=10)
   metrics = model.evaluate(test_data)
   embeddings = model.predict(test_data)

The main difference between the classes is therefore the **loss function** and
the training objective, not the user-facing training loop.


CLIP
----

:class:`deep_lvpm.multi_model.CLIP` is a contrastive model designed to align
different views in a shared embedding space by pulling matching pairs together
and pushing non-matching pairs apart.

Constructor
~~~~~~~~~~~

.. code-block:: python

   CLIP(
       model_list,
       regularizer_list,
       ndims,
       run_from_config=False,
       is_siamese=False,
       **kwargs
   )

How it works
~~~~~~~~~~~~

Each measurement model is given a final Dense projection layer of width
``ndims``. During training, all embeddings are L2-normalized and a CLIP-style
contrastive loss is computed across every ordered pair of views.

In practical terms:

- each sample in view ``m`` should be close to its matching sample in view ``n``
- all non-matching samples in the batch act as negatives
- a learned temperature parameter scales the logits

This is the most natural baseline when your task is cross-modal retrieval, such
as image-text matching.

Metrics
~~~~~~~

``CLIP.evaluate(...)`` reports:

- ``clip_loss``

Use CLIP when:

- your main goal is retrieval or matching
- you want a strong contrastive baseline
- you do not need a path model or orthogonal latent variables


DGCCA
-----

:class:`deep_lvpm.multi_model.DGCCA` implements Deep Generalized Canonical
Correlation Analysis.

Constructor
~~~~~~~~~~~

.. code-block:: python

   DGCCA(
       model_list,
       regularizer_list,
       ndims,
       gcca_reg=1e-3,
       momentum=0.0,
       eps=1e-6,
       center_outputs=True,
       run_from_config=False,
       is_siamese=False,
       **kwargs
   )

How it works
~~~~~~~~~~~~

DGCCA extends classical GCCA by learning nonlinear encoders for each view and
then solving a generalized correlation objective in the shared embedding space.
In this implementation:

- each view encoder produces an ``ndims``-wide embedding
- the embeddings are optionally mean-centered
- the model builds the GCCA projection system across all views jointly
- training minimizes the paper-style GCCA objective itself
- ``cross_metric`` is retained only as a sanity check and is not the optimized
  quantity
- that sanity check is computed on the per-view shared-space estimates
  :math:`U_j^T Y_j`, not on the raw encoder outputs

When ``momentum=0.0``, DGCCA uses the current batch covariance directly
and differentiates through the GCCA objective, which is the closest match to
the paper in this toolbox. If you set ``momentum`` above zero, the model
instead uses an exponential moving average of the per-view covariance matrices
as an optional stabilisation trick. That can help on noisy problems, but it is
no longer the pure paper formulation.

During training, DGCCA also stores running averages of the per-view latent
means and analytic :math:`U_j` projection matrices. Those stored quantities are
used to form clean out-of-sample shared projections on test data without
re-solving GCCA on the test set itself.

This makes DGCCA a natural baseline when your scientific question is mainly
about shared correlation structure across *all* views, rather than retrieval.

Metrics
~~~~~~~

``DGCCA.evaluate(...)`` reports:

- ``total_loss``
- ``cross_metric``
- ``gcca_loss``
- ``redundancy``

Unique helper
~~~~~~~~~~~~~

DGCCA also exposes ``calculate_corrmat(DLVs)``, which returns per-dimension
correlation matrices from the stacked output of ``predict``.

For DGCCA, ``predict(...)`` now returns the stacked shared-space estimates
obtained by applying the stored training-side :math:`U_j` maps to the new data.
``predict_shared(...)`` is kept as an explicit alias for the same behavior.

Use DGCCA when:

- you want a classical multiview correlation baseline
- you want all views to contribute jointly to one shared representation
- you want something closer in spirit to GCCA than to contrastive learning


VICReg
------

:class:`deep_lvpm.multi_model.VICReg` is a multi-view adaptation of the VICReg
objective.

Constructor
~~~~~~~~~~~

.. code-block:: python

   VICReg(
       model_list,
       regularizer_list,
       ndims,
       var_weight=25.0,
       inv_weight=25.0,
       cov_weight=1.0,
       gamma=1.0,
       run_from_config=False,
       is_siamese=False,
       eps=1e-4,
       **kwargs
   )

How it works
~~~~~~~~~~~~

VICReg balances three terms:

- **invariance**: corresponding views should have similar embeddings
- **variance**: each latent dimension should keep enough spread and avoid
  collapse
- **covariance**: different embedding dimensions should remain decorrelated

In this implementation, each view is compared against **every other view**
rather than against a path-masked subset. That makes it a full multiview
baseline rather than a path-model method.

Metrics
~~~~~~~

``VICReg.evaluate(...)`` reports:

- ``total_loss``
- ``cross_metric``
- ``mse_loss``  (the invariance term)
- ``redundancy``

Use VICReg when:

- you want a collapse-resistant self-supervised baseline
- you want to align all views without negative pairs
- you want a method that explicitly penalizes within-view redundancy


LeJEPA
------

:class:`deep_lvpm.multi_model.LeJEPA` is a multiview predictive model that
combines a JEPA-style prediction objective with a SIGReg regularizer.

Constructor
~~~~~~~~~~~

.. code-block:: python

   LeJEPA(
       model_list,
       regularizer_list,
       ndims,
       lambda_weight=0.05,
       num_global_views=None,
       num_slices=256,
       integration_min=-5.0,
       integration_max=5.0,
       integration_points=17,
       run_from_config=False,
       is_siamese=False,
       eps=1e-6,
       **kwargs
   )

How it works
~~~~~~~~~~~~

The LeJEPA objective used here has two parts:

1. a **prediction loss** that uses the first ``V_g`` views as global views and
   pulls every view toward the mean embedding of those global views
2. a **SIGReg** term that encourages the embeddings of each view to resemble an
   isotropic Gaussian distribution

If ``num_global_views`` is omitted, then all views are treated as global and
``V_l = 0``.

So, unlike CLIP, it does not rely on contrastive negatives; and unlike DGCCA,
it does not solve a global correlation eigensystem. It is a predictive,
distribution-regularized multiview objective.

Metrics
~~~~~~~

``LeJEPA.evaluate(...)`` reports:

- ``total_loss``
- ``cross_metric``
- ``pred_loss``
- ``sigreg_loss``
- ``redundancy``

Use LeJEPA when:

- you want a predictive multiview baseline rather than a contrastive one
- you want every view to predict a shared global-view target
- you want to regularize the embedding distribution directly


Choosing between DLVPM and the multimodal baselines
---------------------------------------------------

Use :class:`deep_lvpm.model.StructuralModel` when:

- you want an explicit path model
- you want orthogonal DLVs
- you want a latent-variable interpretation guided by known structure

Use the models in :mod:`deep_lvpm.multi_model` when:

- you want benchmark baselines without a path matrix
- you want to compare DLVPM against common multimodal objectives
- your priority is retrieval, shared embedding quality, or representation
  learning rather than path-model interpretation

In short:

- **CLIP**: best matched to retrieval and contrastive alignment
- **DGCCA**: best matched to classical shared-correlation modelling
- **VICReg**: best matched to redundancy-aware self-supervised alignment
- **LeJEPA**: best matched to predictive multiview alignment with distributional regularization
