Multimodal Models
=================

In addition to :class:`deep_lvpm.model.StructuralModel`, the package includes
alternative multi-view and multimodal representation-learning models in
:mod:`deep_lvpm.multi_model`:

- :class:`deep_lvpm.multi_model.CLIP`
- :class:`deep_lvpm.multi_model.DGCCA`
- :class:`deep_lvpm.multi_model.VICReg`
- :class:`deep_lvpm.multi_model.LeJEPA`

All four classes subclass ``torch.nn.Module`` and follow the same high-level
pattern as ``StructuralModel``:

- provide one ``torch.nn.Module`` measurement model per data view
- provide one regularizer entry per view
- choose ``ndims`` for the shared embedding width
- call ``compile`` with PyTorch optimizer objects
- train with ``fit`` and evaluate/predict with ``evaluate`` and ``predict``

Example
-------

.. code-block:: python

   import torch
   from deep_lvpm.multi_model import CLIP

   model = CLIP(
       model_list=view_models,
       regularizer_list=[None for _ in view_models],
       ndims=512,
   )

   optimizers = [
       torch.optim.Adam(view_model.parameters(), lr=1e-4)
       for view_model in model.model_list
   ]
   model.compile(optimizers)
   history = model.fit(train_data, epochs=10)
   metrics = model.evaluate(test_data)
   embeddings = model.predict(test_data)

The main difference between the classes is the loss function and training
objective, not the user-facing training loop.

Metrics
-------

``CLIP`` reports ``clip_loss``.

``VICReg`` reports ``total_loss``, ``cross_metric``, ``mse_loss``, and
``redundancy``.

``DGCCA`` reports ``total_loss``, ``cross_metric``, ``gcca_loss``, and
``redundancy``.

``LeJEPA`` reports ``total_loss``, ``cross_metric``, ``pred_loss``,
``sigreg_loss``, and ``redundancy``.
