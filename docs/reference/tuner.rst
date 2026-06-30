Tuner
=====

Deep LVPM includes a small internal coordinate descent tuner for PyTorch
``StructuralModel`` workflows.  It takes view-builder functions, samples
view-specific hyperparameters, trains candidate ``StructuralModel`` instances,
and retains the best configuration.

The local :class:`deep_lvpm.tuner.HyperParameters` helper supports the common
methods ``Choice``, ``Float``, and ``Int``.  The
:class:`deep_lvpm.tuner.Tuner` class coordinates view-by-view search using
native PyTorch modules and optimizers.

Example
-------

.. code-block:: python

   import torch
   from deep_lvpm.tuner import HyperParameters, Tuner

   def build_view(hp: HyperParameters, view_index: int):
       hidden = hp.Int(f"view{view_index}_hidden", 32, 128, step=32)
       return torch.nn.Sequential(
           torch.nn.Linear(input_widths[view_index], hidden),
           torch.nn.ReLU(),
       )

   tuner = Tuner(
       view_builders=[build_view, build_view],
       structural_kwargs={
           "Path": Path,
           "tot_num": n_samples,
           "ndims": 8,
           "orthogonalization": "zca",
       },
       max_trials_per_view=3,
   )

   tuner.search(
       train_data,
       optimizers=torch.optim.Adam(torch.nn.Linear(1, 1).parameters(), lr=1e-3),
       validation_data=validation_data,
   )
