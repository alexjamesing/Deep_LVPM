TCGA Lung Cancer Tutorial
=========================

This tutorial demonstrates how to integrate five lung cancer data modalities
from The Cancer Genome Atlas (TCGA): histology image features, RNA-seq,
methylation, miRNA, and somatic mutation data.  Each modality is encoded by a
small residual PyTorch module.  DLVPM then learns latent variables that follow
the structural path matrix defined below.

The full runnable script is :mod:`deep_lvpm.tutorial.tutorial_tcga`.

Run it with:

.. code-block:: bash

   python -m deep_lvpm.tutorial.tutorial_tcga

Prerequisites
-------------

Install :mod:`deep_lvpm` as described in :doc:`/installation`.  This tutorial
uses small packaged NumPy archives under ``deep_lvpm.data`` and can run on CPU,
although the default epoch count is intended for a real training run rather
than a minimal smoke test.

1. Load the multi-omics dataset
-------------------------------

The tutorial loads the training arrays from package data with
``importlib.resources``.  The order of ``X_arr`` matters because it must match
both the measurement model list and the path matrix.

.. code-block:: python

   import numpy as np
   import torch
   import torch.nn as nn
   from importlib import resources

   import deep_lvpm as DLVPM
   from deep_lvpm import regularizers
   from deep_lvpm.model import StructuralModel

   with resources.as_file(
       resources.files("deep_lvpm.data") / "Lung_multiomics_sample_train.npz"
   ) as f:
       arrays = np.load(f)
       rnaseq = arrays["rnaseq"]
       snv = arrays["snv"]
       methylation = arrays["methylation"]
       mirna = arrays["mirna"]
       histo20 = arrays["histo20"]

   X_arr = [histo20, rnaseq, methylation, mirna, snv]

2. Define residual measurement models
-------------------------------------

Each data type is a flat feature matrix, so the tutorial uses a fully connected
residual block.  The two linear layers are initialized as identity transforms.
This makes the starting model close to the input representation while still
allowing nonlinear residual updates during training.

.. code-block:: python

   class ResidualBlockModule(nn.Module):
       """Fully-connected residual block mirroring the original tutorial."""

       def __init__(
           self,
           input_dim: int,
           kernel_reg_l1: float = 0.01,
           kernel_reg_l2: float = 0.01,
           dropout_rate: float = 0.5,
       ) -> None:
           super().__init__()
           self.kernel_reg_l1 = kernel_reg_l1
           self.kernel_reg_l2 = kernel_reg_l2
           self.linear1 = nn.Linear(input_dim, input_dim)
           self.bn = nn.BatchNorm1d(input_dim)
           self.relu = nn.ReLU()
           self.linear2 = nn.Linear(input_dim, input_dim)
           self.dropout = nn.Dropout(dropout_rate)
           self.n_inputs = 1
           self._init_weights()

       def _init_weights(self) -> None:
           nn.init.eye_(self.linear1.weight)
           nn.init.eye_(self.linear2.weight)
           nn.init.zeros_(self.linear1.bias)
           nn.init.zeros_(self.linear2.bias)

       def forward(self, inputs: torch.Tensor) -> torch.Tensor:
           x = self.linear1(inputs)
           x = self.bn(x)
           x = self.relu(x)
           x = self.linear2(x)
           x = x + inputs
           x = self.dropout(x)
           return x

       def regularization_loss(self) -> torch.Tensor:
           penalty = torch.zeros(
               (),
               dtype=self.linear1.weight.dtype,
               device=self.linear1.weight.device,
           )
           if self.kernel_reg_l1:
               penalty = penalty + self.kernel_reg_l1 * (
                   torch.sum(torch.abs(self.linear1.weight))
                   + torch.sum(torch.abs(self.linear2.weight))
               )
           if self.kernel_reg_l2:
               penalty = penalty + self.kernel_reg_l2 * (
                   torch.sum(self.linear1.weight ** 2)
                   + torch.sum(self.linear2.weight ** 2)
               )
           return penalty

The helper below keeps the model list easy to read.  DLVPM will append the
final DLVPM projection layer to each residual block.

.. code-block:: python

   def residual_block(
       input_dim: int,
       kernel_reg_l1: float = 0.01,
       kernel_reg_l2: float = 0.01,
       dropout_rate: float = 0.5,
       name: str = "residual_block",
   ) -> nn.Module:
       return ResidualBlockModule(
           input_dim=input_dim,
           kernel_reg_l1=kernel_reg_l1,
           kernel_reg_l2=kernel_reg_l2,
           dropout_rate=dropout_rate,
       )

   model_list = [
       residual_block(histo20.shape[1], name="histo20_enc"),
       residual_block(rnaseq.shape[1], name="rnaseq_enc"),
       residual_block(methylation.shape[1], name="meth_enc"),
       residual_block(mirna.shape[1], name="mirna_enc"),
       residual_block(snv.shape[1], name="snv_enc"),
   ]

3. Specify the structural path matrix
-------------------------------------

The path matrix encodes the scientific assumption about which modalities should
share latent information.  In this tutorial RNA-seq is the central molecular
view: histology, methylation, miRNA, and SNV are each connected to RNA-seq, but
not directly to one another.

.. code-block:: python

   ndims = 5

   Path = np.array(
       [
           [0, 1, 0, 0, 0],
           [1, 0, 1, 1, 1],
           [0, 1, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 1, 0, 0, 0],
       ],
       dtype="float32",
   )

   batch_size = 256
   epochs = 300
   total_steps = int(rnaseq.shape[0] / batch_size) * epochs

   init_lr, final_lr = 1e-4, 1e-5
   lr_gamma = (final_lr / init_lr) ** (1.0 / max(epochs, 1))
   tot_num = rnaseq.shape[0]

4. Build and compile the model
------------------------------

The current tutorial uses ZCA orthogonalization with ordered latent variables.
``order=True`` means the learned ZCA basis is ordered by the structural
association accumulated during training.

.. code-block:: python

   regularizer_list = [
       regularizers.l1_l2(l1=0.001, l2=0.001),
       regularizers.l1_l2(l1=0.001, l2=0.001),
       regularizers.l1_l2(l1=0.001, l2=0.001),
       regularizers.l1_l2(l1=0.001, l2=0.001),
       regularizers.l1_l2(l1=0.001, l2=0.001),
   ]

   DLVPM_Structural_instance = StructuralModel(
       Path,
       model_list,
       regularizer_list,
       tot_num,
       ndims,
       momentum=0.95,
       epsilon=0.001,
       orthogonalization="zca",
       train_DLV=True,
       order=True,
   )

Compile with one PyTorch optimizer per wrapped view model.  The learning-rate
schedulers are passed to ``fit`` so they step once per epoch.

.. code-block:: python

   opt_list = [
       torch.optim.Adam(model.parameters(), lr=init_lr)
       for model in DLVPM_Structural_instance.model_list
   ]
   scheduler_list = [
       torch.optim.lr_scheduler.ExponentialLR(opt, gamma=lr_gamma)
       for opt in opt_list
   ]
   DLVPM_Structural_instance.compile(optimizer=opt_list)

5. Train and evaluate on the training cohort
--------------------------------------------

``fit`` reports the expanded DLVPM metrics during training.  ``evaluate``
returns a dictionary; ``cross_metric`` is the mean correlation between
structurally connected views.

.. code-block:: python

   DLVPM_Structural_instance.fit(
       X_arr,
       batch_size=batch_size,
       epochs=epochs,
       verbose=True,
       schedulers=scheduler_list,
   )
   mean_corr = DLVPM_Structural_instance.evaluate(X_arr)

   print(
       "The mean correlation between data-types connected by the path model is r="
       + str(mean_corr["cross_metric"])
   )

6. Evaluate on the held-out test cohort
---------------------------------------

The test arrays are loaded in the same modality order.  The example evaluates
the model and then calls ``predict`` to obtain the DLV tensor for all test
samples.

.. code-block:: python

   with resources.as_file(
       resources.files("deep_lvpm.data") / "Lung_multiomics_sample_test.npz"
   ) as f:
       arrays = np.load(f)
       rnaseq_test = arrays["rnaseq"]
       snv_test = arrays["snv"]
       methylation_test = arrays["methylation"]
       mirna_test = arrays["mirna"]
       histo20_test = arrays["histo20"]

   X_arr_test = [histo20_test, rnaseq_test, methylation_test, mirna_test, snv_test]
   mean_corr_test = DLVPM_Structural_instance.evaluate(X_arr_test)

   print(
       "The mean correlation between data-types connected by the path model is r="
       + str(mean_corr_test["cross_metric"])
   )

   test_DLVs = DLVPM_Structural_instance.predict(X_arr_test)
   print(np.corrcoef(test_DLVs[:, 0, :].T))
   print(np.corrcoef(test_DLVs[:, 1, :].T))

7. Visualise cross-modal correlations
-------------------------------------

``calculate_corrmat`` returns one modality-by-modality correlation matrix per
DLV.  The plotting helper renders those matrices as chord diagrams, which makes
it easier to inspect which modalities are linked by each latent factor.

.. code-block:: python

   corr_mat = DLVPM_Structural_instance.calculate_corrmat(test_DLVs)

   from deep_lvpm.plot import plot_correlation_chord_row

   data_names = ["Histology", "RNASeq", "miRNASeq", "Methylation", "SNVs"]

   fig, ax = plot_correlation_chord_row(
       corr_mat,
       data_names,
       min_corr=0,
       node_cmap_name="Pastel1",
       figure_title="Correlation Plots Between Omics and Imaging Data Types in Lung Cancer",
       show_edge_labels=True,
       dpi=300,
       show=True,
   )

Summary
-------

This tutorial is the standard multi-view DLVPM pattern for tabular biological
data: load aligned views, build one PyTorch encoder per view, encode the
scientific structure in ``Path``, train a ``StructuralModel``, and inspect
cross-modal DLV correlations on held-out data.
