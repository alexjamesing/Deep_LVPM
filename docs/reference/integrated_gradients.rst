Integrated Gradients
====================

Deep LVPM includes an integrated gradients helper for feature attribution on
trained :class:`deep_lvpm.model.StructuralModel` instances.  It explains one
DLV dimension at a time by measuring how each input feature contributes to the
selected DLV-specific structural loss as the input moves from a baseline value
to the observed value.

The implementation lives in :mod:`deep_lvpm.integrated_gradients`:

.. code-block:: python

   from deep_lvpm.integrated_gradients import calculate_integrated_gradients

   attributions = calculate_integrated_gradients(
       structural_model,
       data,
       baseline=0.0,
       dlv_index=0,
       steps=50,
   )

``data`` should be the same flat list of arrays or tensors that you pass to
``StructuralModel.fit()``, ``StructuralModel.evaluate()``, or
``StructuralModel.predict()``.  The returned attribution values have the same
structure and shapes as ``data``.

Function Arguments
------------------

``structural_model``
    A trained ``StructuralModel``.

``data``
    Input arrays or tensors in model-input order.  Inputs must be floating
    point.  Missing all-``NaN`` view rows are handled in the same way as the
    model handles missing views during evaluation.

``baseline``
    A scalar baseline such as ``0.0`` or a list of baseline arrays matching the
    input list.  Per-view baselines can either have the same shape as the input
    data or broadcast to that shape.  For tabular data, a one-dimensional
    feature baseline vector is often the most convenient form.

``dlv_index``
    Zero-based DLV dimension to explain.  ``dlv_index=0`` explains DLV1.

``steps``
    Number of interpolation steps between the baseline and observed data.
    Larger values give a more accurate approximation and take longer to run.

``explain_loss_reduction``
    If ``True`` by default, positive attribution means the feature helps reduce
    the selected DLV structural loss relative to the baseline.  Set this to
    ``False`` to attribute the loss itself.

``return_numpy``
    If ``True`` by default, returns NumPy arrays.  Set this to ``False`` to
    keep PyTorch tensors.

Choosing Baselines
------------------

Baselines should be defined on the same preprocessed scale as the model inputs.
There is no single best baseline for every data type, but the following choices
are often practical:

- Use a zero baseline for binary or presence/absence features when zero means
  absence.  The TCGA survival example uses zero baselines for CNV and SNV.
- Use a feature-wise mean baseline for continuous views when the question is
  how each subject deviates from an average subject.  The TCGA survival example
  uses mean vectors across subjects for RNA-seq, miRNA-seq, and methylation.
- Avoid baselines that are outside the preprocessing distribution seen by the
  model unless that contrast is intentional.

Aggregating Feature Importance
------------------------------

Integrated gradients are returned per subject and per feature.  For factor
interpretation, a feature can push a DLV upward in one subject and downward in
another.  If you average signed values directly, important features can cancel
out:

.. code-block:: python

   mean_signed_ig = np.nanmean(view_attributions, axis=0)

For ranking loci or features that define a DLV factor, rank by the mean
absolute attribution across subjects:

.. code-block:: python

   mean_abs_ig = np.nanmean(np.abs(view_attributions), axis=0)
   top_indices = np.argsort(mean_abs_ig)[-10:][::-1]

It is still useful to keep the signed mean alongside the absolute importance,
because the signed mean shows whether the attribution is directionally biased
across the cohort:

.. code-block:: python

   summary = pd.DataFrame({
       "feature_id": [feature_names[i] for i in top_indices],
       "mean_abs_integrated_gradient": mean_abs_ig[top_indices],
       "mean_signed_integrated_gradient": mean_signed_ig[top_indices],
   })

Example With Multiple Omics Views
---------------------------------

This example explains DLV1 for a trained five-view TCGA-style model.  CNV and
SNV use zero baselines, while continuous views use feature-wise means.

.. code-block:: python

   import numpy as np
   import pandas as pd

   from deep_lvpm.integrated_gradients import calculate_integrated_gradients

   view_names = [
       "rnaseq_gene",
       "mirna_gene",
       "methylation_gene",
       "cnv_gene",
       "mutation_gene",
   ]

   data = [rnaseq, mirnaseq, methylation, cnv, snv]
   baselines = [
       rnaseq.mean(axis=0).astype("float32"),
       mirnaseq.mean(axis=0).astype("float32"),
       methylation.mean(axis=0).astype("float32"),
       np.zeros(cnv.shape[1], dtype="float32"),
       np.zeros(snv.shape[1], dtype="float32"),
   ]

   attributions = calculate_integrated_gradients(
       structural_model,
       data,
       baseline=baselines,
       dlv_index=0,
       steps=50,
   )

   rows = []
   for view_name, view_attributions, feature_names in zip(
       view_names,
       attributions,
       feature_name_lists,
   ):
       mean_abs_ig = np.nanmean(np.abs(view_attributions), axis=0)
       mean_signed_ig = np.nanmean(view_attributions, axis=0)
       top_indices = np.argsort(mean_abs_ig)[-10:][::-1]

       for rank, feature_index in enumerate(top_indices, start=1):
           rows.append({
               "view": view_name,
               "rank": rank,
               "feature_id": feature_names[feature_index],
               "feature_column": int(feature_index),
               "mean_abs_integrated_gradient": float(mean_abs_ig[feature_index]),
               "mean_signed_integrated_gradient": float(mean_signed_ig[feature_index]),
           })

   top_loci = pd.DataFrame(rows)

The TCGA survival tutorial uses this pattern after fitting the DLVPM survival
model and writes one top-loci table per DLV.

Practical Notes
---------------

- Integrated gradients are an evaluation-time attribution method.  Train the
  ``StructuralModel`` first, then calculate attributions.
- Runtime scales with the number of views, samples, features, DLV dimensions,
  and interpolation ``steps``.  Use a small ``steps`` value for smoke tests and
  increase it for final analyses.
- The helper snapshots and restores the model state while it runs, because
  DLVPM projection layers keep moving statistics.
- Always interpret attributions in the context of the chosen baseline and the
  preprocessing applied before model training.
