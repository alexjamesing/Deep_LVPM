TCGA Pan-Cancer Survival Tutorial
=================================

This tutorial trains several multi-omic models on a pan-cancer TCGA survival
dataset and evaluates whether the learned representations predict clinical
outcome.  The script compares:

- DLVPM representations followed by penalised Cox regression
- a direct multimodal neural Cox model
- a direct linear Cox model fitted on the omics features
- CLIP, VICReg, LeJEPA, and DGCCA representations followed by penalised Cox

The full runnable script is :mod:`deep_lvpm.tutorial.tutorial_tcga_survival`.

Run it with:

.. code-block:: bash

   python -m deep_lvpm.tutorial.tutorial_tcga_survival

For a short syntax and smoke run, use:

.. code-block:: bash

   DLVPM_SURVIVAL_SMOKE_TEST=1 python -m deep_lvpm.tutorial.tutorial_tcga_survival

Prerequisites
-------------

Install :mod:`deep_lvpm` as described in :doc:`/installation`.  This tutorial
uses ``pandas`` and ``lifelines`` for survival modelling.  The first run
downloads a compact tutorial archive from Zenodo into
``deep_lvpm/data/dlvpm_tcga_survival_demo`` and reuses cached preprocessed
arrays on later runs.

1. Configure the run
--------------------

The script keeps the main user-editable settings near the top.  The default
clinical endpoint is ``pfi`` (progression-free interval).  The smoke-test
environment variable reduces the run to one epoch and one baseline.

.. code-block:: python

   from pathlib import Path

   RANDOM_SEED = 42
   SURVIVAL_ENDPOINT = "pfi"

   PACKAGE_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
   DATA_DIR = PACKAGE_DATA_DIR / "dlvpm_tcga_survival_demo"
   CACHE_DIR = DATA_DIR / "preprocessed_omics_cache"
   DATA_URL = "https://zenodo.org/records/20305527/files/dlvpm_tcga_survival_demo.zip?download=1"
   DATA_ZIP = PACKAGE_DATA_DIR / "dlvpm_tcga_survival_demo.zip"

   NDIMS = 100
   BATCH_SIZE = 1024
   EPOCHS = 100
   LEARNING_RATE = 1e-4
   BOOTSTRAP_SAMPLES = 1000
   PERMUTATION_SAMPLES = 1000
   SIGNIFICANCE_LEVEL = 0.05

   MULTIMODAL_METHODS = ["CLIP", "VICReg", "LeJEPA", "DGCCA"]

   RESIDUAL_ENCODER_LATENT_DIM = 256
   RESIDUAL_BLOCK_HIDDEN_DIM = 256
   RESIDUAL_DEPTH = 1
   RESIDUAL_DROPOUT = 0.30

   NEURAL_COX_DROPOUT = 0.60
   NEURAL_COX_L2 = 1e-2
   LINEAR_COX_L2 = 1e-3
   COX_PENALIZER = 0.10
   COX_L1_RATIO = 0.00

   RUN_INTEGRATED_GRADIENTS = True
   INTEGRATED_GRADIENTS_DLV_INDICES = [0, 1]
   INTEGRATED_GRADIENTS_STEPS = 50
   INTEGRATED_GRADIENTS_TOP_N = 10
   INTEGRATED_GRADIENTS_OUTPUT_DIR = DATA_DIR

   import os
   if os.environ.get("DLVPM_SURVIVAL_SMOKE_TEST") == "1":
       EPOCHS = 1
       MULTIMODAL_METHODS = ["CLIP"]
       INTEGRATED_GRADIENTS_STEPS = 2

2. Import dependencies and seed PyTorch
---------------------------------------

The tutorial uses native PyTorch for the neural models and ``lifelines`` for
the downstream Cox proportional hazards models.

.. code-block:: python

   import gc
   import json
   import random
   import urllib.request
   import zipfile

   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from lifelines import CoxPHFitter
   from lifelines.utils import concordance_index
   from torch.utils.data import DataLoader, TensorDataset

   from deep_lvpm import regularizers as dlvpm_regularizers
   from deep_lvpm.integrated_gradients import calculate_integrated_gradients
   from deep_lvpm.model import StructuralModel
   from deep_lvpm.multi_model import CLIP, DGCCA, LeJEPA, VICReg

   random.seed(RANDOM_SEED)
   np.random.seed(RANDOM_SEED)
   torch.manual_seed(RANDOM_SEED)
   if torch.cuda.is_available():
       torch.cuda.manual_seed_all(RANDOM_SEED)

3. Define the residual encoder
------------------------------

Each omics view has a different number of features.  The tutorial builds one
``ResidualEncoder`` per view.  DLVPM skips all-``NaN`` missing views before
encoding them, but the encoder also detects all-``NaN`` rows, replaces
``NaN`` values with zeros for computation, and masks the output back to zero so
the same encoder can be reused by the direct Cox baseline.

.. code-block:: python

   def residual_encoder(input_dim, name):
       return ResidualEncoder(input_dim=input_dim, name=name)

   class ResidualEncoder(nn.Module):
       def __init__(self, input_dim, name):
           super().__init__()
           self.input_dim = int(input_dim)
           self.name = name
           self.input_norm = nn.LayerNorm(input_dim)
           self.blocks = nn.ModuleList()
           for block_index in range(RESIDUAL_DEPTH):
               self.blocks.append(ResidualBlock(input_dim, f"{name}_residual_{block_index + 1}"))
           self.head_norm = nn.LayerNorm(input_dim)
           self.latent = nn.Linear(input_dim, RESIDUAL_ENCODER_LATENT_DIM)
           self.latent_dropout = nn.Dropout(RESIDUAL_DROPOUT)
           self.n_inputs = 1

       def forward(self, inputs):
           present = torch.any(torch.logical_not(torch.isnan(inputs)), dim=1, keepdim=True).to(inputs.dtype)
           x = torch.where(torch.isnan(inputs), torch.zeros_like(inputs), inputs)
           x = self.input_norm(x)

           for block in self.blocks:
               x = block(x)

           x = self.head_norm(x)
           x = self.latent(x)
           x = F.gelu(x)
           x = self.latent_dropout(x)
           outputs = x * present
           return outputs

``ResidualBlock`` is an ordinary residual MLP block.  Increase or decrease
``RESIDUAL_DEPTH`` near the top of the script to stack more or fewer blocks.

.. code-block:: python

   class ResidualBlock(nn.Module):
       def __init__(self, input_dim, name):
           super().__init__()
           self.name = name
           self.norm = nn.LayerNorm(input_dim)
           self.linear1 = nn.Linear(input_dim, RESIDUAL_BLOCK_HIDDEN_DIM)
           self.linear2 = nn.Linear(RESIDUAL_BLOCK_HIDDEN_DIM, input_dim)
           self.dropout = nn.Dropout(RESIDUAL_DROPOUT)

       def forward(self, inputs):
           h = self.norm(inputs)
           h = self.linear1(h)
           h = F.gelu(h)
           h = self.dropout(h)
           h = self.linear2(h)
           h = self.dropout(h)
           return inputs + h

4. Define Cox survival helpers
------------------------------

The direct neural Cox model uses a PyTorch implementation of the partial
likelihood loss.  The representation-learning methods use ``lifelines`` Cox
models fitted on the learned patient-level features.

.. code-block:: python

   def cox_partial_likelihood_loss(y_true, y_pred):
       times = y_true[:, 0]
       events = y_true[:, 1]
       risks = y_pred.reshape(-1)

       order = torch.argsort(-times)
       events = torch.take(events, order)
       risks = torch.take(risks, order)

       log_cumulative_hazard = torch.log(torch.cumsum(torch.exp(risks), dim=0) + 1e-8)
       log_likelihood = (risks - log_cumulative_hazard) * events
       return -torch.sum(log_likelihood) / (torch.sum(events) + 1e-8)

.. code-block:: python

   def fit_penalised_cox(method_name, train_features, test_features):
       mean = train_features.mean(axis=0)
       std = train_features.std(axis=0)
       std[(~np.isfinite(std)) | (std < 1e-6)] = 1.0
       train_features = ((train_features - mean) / std).astype("float32")
       test_features = ((test_features - mean) / std).astype("float32")

       feature_columns = [f"feature_{i + 1:03d}" for i in range(train_features.shape[1])]
       train_df = pd.DataFrame(train_features, columns=feature_columns)
       train_df["time"] = train_times
       train_df["event"] = train_events

       cox_model = CoxPHFitter(penalizer=COX_PENALIZER, l1_ratio=COX_L1_RATIO)
       cox_model.fit(train_df, duration_col="time", event_col="event", show_progress=True)

       test_df = pd.DataFrame(test_features, columns=feature_columns)
       train_risk = np.log(cox_model.predict_partial_hazard(train_df[feature_columns]).to_numpy().reshape(-1))
       test_risk = np.log(cox_model.predict_partial_hazard(test_df[feature_columns]).to_numpy().reshape(-1))
       train_cindex = concordance_index(train_times, -train_risk, train_events)
       test_cindex = concordance_index(test_times, -test_risk, test_events)

       print(f"{method_name}: train C-index={train_cindex:.3f}, test C-index={test_cindex:.3f}")
       return {
           "method": method_name,
           "train_c_index": train_cindex,
           "test_c_index": test_cindex,
           "test_risk": test_risk,
       }

The direct omics linear Cox baseline is fitted on the concatenated omics
features plus view-presence flags.  The current tutorial filters to
complete-view patients before modelling, but the helper still zero-fills any
``NaN`` values defensively before standardization.

.. code-block:: python

   def make_direct_omics_features(view_arrays, view_present):
       feature_blocks = []
       for view_array in view_arrays:
           view_array = np.asarray(view_array, dtype="float32")
           feature_blocks.append(np.nan_to_num(view_array, nan=0.0))
       feature_blocks.append(np.asarray(view_present, dtype="float32"))
       return np.concatenate(feature_blocks, axis=1).astype("float32", copy=False)


   def standardize_direct_omics_features(train_features, test_features):
       mean = train_features.mean(axis=0)
       std = train_features.std(axis=0)
       std[(~np.isfinite(std)) | (std < 1e-6)] = 1.0
       train_features = ((train_features - mean) / std).astype("float32")
       test_features = ((test_features - mean) / std).astype("float32")
       return train_features, test_features

Pairwise method comparisons use a paired permutation test on held-out risk
scores.  Under the null hypothesis, the two methods' predictions are
exchangeable within each test patient, so each permutation randomly swaps the
two risk scores patient by patient and recomputes the C-index difference.

.. code-block:: python

   def calculate_bh_fdr_p_values(raw_p_values):
       raw_p_values = np.asarray(raw_p_values, dtype="float64")
       adjusted_p_values = np.empty_like(raw_p_values)
       n_tests = len(raw_p_values)

       if n_tests == 0:
           return adjusted_p_values

       sorted_indices = np.argsort(raw_p_values)
       sorted_p_values = raw_p_values[sorted_indices]
       sorted_adjusted = np.empty_like(sorted_p_values)

       running_minimum = 1.0
       for reverse_index in range(n_tests - 1, -1, -1):
           rank = reverse_index + 1
           adjusted_value = sorted_p_values[reverse_index] * n_tests / rank
           running_minimum = min(running_minimum, adjusted_value)
           sorted_adjusted[reverse_index] = min(running_minimum, 1.0)

       adjusted_p_values[sorted_indices] = sorted_adjusted
       return adjusted_p_values


   def permutation_test_c_index_difference(risk_a, risk_b, rng):
       risk_a = np.asarray(risk_a)
       risk_b = np.asarray(risk_b)
       if len(risk_a) != len(risk_b):
           raise ValueError("Both methods must have risk scores for the same number of test patients.")
       if len(risk_a) != len(test_times):
           raise ValueError("Risk scores must match the number of test patients.")

       c_index_a = concordance_index(test_times, -risk_a, test_events)
       c_index_b = concordance_index(test_times, -risk_b, test_events)
       observed_delta = c_index_a - c_index_b

       extreme_count = 0
       for _ in range(PERMUTATION_SAMPLES):
           swap_mask = rng.random(len(risk_a)) < 0.5
           permuted_risk_a = np.where(swap_mask, risk_b, risk_a)
           permuted_risk_b = np.where(swap_mask, risk_a, risk_b)

           permuted_c_index_a = concordance_index(test_times, -permuted_risk_a, test_events)
           permuted_c_index_b = concordance_index(test_times, -permuted_risk_b, test_events)
           permuted_delta = permuted_c_index_a - permuted_c_index_b

           if abs(permuted_delta) >= abs(observed_delta):
               extreme_count += 1

       p_value = (extreme_count + 1) / (PERMUTATION_SAMPLES + 1)
       return c_index_a, c_index_b, observed_delta, p_value


   def build_pairwise_significance_table(sorted_results, rng):
       rows = []

       for first_index in range(len(sorted_results)):
           for second_index in range(first_index + 1, len(sorted_results)):
               result_a = sorted_results[first_index]
               result_b = sorted_results[second_index]
               c_index_a, c_index_b, delta_c_index, p_value = permutation_test_c_index_difference(
                   result_a["test_risk"],
                   result_b["test_risk"],
                   rng,
               )

               rows.append({
                   "method_a": result_a["method"],
                   "method_b": result_b["method"],
                   "c_index_a": c_index_a,
                   "c_index_b": c_index_b,
                   "delta_c_index": delta_c_index,
                   "permutation_p_value": p_value,
               })

       significance_table = pd.DataFrame(rows)
       if len(significance_table) > 0:
           significance_table["bh_fdr_p_value"] = calculate_bh_fdr_p_values(
               significance_table["permutation_p_value"]
           )
           significance_table["significant_at_0_05"] = (
               significance_table["bh_fdr_p_value"] < SIGNIFICANCE_LEVEL
           )

       return significance_table

5. Load the cached TCGA data
----------------------------

The script downloads the tutorial archive if it is not already present,
extracts it, and loads preprocessed omics matrices plus clinical metadata.  The
exact cache-loading code is kept in the script; the important result is a set
of aligned training and test arrays.

.. code-block:: python

   print("\nLoading preprocessed TCGA survival data")

   if not DATA_DIR.exists():
       DATA_DIR.mkdir(parents=True, exist_ok=True)
   if not DATA_ZIP.exists():
       urllib.request.urlretrieve(DATA_URL, DATA_ZIP)
   with zipfile.ZipFile(DATA_ZIP, "r") as archive:
       archive.extractall(PACKAGE_DATA_DIR)

   # The script then loads cached omics arrays and clinical labels from DATA_DIR.
   # It creates X_train, X_test, train_times, test_times, train_events,
   # test_events, and view-presence masks used below.

The view-presence masks are important because not every patient has every
omics view.  For a direct comparison between methods, the tutorial filters both
the training split and the test split to complete-view patients before any
model is trained.  DLVPM therefore uses the same complete-case cohort as the
other representation-learning methods in this benchmark.

6. Train DLVPM and fit penalised Cox
------------------------------------

The survival path matrix connects every available omics view to every other
available view.  DLVPM learns view-specific DLVs, then the tutorial averages
the available view representations for each patient before fitting the Cox
model.  Because the data has already been filtered to complete-view patients,
the model sets ``missing_strategy="project"`` and does not use latent
missing-view imputation in this benchmark.

.. code-block:: python

   Path = np.array(
       [
           [0, 1, 1, 1, 1],
           [1, 0, 1, 1, 1],
           [1, 1, 0, 1, 1],
           [1, 1, 1, 0, 1],
           [1, 1, 1, 1, 0],
       ],
       dtype="float32",
   )

   dlvpm_encoders = [
       residual_encoder(view.shape[1], view_key)
       for view_key, view in zip(available_views, X_train)
   ]
   regularizer_list = [
       dlvpm_regularizers.l1_l2(l1=0.0, l2=0.0)
       for _ in available_views
   ]

   dlvpm_model = StructuralModel(
       Path=Path,
       model_list=dlvpm_encoders,
       regularizer_list=regularizer_list,
       tot_num=len(train_split),
       ndims=NDIMS,
       momentum=0.95,
       epsilon=0.001,
       orthogonalization="zca",
       train_DLV=True,
       order=True,
       missing_strategy="project",
   )

   optimizer_list = make_optimizer_list(dlvpm_model)
   dlvpm_model.compile(optimizer=optimizer_list)
   dlvpm_model.fit(X_train, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False, verbose=True)

.. code-block:: python

   print("DLVPM train metrics:", dlvpm_model.evaluate(X_train, batch_size=BATCH_SIZE, verbose=False))
   print("DLVPM test metrics:", dlvpm_model.evaluate(X_test, batch_size=BATCH_SIZE, verbose=False))

   train_dlvs = dlvpm_model.predict(X_train, batch_size=BATCH_SIZE, verbose=False)
   test_dlvs = dlvpm_model.predict(X_test, batch_size=BATCH_SIZE, verbose=False)
   train_patient_dlvs = (
       train_dlvs * train_view_present[:, np.newaxis, :]
   ).sum(axis=2) / train_counts[:, np.newaxis]
   test_patient_dlvs = (
       test_dlvs * test_view_present[:, np.newaxis, :]
   ).sum(axis=2) / test_counts[:, np.newaxis]

   results = [
       fit_penalised_cox("DLVPM + penalised Cox", train_patient_dlvs, test_patient_dlvs)
   ]

Integrated gradients feature attribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the DLVPM survival model has been trained, the script can calculate
integrated gradients for selected DLV factors with
:func:`deep_lvpm.integrated_gradients.calculate_integrated_gradients`.  This
is done before ``dlvpm_model`` is deleted so that attribution is calculated on
the fitted ``StructuralModel``.

The TCGA survival workflow uses modality-specific baselines:

- CNV and SNV use zero vectors.
- RNA-seq, miRNA-seq, and methylation use feature-wise mean vectors across the
  analysed subjects.

The script explains DLV1 and DLV2 by default.  For each modality and DLV, it
averages absolute integrated gradients across subjects and displays top-10
locus bar plots with ``matplotlib``.  The plots are shown on screen and are not
saved as PNG files.  Absolute values are used for ranking because a DLV factor
can be pushed up by increased values and down by decreased values; signed
values can cancel out across subjects even when a locus is important.  The
output tables also keep the signed mean attribution so directional bias can
still be inspected.

The top-loci tables are saved in
``deep_lvpm/data/dlvpm_tcga_survival_demo``:

- ``dlv1_integrated_gradients_top_loci.tsv``
- ``dlv2_integrated_gradients_top_loci.tsv``

See :doc:`/reference/integrated_gradients` for the standalone API, baseline
guidance, and aggregation examples.

7. Train the direct neural Cox model
------------------------------------

The direct model uses the same per-view encoder idea, but instead of learning a
DLVPM representation first, it concatenates all view embeddings plus view
presence flags and trains a neural risk head directly with the Cox partial
likelihood.

.. code-block:: python

   class DirectMultimodalDeepCox(nn.Module):
       def __init__(self, available_views, X_train):
           super().__init__()
           self.available_views = list(available_views)
           self.n_views = len(available_views)
           self.encoders = nn.ModuleList([
               residual_encoder(train_view.shape[1], f"direct_{view_key}")
               for view_key, train_view in zip(available_views, X_train)
           ])
           merged_dim = RESIDUAL_ENCODER_LATENT_DIM * self.n_views + self.n_views
           self.risk_head = nn.Sequential(
               nn.Linear(merged_dim, 128),
               nn.ReLU(),
               nn.BatchNorm1d(128),
               nn.Dropout(NEURAL_COX_DROPOUT),
               nn.Linear(128, 64),
               nn.ReLU(),
               nn.Linear(64, 1),
           )

       def forward(self, inputs):
           feature_inputs = inputs[: self.n_views]
           flag_inputs = inputs[self.n_views :]
           direct_embeddings = []
           for encoder, feature_input, flag_input in zip(self.encoders, feature_inputs, flag_inputs):
               embedding = encoder(feature_input)
               embedding = embedding * flag_input
               direct_embeddings.append(embedding)
           merged = torch.cat(direct_embeddings + list(flag_inputs), dim=1)
           risk = self.risk_head(merged)
           return risk

8. Train the direct omics linear Cox baseline
---------------------------------------------

The direct omics linear Cox baseline is a single linear risk head trained on
the raw concatenated omics features.  It uses the same complete-view training
and test patients as DLVPM and the other methods.

.. code-block:: python

   class DirectOmicsLinearCox(nn.Module):
       def __init__(self, input_dim):
           super().__init__()
           self.risk_head = nn.Linear(input_dim, 1)

       def forward(self, inputs):
           return self.risk_head(inputs)

       def regularization_loss(self):
           penalty = torch.zeros(
               (),
               dtype=next(self.parameters()).dtype,
               device=next(self.parameters()).device,
           )
           if LINEAR_COX_L2:
               penalty = penalty + LINEAR_COX_L2 * torch.sum(self.risk_head.weight ** 2)
           return penalty


   print("\nDirect omics linear Cox")
   linear_train_features = make_direct_omics_features(X_train, train_view_present)
   linear_test_features = make_direct_omics_features(X_test, test_view_present)
   linear_train_features, linear_test_features = standardize_direct_omics_features(
       linear_train_features,
       linear_test_features,
   )
   linear_cox_model = DirectOmicsLinearCox(linear_train_features.shape[1])
   fit_direct_linear_cox_model(
       linear_cox_model,
       linear_train_features,
       train_y,
       batch_size=BATCH_SIZE,
       epochs=EPOCHS,
   )
   train_risk = predict_direct_linear_cox_model(
       linear_cox_model,
       linear_train_features,
       batch_size=BATCH_SIZE,
   ).reshape(-1)
   test_risk = predict_direct_linear_cox_model(
       linear_cox_model,
       linear_test_features,
       batch_size=BATCH_SIZE,
   ).reshape(-1)
   results.append({
       "method": "Direct omics linear Cox",
       "train_c_index": concordance_index(train_times, -train_risk, train_events),
       "test_c_index": concordance_index(test_times, -test_risk, test_events),
       "test_risk": test_risk,
   })
   clear_torch_memory()

9. Train representation-learning baselines
------------------------------------------

CLIP, VICReg, LeJEPA, and DGCCA use the same complete-view training patients as
DLVPM, so all required views are present during representation learning.

.. code-block:: python

   representation_train_data = X_train

   for method_name in MULTIMODAL_METHODS:
       model_name = method_name.lower()
       encoders = [
           residual_encoder(view.shape[1], f"{model_name}_{view_key}")
           for view_key, view in zip(available_views, X_train)
       ]
       regularizer_list = [
           dlvpm_regularizers.l1_l2(l1=0.0, l2=0.0)
           for _ in available_views
       ]

       if method_name == "CLIP":
           representation_model = CLIP(encoders, regularizer_list, NDIMS)
       elif method_name == "VICReg":
           representation_model = VICReg(encoders, regularizer_list, NDIMS)
       elif method_name == "LeJEPA":
           representation_model = LeJEPA(encoders, regularizer_list, NDIMS, num_slices=64)
       elif method_name == "DGCCA":
           representation_model = DGCCA(encoders, regularizer_list, NDIMS)
       else:
           raise ValueError(f"Unknown multimodal method: {method_name}")

       optimizer_list = make_optimizer_list(representation_model)
       representation_model.compile(optimizer=optimizer_list)
       representation_model.fit(
           representation_train_data,
           batch_size=BATCH_SIZE,
           epochs=EPOCHS,
           shuffle=True,
           verbose=True,
       )

10. Compare survival prediction results
---------------------------------------

The final tables sort methods by held-out test concordance index.  The
permutation table reports all pairwise C-index comparisons with raw two-sided
p-values and Benjamini-Hochberg FDR-adjusted p-values.  The plot still
bootstraps test patients to show mean test C-index and a 90 percent confidence
interval for each method, with the strongest test performer furthest to the
left.

.. code-block:: python

   sorted_results = sorted(results, key=lambda result: result["test_c_index"], reverse=True)
   results_table = pd.DataFrame(sorted_results).drop(columns=["test_risk"]).reset_index(drop=True)
   print("\nSurvival prediction results")
   print(
       results_table.to_string(
           index=False,
           formatters={
               "train_c_index": "{:.3f}".format,
               "test_c_index": "{:.3f}".format,
           },
       )
   )

   permutation_rng = np.random.default_rng(RANDOM_SEED + 1)
   significance_table = build_pairwise_significance_table(sorted_results, permutation_rng)
   print("\nPairwise permutation tests for test C-index differences")
   if len(significance_table) == 0:
       print("Not enough methods to compare.")
   else:
       print(significance_table.to_string(
           index=False,
           formatters={
               "c_index_a": "{:.3f}".format,
               "c_index_b": "{:.3f}".format,
               "delta_c_index": "{:+.3f}".format,
               "permutation_p_value": "{:.4f}".format,
               "bh_fdr_p_value": "{:.4f}".format,
           },
       ))

   rng = np.random.default_rng(RANDOM_SEED)
   plot_rows = []
   for result in sorted_results:
       boot = []
       for _ in range(BOOTSTRAP_SAMPLES):
           idx = rng.integers(0, len(test_times), len(test_times))
           boot.append(concordance_index(test_times[idx], -result["test_risk"][idx], test_events[idx]))
       plot_rows.append((result["method"], np.mean(boot), *np.percentile(boot, [5, 95])))

   plot_df = pd.DataFrame(plot_rows, columns=["method", "mean_c_index", "ci_low", "ci_high"])
   plt.errorbar(
       plot_df["method"],
       plot_df["mean_c_index"],
       yerr=[
           plot_df["mean_c_index"] - plot_df["ci_low"],
           plot_df["ci_high"] - plot_df["mean_c_index"],
       ],
       fmt="o",
       capsize=4,
   )
   plt.ylabel("Test C-index")
   plt.xticks(rotation=45, ha="right")
   plt.tight_layout()
   plt.show()

Summary
-------

The survival tutorial keeps the modelling stages explicit: load aligned
multi-omic views and survival labels, train DLVPM or baseline representation
models, reduce view-level embeddings to patient-level features, and compare
the representations with a consistent penalised Cox evaluation.
