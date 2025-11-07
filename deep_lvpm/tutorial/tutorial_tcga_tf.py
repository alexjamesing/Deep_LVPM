# """
# TCGA lung cancer tutorial using Deep LVPM with the TensorFlow backend.

# Run from the command line:

#     python -m deep_lvpm.tutorial.tutorial_tcga_tf

# This script integrates five modalities (histology features, RNA-seq, DNA
# methylation, miRNA, and somatic mutations) using small residual encoders and
# reports the StructuralModel evaluation metrics.
# """

# from __future__ import annotations

# import os
# from importlib import resources

# # Use the TensorFlow backend for Keras 3.
# os.environ.setdefault("KERAS_BACKEND", "tensorflow")

# import numpy as np
# import tensorflow as tf

# import keras
# from keras import layers, regularizers
# from keras.optimizers import Adam, schedules

# from deep_lvpm.models.StructuralModel import StructuralModel


# def _evaluate_structural_model(model: StructuralModel, data) -> dict[str, float]:
#     """Return eval metrics as plain floats regardless of backend response type."""

#     results = model.evaluate(data, verbose=False)
#     if isinstance(results, dict):
#         return {key: float(value) for key, value in results.items()}
#     return {f"metric_{idx}": float(val) for idx, val in enumerate(results)}


# def _load_tcga_sample() -> list[np.ndarray]:
#     """Load the packaged TCGA sample arrays shipped with the toolbox."""

#     with resources.as_file(resources.files("deep_lvpm.data") / "Lung_multiomics_sample_train.npz") as train_file:
#         arrays = np.load(train_file)
#         rnaseq = arrays["rnaseq"]
#         snv = arrays["snv"]
#         methylation = arrays["methylation"]
#         mirna = arrays["mirna"]
#         histo20 = arrays["histo20"]
#     return [histo20, rnaseq, methylation, mirna, snv]


# def _residual_block(input_dim: int, name: str) -> keras.Model:
#     """Build a shallow residual encoder for a tabular modality."""

#     inputs = keras.Input(shape=(input_dim,), name=f"{name}_in")
#     x = layers.Dense(
#         input_dim,
#         activation="linear",
#         kernel_initializer=keras.initializers.Identity(),
#         kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
#         name=f"{name}_dense1",
#     )(inputs)
#     x = layers.BatchNormalization(momentum=0.9, name=f"{name}_bn1")(x)
#     x = layers.ReLU(name=f"{name}_relu")(x)
#     x = layers.Dense(
#         input_dim,
#         activation="linear",
#         kernel_initializer=keras.initializers.Identity(),
#         kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
#         name=f"{name}_dense2",
#     )(x)
#     x = layers.Add(name=f"{name}_add")([inputs, x])
#     x = layers.Dropout(0.5, name=f"{name}_drop")(x)
#     return keras.Model(inputs=inputs, outputs=x, name=f"{name}_encoder")


# if __name__ == "__main__":
#     print("=== Deep LVPM TCGA tutorial (TensorFlow backend) ===")
#     print(f"Keras backend: {keras.backend.backend()}")
#     print(f"Physical GPUs: {tf.config.list_physical_devices('GPU')}")

#     # ------------------------------------------------------------------
#     # Step 1. Load the packaged multi-omics training views.
#     # ------------------------------------------------------------------
#     # ``_load_tcga_sample`` returns the five modalities shipped with the repo.
#     # Keeping them in a fixed order is important so the encoders, optimiser
#     # assignments, and adjacency matrix stay aligned.
#     views = _load_tcga_sample()
#     view_names = ["histo20", "rnaseq", "methylation", "mirna", "snv"]

#     # ------------------------------------------------------------------
#     # Step 2. Build the measurement encoders (one per modality).
#     # ------------------------------------------------------------------
#     # Each encoder is a shallow residual MLP that preserves the dimensionality
#     # of its input.  Residual connections help stabilise FactorLayer training
#     # on tabular data without significant feature engineering.
#     encoders = [_residual_block(view.shape[1], name) for view, name in zip(views, view_names)]

#     # ------------------------------------------------------------------
#     # Step 3. Define the structural adjacency and training schedule.
#     # ------------------------------------------------------------------
#     # The adjacency matrix expresses a hub-and-spoke structure where the second
#     # factor (F2) connects bidirectionally with the remaining factors.  A gentle
#     # exponential learning-rate decay keeps optimisation stable over 300 epochs.
#     adjacency = np.array(
#         [
#             [0, 1, 0, 0, 0],
#             [1, 0, 1, 1, 1],
#             [0, 1, 0, 0, 0],
#             [0, 1, 0, 0, 0],
#             [0, 1, 0, 0, 0],
#         ],
#         dtype="float32",
#     )

#     batch_size = 256
#     epochs = 300
#     total_steps = max(1, (views[0].shape[0] // batch_size) * epochs)
#     lr_schedule = schedules.ExponentialDecay(
#         initial_learning_rate=1e-4,
#         decay_steps=total_steps,
#         decay_rate=1e-5 / 1e-4,
#         staircase=False,
#     )

#     regulariser_list = [regularizers.L1L2(l1=1e-4, l2=1e-4) for _ in encoders]

#     # ``tot_num`` is inferred from any view (they all share the same sample set).
#     model = StructuralModel(
#         Path=adjacency,
#         model_list=encoders,
#         regularizer_list=regulariser_list,
#         tot_num=views[0].shape[0],
#         ndims=5,
#         orthogonalization="Moore-Penrose",
#         momentum=0.95,
#         epsilon=1e-3,
#         train_DLV=True,
#     )

#     # One Adam optimiser per modality, all driven by the same decay schedule.
#     optimisers = [Adam(learning_rate=lr_schedule) for _ in encoders]
#     model.compile(optimizer=optimisers)

#     # ------------------------------------------------------------------
#     # Step 4. Fit the model and monitor optimisation history.
#     # ------------------------------------------------------------------
#     history = model.fit(
#         views,
#         batch_size=batch_size,
#         epochs=epochs,
#         verbose=True,
#     )

#     # ------------------------------------------------------------------
#     # Step 5. Inspect metrics and latent representations.
#     # ------------------------------------------------------------------
#     metrics = _evaluate_structural_model(model, views)
#     print("Training metrics:", metrics)
#     print("Training history keys:", list(history.history))

#     latent = model.predict(views, verbose=False)
#     print("Latent tensor shape:", latent.shape)


####### Tutorial 2 #########

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import layers, regularizers, optimizers
from importlib import resources

import deep_lvpm as DLVPM
from deep_lvpm.models.StructuralModel import StructuralModel

tf.config.run_functions_eagerly(False)   # keep graph mode for performance

with resources.as_file(resources.files("deep_lvpm.data") /
                       "Lung_multiomics_sample_train.npz") as f:
    arrays = np.load(f)
    rnaseq      = arrays["rnaseq"]
    snv         = arrays["snv"]
    methylation = arrays["methylation"]
    mirna       = arrays["mirna"]
    histo20     = arrays["histo20"]

X_arr = [histo20, rnaseq, methylation, mirna, snv]   # preserve this order!



def residual_block(
        input_dim: int,
        kernel_reg_l1: float = 0.01,
        kernel_reg_l2: float = 0.01,
        dropout_rate: float = 0.5,
        name: str = "residual_block"
    ) -> keras.Model:
    """
    Builds a simple fully-connected residual block.

    Parameters
    ----------
    input_dim : int
        Number of features in the (flat) input vector.
    kernel_reg_l1 : float, optional
        L1 regularisation factor for dense layers (default 0.01).
    kernel_reg_l2 : float, optional
        L2 regularisation factor for dense layers (default 0.01).
    dropout_rate : float, optional
        Drop-out probability applied after the residual connection (default 0.5).
    name : str, optional
        Name for the returned `tf.keras.Model`.

    Returns
    -------
    tf.keras.Model
        A Keras `Model` representing the residual block.
    """
    # -------- input --------
    inputs = keras.Input(shape=(input_dim,), name=f"{name}_in")

    # -------- first linear projection --------
    x = layers.Dense(
        input_dim,
        activation="linear",
        kernel_initializer=keras.initializers.Identity(),
        kernel_regularizer=keras.regularizers.l1_l2(
            l1=kernel_reg_l1, l2=kernel_reg_l2
        ),
        name=f"{name}_dense1",
    )(inputs)

    # -------- normalise & non-linear activation --------
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)

    # -------- second linear projection --------
    x = layers.Dense(
        input_dim,
        activation="linear",
        kernel_initializer=keras.initializers.Identity(),
        kernel_regularizer=keras.regularizers.l1_l2(
            l1=kernel_reg_l1, l2=kernel_reg_l2
        ),
        name=f"{name}_dense2",
    )(x)

    # -------- residual connection --------
    x = layers.Add(name=f"{name}_add")([inputs, x])

    # -------- optional regularisation --------
    x = layers.Dropout(dropout_rate, name=f"{name}_drop")(x)

    # -------- wrap into a model --------
    return keras.Model(inputs=inputs, outputs=x, name=name)


model_list = [
    residual_block(histo20.shape[1], name="histo20_enc"),
    residual_block(rnaseq.shape[1],  name="rnaseq_enc"),
    residual_block(methylation.shape[1], name="meth_enc"),
    residual_block(mirna.shape[1],   name="mirna_enc"),
    residual_block(snv.shape[1],     name="snv_enc"),
]


ndims = 5        # number of latent factors

Path = np.array([
    # F₁ F₂ F₃ F₄ F₅
    [0, 1, 0, 0, 0],  # F₁ ← F₂
    [1, 0, 1, 1, 1],  # F₂ ← F₁,F₃,F₄,F₅
    [0, 1, 0, 0, 0],  # F₃ ← F₂
    [0, 1, 0, 0, 0],  # F₄ ← F₂
    [0, 1, 0, 0, 0],  # F₅ ← F₂
], dtype="float32")

batch_size  = 256
epochs      = 300
total_steps = int(rnaseq.shape[0] / batch_size) * epochs

init_lr, final_lr = 1e-4, 1e-5

lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=init_lr,
    decay_steps=total_steps,
    decay_rate=final_lr / init_lr,
    staircase=False
)

tot_num = rnaseq.shape[0] ## This is the total number of samples under analysis and is needed by DLVPM


from keras import regularizers

regularizer_list = [regularizers.L1L2(l1=0.001, l2=0.001),regularizers.L1L2(l1=0.001, l2=0.001),regularizers.L1L2(l1=0.001, l2=0.001),regularizers.L1L2(l1=0.001, l2=0.001),regularizers.L1L2(l1=0.001, l2=0.001)] ## These regularizers are applied to the final "projection" layer of the DLVPM model, used internally

DLVPM_Structural_instance = StructuralModel(Path, model_list, regularizer_list, tot_num, ndims, momentum=0.95,epsilon=0.001, orthogonalization='Moore-Penrose', train_DLV =True)

opt_list = [keras.optimizers.Adam(learning_rate=lr_schedule),keras.optimizers.Adam(learning_rate=lr_schedule),keras.optimizers.Adam(learning_rate=lr_schedule),keras.optimizers.Adam(learning_rate=lr_schedule),keras.optimizers.Adam(learning_rate=lr_schedule)]
DLVPM_Structural_instance.compile(optimizer=opt_list)


DLVPM_Structural_instance.fit(X_arr, batch_size=batch_size, epochs=epochs,verbose=True)
mean_corr = DLVPM_Structural_instance.evaluate(X_arr)

print('The mean correlation between data-types connected by the path model is r=' + str(mean_corr[1]))


with resources.as_file(resources.files("deep_lvpm.data") /
                       "Lung_multiomics_sample_test.npz") as f:
    arrays = np.load(f)
    rnaseq_test      = arrays["rnaseq"]
    snv_test         = arrays["snv"]
    methylation_test = arrays["methylation"]
    mirna_test       = arrays["mirna"]
    histo20_test     = arrays["histo20"]

X_arr_test = [histo20_test, rnaseq_test, methylation_test, mirna_test, snv_test]   # Here, is the full test dataset list
mean_corr_test = DLVPM_Structural_instance.evaluate(X_arr_test)

print('The mean correlation between data-types connected by the path model is r=' + str(mean_corr_test[1]))

test_DLVs = DLVPM_Structural_instance.predict(X_arr_test) ## Here, we obtain the full set of test_DLVs

## Associations between the first set of DLVs are:
print(np.corrcoef(test_DLVs[:,0,:].T))
## Associations between the second set of DLVs are:
print(np.corrcoef(test_DLVs[:,1,:].T))
