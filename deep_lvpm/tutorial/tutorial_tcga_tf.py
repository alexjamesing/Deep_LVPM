
####### Tutorial 2 #########

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import layers, regularizers, optimizers
from importlib import resources

# import keras_tuner as kt  # temporarily disabled

import deep_lvpm 
from deep_lvpm.model import StructuralModel
# from deep_lvpm.tuner import Tuner  # temporarily disabled

tf.config.run_functions_eagerly(False)   # default graph mode for final training

with resources.as_file(resources.files("deep_lvpm.data") /
                       "Lung_multiomics_sample_train.npz") as f:
    arrays = np.load(f)
    rnaseq      = arrays["rnaseq"]
    snv         = arrays["snv"]
    methylation = arrays["methylation"]
    mirna       = arrays["mirna"]
    histo20     = arrays["histo20"]

X_full = [histo20, rnaseq, methylation, mirna, snv]   # preserve this order!

# Split into 80/20 train/validation for tuning
n_samples = rnaseq.shape[0]
rng = np.random.default_rng(42)
indices = rng.permutation(n_samples)
split_idx = int(0.8 * n_samples)
train_idx, val_idx = indices[:split_idx], indices[split_idx:]

def split_view(view):
    return view[train_idx], view[val_idx]

X_train, X_val = [], []
for view in X_full:
    tr_view, val_view = split_view(view)
    X_train.append(tr_view)
    X_val.append(val_view)



def build_view_model(input_dim: int, base_name: str) -> keras.Model:
    """Simple fixed architecture for a single view (temporarily replacing HyperModel)."""
    width = max(256, min(input_dim, 512))
    l1 = 1e-4
    l2 = 1e-4
    dropout = 0.5

    inputs = keras.Input(shape=(input_dim,), name=f"{base_name}_in")
    x = layers.Dense(
        width,
        activation="relu",
        kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
        name=f"{base_name}_dense1",
    )(inputs)
    x = layers.BatchNormalization(name=f"{base_name}_bn")(x)
    skip = layers.Dense(width, activation="relu", name=f"{base_name}_skip")(inputs)
    x = layers.Add(name=f"{base_name}_add")([x, skip])
    x = layers.Dropout(dropout, name=f"{base_name}_drop")(x)
    outputs = layers.Dense(width, activation="linear", name=f"{base_name}_out")(x)
    return keras.Model(inputs, outputs, name=f"{base_name}_model")


# Build fixed models per view (temporarily replacing HyperModel/Tuner)
view_models = [
    build_view_model(histo20.shape[1], "histo20"),
    build_view_model(rnaseq.shape[1], "rnaseq"),
    build_view_model(methylation.shape[1], "meth"),
    build_view_model(mirna.shape[1], "mirna"),
    build_view_model(snv.shape[1], "snv"),
]


ndims = 5 # number of latent factors

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

tot_num = rnaseq.shape[0]

regularizer_list = [
    regularizers.L1L2(l1=0.001, l2=0.001)
    for _ in range(len(view_models))
]

# Build DLVPM directly and train with ZCA orthogonalization.
opt_list = [keras.optimizers.Adam(learning_rate=lr_schedule) for _ in range(len(view_models))]

best_model = StructuralModel(
    Path=Path,
    model_list=view_models,
    regularizer_list=regularizer_list,
    tot_num=tot_num,
    ndims=ndims,
    momentum=0.95,
    epsilon=0.001,
    orthogonalization="zca",
    train_DLV=True,
    order=True,
)

best_model.compile(optimizer=opt_list)


# --- Rotation verification: internal vs post-hoc ---
def check_internal_vs_posthoc_rotation(model, X, batch_size=None, label="train"):
    import numpy as np

    Y = model.predict(X, batch_size=batch_size, verbose=0)  # shape: (N, D, K)

    # Build consensus and its centered covariance
    A = np.sum(Y, axis=-1)  # (N, D)
    A = A - A.mean(axis=0, keepdims=True)
    Cov = A.T @ A  # (D, D)

    # Off-diagonal energy ratio before rotation
    Cov_diag = np.diag(np.diag(Cov))
    off_before = np.linalg.norm(Cov - Cov_diag, ord='fro')
    tot_before = np.linalg.norm(Cov, ord='fro') + 1e-12
    off_ratio_before = off_before / tot_before

    # Post-hoc rotation via SVD (numerically stable)
    # A = U S V^T -> use V as rotation in D-space
    try:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        V = Vt.T
    except Exception:
        # Fallback to eigendecomposition
        evals, V = np.linalg.eigh(Cov + 1e-12 * np.eye(Cov.shape[0]))
        # Descending
        V = V[:, ::-1]

    # Check how close V is to identity up to column sign/permutation:
    # For each column, record the max absolute component.
    col_peaks = np.max(np.abs(V), axis=0)
    mean_peak = float(np.mean(col_peaks))
    min_peak = float(np.min(col_peaks))

    # Rotate covariance and re-check off-diagonality
    Cov_rot = V.T @ Cov @ V
    Cov_rot_diag = np.diag(np.diag(Cov_rot))
    off_after = np.linalg.norm(Cov_rot - Cov_rot_diag, ord='fro')
    tot_after = np.linalg.norm(Cov_rot, ord='fro') + 1e-12
    off_ratio_after = off_after / tot_after

    print(f"\n[Rotation check - {label}]")
    print(f"Consensus covariance off-diagonal ratio BEFORE: {off_ratio_before:.6f}")
    print(f"Consensus covariance off-diagonal ratio AFTER:  {off_ratio_after:.6f}")
    print(f"V column-wise max |.| — mean: {mean_peak:.6f}, min: {min_peak:.6f}")

    return {
        "off_ratio_before": off_ratio_before,
        "off_ratio_after": off_ratio_after,
        "mean_col_peak_abs": mean_peak,
        "min_col_peak_abs": min_peak,
    }


best_model.fit(X_full, batch_size=batch_size, epochs=epochs, verbose=True)

train_metrics = best_model.evaluate(X_train, verbose=False, return_dict=True)
val_metrics = best_model.evaluate(X_val, verbose=False, return_dict=True)

print('Training total loss=' + str(train_metrics["total_loss"]))
print('Validation total loss=' + str(val_metrics["total_loss"]))
print('Training mean correlation r=' + str(train_metrics["cross_metric"]))
print('Validation mean correlation r=' + str(val_metrics["cross_metric"]))

with resources.as_file(resources.files("deep_lvpm.data") /
                       "Lung_multiomics_sample_test.npz") as f:
    arrays = np.load(f)
    rnaseq_test      = arrays["rnaseq"]
    snv_test         = arrays["snv"]
    methylation_test = arrays["methylation"]
    mirna_test       = arrays["mirna"]
    histo20_test     = arrays["histo20"]

X_arr_test = [histo20_test, rnaseq_test, methylation_test, mirna_test, snv_test]   # Here, is the full test dataset list


train_DLVs = best_model.predict(X_train)

corr_mat = best_model.calculate_corrmat(train_DLVs) # This outputs correlation matrices between different data types included in the model


test_metrics = best_model.evaluate(X_arr_test, return_dict=True)

print('Test total loss=' + str(test_metrics["total_loss"]))

test_DLVs = best_model.predict(X_arr_test) ## Here, we obtain the full set of test_DLVs
print('The mean correlation across data types is r=' + str(test_metrics["cross_metric"]))

## Associations between the first set of DLVs are:
print(np.corrcoef(test_DLVs[:,0,:].T))
## Associations between the second set of DLVs are:
print(np.corrcoef(test_DLVs[:,1,:].T))

corr_mat = best_model.calculate_corrmat(test_DLVs) # This outputs correlation matrices between different data types included in the model

corrmean = [np.mean(a) for a in corr_mat]
print(corrmean)

from deep_lvpm.plot import plot_correlation_chord_row

# We can then plot the results in a chord diagram

data_names = ["Histology", "RNASeq", "miRNASeq", "Methylation", "SNVs"]

fig, ax = plot_correlation_chord_row(
    corr_mat,
    data_names,
    min_corr=0,
    node_cmap_name="Pastel1",
    figure_title = "Correlation Plots Between Omics and Imaging Data Types in Lung Cancer",
    show_edge_labels=True,
    dpi=300,
    show=True  # don't pop up a window
    )
