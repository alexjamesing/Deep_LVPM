#!/usr/bin/env python3
# Configuration

import os
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path

RANDOM_SEED = 42
SURVIVAL_ENDPOINT = "pfi"

PACKAGE_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR = PACKAGE_DATA_DIR / "dlvpm_tcga_survival_demo"
CACHE_DIR = DATA_DIR / "preprocessed_omics_cache"
DATA_URL = "https://zenodo.org/records/20305527/files/dlvpm_tcga_survival_demo.zip?download=1"
DATA_ZIP = PACKAGE_DATA_DIR / "dlvpm_tcga_survival_demo.zip"

NDIMS = 100
BATCH_SIZE = 2048
EPOCHS = 50
LEARNING_RATE = 1e-4

MULTIMODAL_METHODS = ["CLIP", "VICReg", "LeJEPA", "DGCCA"]

GENE_MIXER_LATENT_DIM = 512
GENE_MIXER_RANK = 512
GENE_MIXER_DEPTH = 10
GENE_MIXER_DROPOUT = 0.30

NEURAL_COX_DROPOUT = 0.60
NEURAL_COX_L2 = 1e-2
COX_PENALIZER = 0.10
COX_L1_RATIO = 0.00

USE_CPU_ONLY = os.environ.get("DLVPM_SURVIVAL_USE_CPU", "1") == "1"

if os.environ.get("DLVPM_SURVIVAL_SMOKE_TEST") == "1":
    EPOCHS = 1
    MULTIMODAL_METHODS = ["CLIP"]

import gc
import json
import random
import urllib.request
import zipfile

import keras
import numpy as np
import pandas as pd
import torch
from keras import layers, ops, regularizers
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from deep_lvpm.model import StructuralModel
from deep_lvpm.multi_model import CLIP, DGCCA, LeJEPA, VICReg

if USE_CPU_ONLY:
    import keras.src.backend.torch.core as keras_torch_core
    keras_torch_core.DEFAULT_DEVICE = "cpu"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def gene_mixer_encoder(input_dim, name):
    inputs = keras.Input(shape=(input_dim,), name=f"{name}_input")
    present = layers.Lambda(
        lambda z: ops.cast(ops.any(ops.logical_not(ops.isnan(z)), axis=1, keepdims=True), ops.dtype(z)),
        output_shape=(1,), name=f"{name}_present",
    )(inputs)
    x = layers.Lambda(lambda z: ops.where(ops.isnan(z), ops.zeros_like(z), z),
                      output_shape=(input_dim,), name=f"{name}_nan_to_zero")(inputs)
    x = layers.LayerNormalization(name=f"{name}_input_norm")(x)

    for block_index in range(GENE_MIXER_DEPTH):
        block_name = f"{name}_mixer_{block_index + 1}"
        h = layers.LayerNormalization(name=f"{block_name}_norm")(x)

        context = layers.Dense(GENE_MIXER_RANK, use_bias=False, name=f"{block_name}_context_down")(h)
        context = layers.Activation("gelu", name=f"{block_name}_context_gelu")(context)
        context = layers.Dense(input_dim, use_bias=False, name=f"{block_name}_context_up")(context)

        gate = layers.Dense(GENE_MIXER_RANK, use_bias=False, name=f"{block_name}_gate_down")(h)
        gate = layers.Activation("gelu", name=f"{block_name}_gate_gelu")(gate)
        gate = layers.Dense(input_dim, activation="sigmoid", name=f"{block_name}_gate_up")(gate)

        mixed = layers.Multiply(name=f"{block_name}_gated_context")([context, gate])
        mixed = layers.Dropout(GENE_MIXER_DROPOUT, name=f"{block_name}_dropout")(mixed)
        x = layers.Add(name=f"{block_name}_residual")([x, mixed])

    x = layers.LayerNormalization(name=f"{name}_head_norm")(x)
    x = layers.Dense(GENE_MIXER_LATENT_DIM, name=f"{name}_latent")(x)
    x = layers.Activation("gelu", name=f"{name}_latent_gelu")(x)
    x = layers.Dropout(GENE_MIXER_DROPOUT, name=f"{name}_latent_dropout")(x)
    outputs = layers.Multiply(name=f"{name}_missing_view_mask")([x, present])
    return keras.Model(inputs=inputs, outputs=outputs, name=f"{name}_encoder")


def cox_partial_likelihood_loss(y_true, y_pred):
    times = y_true[:, 0]
    events = y_true[:, 1]
    risks = ops.reshape(y_pred, (-1,))

    order = ops.argsort(-times)
    events = ops.take(events, order, axis=0)
    risks = ops.take(risks, order, axis=0)

    log_cumulative_hazard = ops.log(ops.cumsum(ops.exp(risks)) + 1e-8)
    log_likelihood = (risks - log_cumulative_hazard) * events
    return -ops.sum(log_likelihood) / (ops.sum(events) + 1e-8)


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
    return {"method": method_name, "train_c_index": train_cindex, "test_c_index": test_cindex}


if not (CACHE_DIR / "cache_config.json").exists():
    PACKAGE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading TCGA survival archive to {DATA_ZIP}")
    urllib.request.urlretrieve(DATA_URL, DATA_ZIP)
    print(f"Extracting TCGA survival archive from {DATA_ZIP}")
    with zipfile.ZipFile(DATA_ZIP, "r") as archive:
        archive.extractall(PACKAGE_DATA_DIR)
    DATA_ZIP.unlink()

with open(CACHE_DIR / "cache_config.json", "r", encoding="utf-8") as f:
    cache_config = json.load(f)

available_views = list(cache_config["available_views"])
split_df = pd.read_csv(CACHE_DIR / "patient_split.tsv", sep="\t")
train_split = split_df[split_df["split"] == "train"].reset_index(drop=True)
test_split = split_df[split_df["split"] == "test"].reset_index(drop=True)

X_train = []
X_test = []
train_view_present = []
test_view_present = []

print("\nLoading preprocessed TCGA survival data")
for view_key in available_views:
    train_view = np.load(CACHE_DIR / f"{view_key}_train_dlvpm.npy").astype("float32")
    test_view = np.load(CACHE_DIR / f"{view_key}_test_dlvpm.npy").astype("float32")
    train_flag = np.load(CACHE_DIR / f"{view_key}_train_flag.npy").reshape(-1).astype("float32")
    test_flag = np.load(CACHE_DIR / f"{view_key}_test_flag.npy").reshape(-1).astype("float32")

    X_train.append(train_view)
    X_test.append(test_view)
    train_view_present.append(train_flag)
    test_view_present.append(test_flag)
    print(f"{view_key}: train {train_view.shape}, test {test_view.shape}")

train_view_present = np.column_stack(train_view_present).astype("float32")
test_view_present = np.column_stack(test_view_present).astype("float32")

time_col = f"{SURVIVAL_ENDPOINT}_time_days"
event_col = f"{SURVIVAL_ENDPOINT}_event"
train_times = train_split[time_col].to_numpy(dtype="float32")
train_events = train_split[event_col].to_numpy(dtype="float32")
test_times = test_split[time_col].to_numpy(dtype="float32")
test_events = test_split[event_col].to_numpy(dtype="float32")
train_y = np.column_stack([train_times, train_events]).astype("float32")

n_views = len(available_views)
train_counts = train_view_present.sum(axis=1)
test_counts = test_view_present.sum(axis=1)
if train_counts.min() < 1 or test_counts.min() < 1:
    raise ValueError("Every patient must have at least one available data view.")

print(f"Training patients: {len(train_split)}")
print(f"Test patients:     {len(test_split)}")
print(f"Training events:   {int(train_events.sum())}")
print(f"Test events:       {int(test_events.sum())}")
print(f"Batch size:        {BATCH_SIZE}")
print(f"Epochs per model:  {EPOCHS}")
print(f"Learning rate:     {LEARNING_RATE}")


print("\nDLVPM + penalised Cox")

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

dlvpm_encoders = [gene_mixer_encoder(view.shape[1], view_key) for view_key, view in zip(available_views, X_train)]
regularizer_list = [regularizers.L1L2(l1=0.0, l2=0.0) for _ in available_views]
optimizer_list = [keras.optimizers.Adam(learning_rate=LEARNING_RATE) for _ in available_views]

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
    order=True
)
dlvpm_model.compile(optimizer=optimizer_list)
dlvpm_model.fit(X_train, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False, verbose=True)

print("DLVPM train metrics:", dlvpm_model.evaluate(X_train, batch_size=BATCH_SIZE, verbose=False, return_dict=True))
print("DLVPM test metrics:", dlvpm_model.evaluate(X_test, batch_size=BATCH_SIZE, verbose=False, return_dict=True))

train_dlvs = dlvpm_model.predict(X_train, batch_size=BATCH_SIZE, verbose=False)
test_dlvs = dlvpm_model.predict(X_test, batch_size=BATCH_SIZE, verbose=False)
train_patient_dlvs = (train_dlvs * train_view_present[:, np.newaxis, :]).sum(axis=2) / train_counts[:, np.newaxis]
test_patient_dlvs = (test_dlvs * test_view_present[:, np.newaxis, :]).sum(axis=2) / test_counts[:, np.newaxis]
results = [fit_penalised_cox("DLVPM + penalised Cox", train_patient_dlvs, test_patient_dlvs)]

del dlvpm_model, train_dlvs, test_dlvs
keras.backend.clear_session()
gc.collect()


print("\nDirect multimodal neural Cox")

direct_feature_inputs = []
direct_flag_inputs = []
direct_embeddings = []
for view_key, train_view in zip(available_views, X_train):
    feature_input = keras.Input(shape=(train_view.shape[1],), name=f"direct_{view_key}_features")
    flag_input = keras.Input(shape=(1,), name=f"direct_{view_key}_present")
    embedding = gene_mixer_encoder(train_view.shape[1], f"direct_{view_key}")(feature_input)
    embedding = layers.Multiply(name=f"direct_{view_key}_masked_embedding")([embedding, flag_input])
    direct_feature_inputs.append(feature_input)
    direct_flag_inputs.append(flag_input)
    direct_embeddings.append(embedding)

merged = layers.Concatenate(name="direct_merged_embeddings")(direct_embeddings + direct_flag_inputs)
direct_regularizer = regularizers.L1L2(l1=0.0, l2=NEURAL_COX_L2)
x = layers.Dense(128, activation="relu", kernel_regularizer=direct_regularizer, name="direct_dense1")(merged)
x = layers.BatchNormalization(name="direct_bn1")(x)
x = layers.Dropout(NEURAL_COX_DROPOUT, name="direct_dropout1")(x)
x = layers.Dense(64, activation="relu", kernel_regularizer=direct_regularizer, name="direct_dense2")(x)
risk = layers.Dense(1, activation="linear", kernel_regularizer=direct_regularizer, name="direct_risk_score")(x)
direct_model = keras.Model(direct_feature_inputs + direct_flag_inputs, risk, name="direct_multimodal_deep_cox")
direct_model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=cox_partial_likelihood_loss)

train_flags = [train_view_present[:, i : i + 1].astype("float32") for i in range(n_views)]
test_flags = [test_view_present[:, i : i + 1].astype("float32") for i in range(n_views)]
direct_train_inputs = X_train + train_flags
direct_test_inputs = X_test + test_flags

direct_model.fit(direct_train_inputs, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False, verbose=True)
train_risk = direct_model.predict(direct_train_inputs, batch_size=BATCH_SIZE, verbose=False).reshape(-1)
test_risk = direct_model.predict(direct_test_inputs, batch_size=BATCH_SIZE, verbose=False).reshape(-1)
results.append({
    "method": "Direct multimodal neural Cox",
    "train_c_index": concordance_index(train_times, -train_risk, train_events),
    "test_c_index": concordance_index(test_times, -test_risk, test_events),
})
print(
    "Direct multimodal neural Cox: "
    f"train C-index={results[-1]['train_c_index']:.3f}, "
    f"test C-index={results[-1]['test_c_index']:.3f}"
)

del direct_model
keras.backend.clear_session()
gc.collect()


complete_train_mask = train_view_present.all(axis=1)
representation_train_data = [train_view[complete_train_mask] for train_view in X_train]
print(f"\nTraining multimodal representation models on {int(complete_train_mask.sum())} complete training patients.")

for method_name in MULTIMODAL_METHODS:
    print(f"\n{method_name} + penalised Cox")

    model_name = method_name.lower()
    encoders = [gene_mixer_encoder(view.shape[1], f"{model_name}_{view_key}") for view_key, view in zip(available_views, X_train)]
    regularizer_list = [regularizers.L1L2(l1=0.0, l2=0.0) for _ in available_views]
    optimizer_list = [keras.optimizers.Adam(learning_rate=LEARNING_RATE) for _ in available_views]

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

    representation_model.compile(optimizer=optimizer_list)
    representation_model.fit(representation_train_data, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, verbose=True)

    train_rep_by_view = representation_model.predict(X_train, batch_size=BATCH_SIZE, verbose=False)
    test_rep_by_view = representation_model.predict(X_test, batch_size=BATCH_SIZE, verbose=False)
    train_representations = (train_rep_by_view * train_view_present[:, np.newaxis, :]).sum(axis=2) / train_counts[:, np.newaxis]
    test_representations = (test_rep_by_view * test_view_present[:, np.newaxis, :]).sum(axis=2) / test_counts[:, np.newaxis]
    results.append(fit_penalised_cox(f"{method_name} + penalised Cox", train_representations, test_representations))

    del representation_model
    keras.backend.clear_session()
    gc.collect()


results_table = pd.DataFrame(results).sort_values("test_c_index", ascending=False).reset_index(drop=True)
print("\nSurvival prediction results")
print(results_table.to_string(index=False, formatters={"train_c_index": "{:.3f}".format, "test_c_index": "{:.3f}".format}))
