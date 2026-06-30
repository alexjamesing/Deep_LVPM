#!/usr/bin/env python3
# Configuration

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
    return {"method": method_name, "train_c_index": train_cindex, "test_c_index": test_cindex, "test_risk": test_risk}


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


def make_optimizer_list(model):
    return [torch.optim.Adam(view_model.parameters(), lr=LEARNING_RATE) for view_model in model.model_list]


def clear_torch_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


ZERO_BASELINE_VIEWS = {"cnv_gene", "mutation_gene"}
MEAN_BASELINE_VIEWS = {"rnaseq_gene", "mirna_gene", "methylation_gene"}
VIEW_DISPLAY_NAMES = {
    "rnaseq_gene": "RNA-seq",
    "mirna_gene": "miRNA-seq",
    "methylation_gene": "Methylation",
    "cnv_gene": "CNV",
    "mutation_gene": "SNV",
}


def make_integrated_gradients_subjects(train_views, test_views):
    analysis_views = []
    for train_view, test_view in zip(train_views, test_views):
        combined_view = np.concatenate([train_view, test_view], axis=0)
        analysis_views.append(combined_view.astype("float32", copy=False))
    return analysis_views


def make_integrated_gradients_baselines(view_keys, analysis_views):
    baselines = []

    for view_key, view_values in zip(view_keys, analysis_views):
        if view_key in ZERO_BASELINE_VIEWS:
            baseline_vector = np.zeros(view_values.shape[1], dtype="float32")
        elif view_key in MEAN_BASELINE_VIEWS:
            baseline_vector = np.nanmean(view_values, axis=0).astype("float32")
            baseline_vector[~np.isfinite(baseline_vector)] = 0.0
        else:
            baseline_vector = np.nanmean(view_values, axis=0).astype("float32")
            baseline_vector[~np.isfinite(baseline_vector)] = 0.0

        baselines.append(baseline_vector)
        print(
            f"{VIEW_DISPLAY_NAMES.get(view_key, view_key)} integrated-gradients baseline: "
            f"{'zero' if view_key in ZERO_BASELINE_VIEWS else 'mean across all subjects'}"
        )

    return baselines


def load_feature_names(view_key, expected_count):
    feature_path = CACHE_DIR / f"selected_features_{view_key}.tsv"
    fallback_names = [f"feature_{index + 1:05d}" for index in range(expected_count)]

    if not feature_path.exists():
        print(f"Could not find feature names for {view_key}; using feature indices.")
        return fallback_names

    feature_df = pd.read_csv(feature_path, sep="\t")
    if "feature_id" not in feature_df.columns:
        print(f"{feature_path} does not contain feature_id; using feature indices.")
        return fallback_names

    feature_names = feature_df["feature_id"].astype(str).tolist()
    if len(feature_names) != expected_count:
        print(
            f"{feature_path} contains {len(feature_names)} names but the data has "
            f"{expected_count} columns; using feature indices."
        )
        return fallback_names

    return feature_names


def plot_top_integrated_gradients(
    mean_abs_integrated_gradients,
    mean_signed_integrated_gradients,
    feature_names,
    view_key,
    dlv_index,
):
    top_n = min(INTEGRATED_GRADIENTS_TOP_N, len(mean_abs_integrated_gradients))
    if top_n == 0:
        print(f"No integrated-gradients values to plot for {view_key}.")
        return []

    importance_values = np.asarray(mean_abs_integrated_gradients, dtype="float32")
    importance_values[~np.isfinite(importance_values)] = 0.0
    signed_values = np.asarray(mean_signed_integrated_gradients, dtype="float32")
    signed_values[~np.isfinite(signed_values)] = 0.0

    top_indices = np.argsort(importance_values)[-top_n:][::-1]
    top_values = importance_values[top_indices]
    top_feature_names = [feature_names[index] for index in top_indices]

    plot_values = top_values[::-1]
    plot_names = top_feature_names[::-1]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    y_positions = np.arange(len(plot_names))
    ax.barh(y_positions, plot_values, color="#2f7f72")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_names)
    ax.set_xlabel("Mean absolute integrated gradients value")
    ax.set_title(
        f"{VIEW_DISPLAY_NAMES.get(view_key, view_key)} DLV{dlv_index + 1} top loci"
    )
    fig.tight_layout()
    print(f"Displaying {VIEW_DISPLAY_NAMES.get(view_key, view_key)} integrated-gradients plot.")
    plt.show(block=True)
    plt.close(fig)

    rows = []
    for rank, feature_index in enumerate(top_indices, start=1):
        rows.append({
            "view": view_key,
            "rank": rank,
            "feature_id": feature_names[feature_index],
            "feature_column": int(feature_index),
            "mean_abs_integrated_gradient": float(importance_values[feature_index]),
            "mean_signed_integrated_gradient": float(signed_values[feature_index]),
        })
    return rows


def run_integrated_gradients_analysis(dlvpm_model, view_keys, train_views, test_views):
    output_dir = Path(INTEGRATED_GRADIENTS_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_views = make_integrated_gradients_subjects(train_views, test_views)
    baselines = make_integrated_gradients_baselines(view_keys, analysis_views)

    for dlv_index in INTEGRATED_GRADIENTS_DLV_INDICES:
        print(f"\nIntegrated gradients on StructuralModel DLV{dlv_index + 1}")

        attribution_values = calculate_integrated_gradients(
            dlvpm_model,
            analysis_views,
            baseline=baselines,
            dlv_index=dlv_index,
            steps=INTEGRATED_GRADIENTS_STEPS,
        )

        summary_rows = []
        for view_key, view_attributions in zip(view_keys, attribution_values):
            mean_abs_integrated_gradients = np.nanmean(np.abs(view_attributions), axis=0)
            mean_signed_integrated_gradients = np.nanmean(view_attributions, axis=0)
            feature_names = load_feature_names(view_key, len(mean_abs_integrated_gradients))
            summary_rows.extend(
                plot_top_integrated_gradients(
                    mean_abs_integrated_gradients,
                    mean_signed_integrated_gradients,
                    feature_names,
                    view_key,
                    dlv_index,
                )
            )

        if summary_rows:
            summary_path = output_dir / f"dlv{dlv_index + 1}_integrated_gradients_top_loci.tsv"
            pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)
            print(f"Saved integrated-gradients top-loci summary to {summary_path}")

        del attribution_values
        clear_torch_memory()

    del analysis_views, baselines
    clear_torch_memory()


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

    def regularization_loss(self):
        penalty = torch.zeros((), dtype=next(self.parameters()).dtype, device=next(self.parameters()).device)
        if NEURAL_COX_L2:
            for parameter in self.parameters():
                penalty = penalty + NEURAL_COX_L2 * torch.sum(parameter ** 2)
        return penalty


class DirectOmicsLinearCox(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.risk_head = nn.Linear(input_dim, 1)

    def forward(self, inputs):
        return self.risk_head(inputs)

    def regularization_loss(self):
        penalty = torch.zeros((), dtype=next(self.parameters()).dtype, device=next(self.parameters()).device)
        if LINEAR_COX_L2:
            penalty = penalty + LINEAR_COX_L2 * torch.sum(self.risk_head.weight ** 2)
        return penalty


def fit_direct_model(model, train_inputs, train_y, batch_size, epochs):
    tensors = [torch.as_tensor(value, dtype=torch.float32) for value in train_inputs]
    y_tensor = torch.as_tensor(train_y, dtype=torch.float32)
    dataset = TensorDataset(*(tensors + [y_tensor]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(int(epochs)):
        model.train()
        losses = []
        for batch in loader:
            batch_inputs = [tensor.to(device) for tensor in batch[:-1]]
            batch_y = batch[-1].to(device)
            optimizer.zero_grad(set_to_none=True)
            risk = model(batch_inputs)
            loss = cox_partial_likelihood_loss(batch_y, risk) + model.regularization_loss()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"Epoch {epoch + 1}/{epochs} - loss: {np.mean(losses):.5f}")
    return model


def fit_direct_linear_cox_model(model, train_features, train_y, batch_size, epochs):
    x_tensor = torch.as_tensor(train_features, dtype=torch.float32)
    y_tensor = torch.as_tensor(train_y, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(int(epochs)):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            risk = model(batch_x)
            loss = cox_partial_likelihood_loss(batch_y, risk) + model.regularization_loss()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"Epoch {epoch + 1}/{epochs} - loss: {np.mean(losses):.5f}")
    return model


def predict_direct_model(model, inputs, batch_size):
    tensors = [torch.as_tensor(value, dtype=torch.float32) for value in inputs]
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = next(model.parameters()).device
    model.eval()
    risks = []
    with torch.no_grad():
        for batch in loader:
            batch_inputs = [tensor.to(device) for tensor in batch]
            risks.append(model(batch_inputs).detach().cpu())
    return torch.cat(risks, dim=0).numpy()


def predict_direct_linear_cox_model(model, features, batch_size):
    x_tensor = torch.as_tensor(features, dtype=torch.float32)
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = next(model.parameters()).device
    model.eval()
    risks = []
    with torch.no_grad():
        for (batch_x,) in loader:
            risks.append(model(batch_x.to(device)).detach().cpu())
    return torch.cat(risks, dim=0).numpy()


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

complete_train_mask = train_view_present.all(axis=1)
complete_test_mask = test_view_present.all(axis=1)
X_train = [train_view[complete_train_mask] for train_view in X_train]
X_test = [test_view[complete_test_mask] for test_view in X_test]
train_view_present = train_view_present[complete_train_mask]
test_view_present = test_view_present[complete_test_mask]
train_split = train_split.loc[complete_train_mask].reset_index(drop=True)
test_split = test_split.loc[complete_test_mask].reset_index(drop=True)

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
if not train_view_present.all():
    raise ValueError("All training patients should have every data view after complete-case filtering.")
if not test_view_present.all():
    raise ValueError("All test patients should have every data view after complete-case filtering.")

print(f"Training patients: {len(train_split)}")
print(f"Training patients removed for missing views: {int((~complete_train_mask).sum())}")
print(f"Test patients:     {len(test_split)}")
print(f"Test patients removed for missing views: {int((~complete_test_mask).sum())}")
print(f"Training events:   {int(train_events.sum())}")
print(f"Test events:       {int(test_events.sum())}")
print(f"Batch size:        {BATCH_SIZE}")
print(f"Epochs per model:  {EPOCHS}")
print(f"Learning rate:     {LEARNING_RATE}")


print("\nDLVPM + penalised Cox")

path_matrix = np.ones((n_views, n_views), dtype="float32")
np.fill_diagonal(path_matrix, 0.0)

dlvpm_encoders = [residual_encoder(view.shape[1], view_key) for view_key, view in zip(available_views, X_train)]
regularizer_list = [dlvpm_regularizers.l1_l2(l1=0.0, l2=0.0) for _ in available_views]

dlvpm_model = StructuralModel(
    Path=path_matrix,
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

print("DLVPM train metrics:", dlvpm_model.evaluate(X_train, batch_size=BATCH_SIZE, verbose=False))
print("DLVPM test metrics:", dlvpm_model.evaluate(X_test, batch_size=BATCH_SIZE, verbose=False))

train_dlvs = dlvpm_model.predict(X_train, batch_size=BATCH_SIZE, verbose=False)
test_dlvs = dlvpm_model.predict(X_test, batch_size=BATCH_SIZE, verbose=False)
train_patient_dlvs = (train_dlvs * train_view_present[:, np.newaxis, :]).sum(axis=2) / train_counts[:, np.newaxis]
test_patient_dlvs = (test_dlvs * test_view_present[:, np.newaxis, :]).sum(axis=2) / test_counts[:, np.newaxis]
results = [fit_penalised_cox("DLVPM + penalised Cox", train_patient_dlvs, test_patient_dlvs)]

if RUN_INTEGRATED_GRADIENTS:
    run_integrated_gradients_analysis(dlvpm_model, available_views, X_train, X_test)

del dlvpm_model, train_dlvs, test_dlvs
clear_torch_memory()


print("\nDirect multimodal neural Cox")

train_flags = [train_view_present[:, i : i + 1].astype("float32") for i in range(n_views)]
test_flags = [test_view_present[:, i : i + 1].astype("float32") for i in range(n_views)]
direct_train_inputs = X_train + train_flags
direct_test_inputs = X_test + test_flags

direct_model = DirectMultimodalDeepCox(available_views, X_train)
fit_direct_model(direct_model, direct_train_inputs, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS)
train_risk = predict_direct_model(direct_model, direct_train_inputs, batch_size=BATCH_SIZE).reshape(-1)
test_risk = predict_direct_model(direct_model, direct_test_inputs, batch_size=BATCH_SIZE).reshape(-1)
results.append({
    "method": "Direct multimodal neural Cox",
    "train_c_index": concordance_index(train_times, -train_risk, train_events),
    "test_c_index": concordance_index(test_times, -test_risk, test_events),
    "test_risk": test_risk,
})
print(
    "Direct multimodal neural Cox: "
    f"train C-index={results[-1]['train_c_index']:.3f}, "
    f"test C-index={results[-1]['test_c_index']:.3f}"
)

del direct_model
clear_torch_memory()


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
train_risk = predict_direct_linear_cox_model(linear_cox_model, linear_train_features, batch_size=BATCH_SIZE).reshape(-1)
test_risk = predict_direct_linear_cox_model(linear_cox_model, linear_test_features, batch_size=BATCH_SIZE).reshape(-1)
results.append({
    "method": "Direct omics linear Cox",
    "train_c_index": concordance_index(train_times, -train_risk, train_events),
    "test_c_index": concordance_index(test_times, -test_risk, test_events),
    "test_risk": test_risk,
})
print(
    "Direct omics linear Cox: "
    f"train C-index={results[-1]['train_c_index']:.3f}, "
    f"test C-index={results[-1]['test_c_index']:.3f}"
)

del linear_cox_model, linear_train_features, linear_test_features
clear_torch_memory()


representation_train_data = X_train
print(f"\nTraining multimodal representation models on {len(train_split)} complete training patients.")

for method_name in MULTIMODAL_METHODS:
    print(f"\n{method_name} + penalised Cox")

    model_name = method_name.lower()
    encoders = [residual_encoder(view.shape[1], f"{model_name}_{view_key}") for view_key, view in zip(available_views, X_train)]
    regularizer_list = [dlvpm_regularizers.l1_l2(l1=0.0, l2=0.0) for _ in available_views]

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
    representation_model.fit(representation_train_data, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, verbose=True)

    train_rep_by_view = representation_model.predict(X_train, batch_size=BATCH_SIZE, verbose=False)
    test_rep_by_view = representation_model.predict(X_test, batch_size=BATCH_SIZE, verbose=False)
    train_representations = (train_rep_by_view * train_view_present[:, np.newaxis, :]).sum(axis=2) / train_counts[:, np.newaxis]
    test_representations = (test_rep_by_view * test_view_present[:, np.newaxis, :]).sum(axis=2) / test_counts[:, np.newaxis]
    results.append(fit_penalised_cox(f"{method_name} + penalised Cox", train_representations, test_representations))

    del representation_model
    clear_torch_memory()


sorted_results = sorted(results, key=lambda result: result["test_c_index"], reverse=True)
results_table = pd.DataFrame(sorted_results).drop(columns=["test_risk"]).reset_index(drop=True)
print("\nSurvival prediction results")
print(results_table.to_string(index=False, formatters={"train_c_index": "{:.3f}".format, "test_c_index": "{:.3f}".format}))

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
plt.errorbar(plot_df["method"], plot_df["mean_c_index"], yerr=[plot_df["mean_c_index"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["mean_c_index"]], fmt="o", capsize=4)
plt.ylabel("Test C-index")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
