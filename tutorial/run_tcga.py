####### TCGA Multi-Omics Tutorial (PyTorch) #########
"""
Demonstrates DLVPM on a 5-modality lung cancer dataset using the pure
PyTorch backend.  Trains residual encoders for each omics/imaging
modality and plots inter-view correlation chord diagrams.
"""

from datetime import datetime
from pathlib import Path as _Path

import numpy as np
import torch
import torch.nn as nn

from deep_lvpm.model import StructuralModel
from deep_lvpm.optim import make_encoder_optimizer
from deep_lvpm.plot import (
    plot_correlation_graph,
    plot_correlation_matrix,
    plot_training_history,
)

# ------------------------------------------------------------------
# Log directory
# ------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = _Path("logs") / f"{timestamp}_run_tcga"
log_dir.mkdir(parents=True, exist_ok=True)
print(f"Logging outputs to: {log_dir.resolve()}")

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

# train
arrays = np.load("data/Lung_multiomics_sample_train.npz")
rnaseq = arrays["rnaseq"]
snv = arrays["snv"]
methylation = arrays["methylation"]
mirna = arrays["mirna"]
histo20 = arrays["histo20"]
X_train = [histo20, rnaseq, methylation, mirna, snv]

# test
arrays = np.load("data/Lung_multiomics_sample_test.npz")
rnaseq_test = arrays["rnaseq"]
snv_test = arrays["snv"]
methylation_test = arrays["methylation"]
mirna_test = arrays["mirna"]
histo20_test = arrays["histo20"]
X_val_test = [histo20_test, rnaseq_test, methylation_test, mirna_test, snv_test]

# ------------------------------------------------------------------
# Residual encoder (pure PyTorch nn.Module)
# ------------------------------------------------------------------


class ResidualBlock(nn.Module):
    """Fully-connected residual block."""

    def __init__(
        self,
        input_dim: int,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.bn = nn.BatchNorm1d(input_dim, momentum=0.01, eps=0.001)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.eye_(self.linear1.weight)
        nn.init.eye_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = out + x
        out = self.dropout(out)
        return out

def residual_encoder(input_dim: int, name: str = "residual_enc") -> nn.Sequential:
    """Wrap a ResidualBlock in a Sequential and tag with n_inputs."""
    model = nn.Sequential(ResidualBlock(input_dim))
    model.n_inputs = 1
    return model


model_list = [
    residual_encoder(histo20.shape[1], "histo20_enc"),
    residual_encoder(rnaseq.shape[1], "rnaseq_enc"),
    residual_encoder(methylation.shape[1], "meth_enc"),
    residual_encoder(mirna.shape[1], "mirna_enc"),
    residual_encoder(snv.shape[1], "snv_enc"),
]


# ------------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------------

ndims = 5  # number of latent factors

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
tot_num = rnaseq.shape[0]

regularizer_list = [(0.001, 0.001)] * len(model_list)

model = StructuralModel(
    Path,
    model_list,
    regularizer_list,
    tot_num,
    ndims,
    momentum=0.95,
    epsilon=0.001,
    orthogonalization="Moore-Penrose",
    train_DLV=True,
)

# Learning-rate schedule: exponential decay matching the original
# (init_lr → final_lr over `epochs` epochs, stepped once per epoch).
init_lr, final_lr = 1e-4, 1e-5
gamma = (final_lr / init_lr) ** (1.0 / epochs)

model.build(X_train)


opt_list = [make_encoder_optimizer(m, init_lr, weight_decay=0.02) for m in model.model_list]
schedulers = [
    torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma) for opt in opt_list
]

model.compile(optimizer=opt_list)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

history = model.fit(
    X_train=X_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=True,
    schedulers=schedulers,
    X_val=X_val_test,
)

mean_corr = model.evaluate(X_train)
print(
    "The mean correlation between data-types connected by the path model "
    f"is r={mean_corr['cross_metric']:.4f}"
)


# ------------------------------------------------------------------
# Test-set evaluation
# ------------------------------------------------------------------
mean_corr_test = model.evaluate(X_val_test)
print(
    "Test set — mean correlation between connected data-types: "
    f"r={mean_corr_test['cross_metric']:.4f}"
)

data_names = ["Histology", "RNASeq", "miRNASeq", "Methylation", "SNVs"]

# --- Prediction & visualization ---
for split_name, X in [("train", X_train), ("val_test", X_val_test)]:
    dlv = model.predict(X)

    print(f"\nCorrelations between first set of DLVs ({split_name}):")
    print(np.corrcoef(dlv[:, 0, :].T))
    print(f"Correlations between second set of DLVs ({split_name}):")
    print(np.corrcoef(dlv[:, 1, :].T))

    corr_mat = model.calculate_corrmat(dlv)

    graph_path = log_dir / f"corr_graph_{split_name}.png"
    plot_correlation_graph(
        corr_mat,
        data_names,
        figure_title=(
            "Correlation Graph Between Omics and Imaging Data Types in Lung "
            f"Cancer ({split_name} set)"
        ),
        save_path=graph_path,
        show=False,
    )
    print(f"Saved correlation graph ({split_name}) to {graph_path}")

    matrix_path = log_dir / f"corr_matrix_{split_name}.png"
    plot_correlation_matrix(
        corr_mat,
        data_names,
        figure_title=(
            "Correlation Matrix Between Omics and Imaging Data Types in Lung "
            f"Cancer ({split_name} set)"
        ),
        save_path=matrix_path,
        show=False,
    )
    print(f"Saved correlation matrix ({split_name}) to {matrix_path}")


# --- Training progress plot ---
history_path = log_dir / "training_history.png"
plot_training_history(
    history,
    save_path=history_path,
    show=False,
)
print(f"Saved training history plot to {history_path}")
