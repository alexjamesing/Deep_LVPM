####### TCGA Multi-Omics Tutorial (PyTorch) #########
"""
Demonstrates DLVPM on a 5-modality lung cancer dataset using the pure
PyTorch backend.  Trains residual encoders for each omics/imaging
modality and plots inter-view correlation chord diagrams.
"""

from importlib import resources

import numpy as np
import torch
import torch.nn as nn

from deep_lvpm.model import StructuralModel
from deep_lvpm.plot import plot_correlation_chord_row

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Residual encoder (pure PyTorch nn.Module)
# ------------------------------------------------------------------


class ResidualBlock(nn.Module):
    """Fully-connected residual block."""

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

    def regularization_loss(self) -> torch.Tensor:
        device = self.linear1.weight.device
        dtype = self.linear1.weight.dtype
        penalty = torch.zeros((), device=device, dtype=dtype)
        if self.kernel_reg_l1:
            penalty = penalty + self.kernel_reg_l1 * (
                self.linear1.weight.abs().sum() + self.linear2.weight.abs().sum()
            )
        if self.kernel_reg_l2:
            penalty = penalty + self.kernel_reg_l2 * (
                (self.linear1.weight**2).sum() + (self.linear2.weight**2).sum()
            )
        return penalty


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

DLVPM_Structural_instance = StructuralModel(
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

DLVPM_Structural_instance.build(X_arr)
opt_list = [
    torch.optim.Adam(m.parameters(), lr=init_lr)
    for m in DLVPM_Structural_instance.model_list
]
schedulers = [
    torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma) for opt in opt_list
]

DLVPM_Structural_instance.compile(optimizer=opt_list)


# ------------------------------------------------------------------
# Training  (one epoch at a time so the LR scheduler steps each epoch)
# ------------------------------------------------------------------

for epoch in range(epochs):
    h = DLVPM_Structural_instance.fit(X_arr, batch_size=batch_size, epochs=1, verbose=False)
    for sched in schedulers:
        sched.step()
    print(
        f"Epoch {epoch + 1}/{epochs} — "
        f"loss: {h['total_loss'][-1]:.4f}  "
        f"corr: {h['cross_metric'][-1]:.4f}  "
        f"red: {h['redundancy'][-1]:.4f}"
    )

mean_corr = DLVPM_Structural_instance.evaluate(X_arr)
print(
    "The mean correlation between data-types connected by the path model "
    f"is r={mean_corr['cross_metric']:.4f}"
)


# ------------------------------------------------------------------
# Test-set evaluation
# ------------------------------------------------------------------

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
    "Test set — mean correlation between connected data-types: "
    f"r={mean_corr_test['cross_metric']:.4f}"
)

test_DLVs = DLVPM_Structural_instance.predict(X_arr_test)

print("Correlations between first set of DLVs:")
print(np.corrcoef(test_DLVs[:, 0, :].T))
print("Correlations between second set of DLVs:")
print(np.corrcoef(test_DLVs[:, 1, :].T))

corr_mat = DLVPM_Structural_instance.calculate_corrmat(test_DLVs)

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
