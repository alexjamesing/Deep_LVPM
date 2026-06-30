####### Tutorial 2 #########

import numpy as np
import torch
import torch.nn as nn
from importlib import resources

import deep_lvpm as DLVPM
from deep_lvpm import regularizers
from deep_lvpm.model import StructuralModel


with resources.as_file(resources.files("deep_lvpm.data") /
                       "Lung_multiomics_sample_train.npz") as f:
    arrays = np.load(f)
    rnaseq      = arrays["rnaseq"]
    snv         = arrays["snv"]
    methylation = arrays["methylation"]
    mirna       = arrays["mirna"]
    histo20     = arrays["histo20"]

X_arr = [histo20, rnaseq, methylation, mirna, snv]   # preserve this order!


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
        penalty = torch.zeros((), dtype=self.linear1.weight.dtype, device=self.linear1.weight.device)
        if self.kernel_reg_l1:
            penalty = penalty + self.kernel_reg_l1 * (
                torch.sum(torch.abs(self.linear1.weight)) + torch.sum(torch.abs(self.linear2.weight))
            )
        if self.kernel_reg_l2:
            penalty = penalty + self.kernel_reg_l2 * (
                torch.sum(self.linear1.weight ** 2) + torch.sum(self.linear2.weight ** 2)
            )
        return penalty


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
    residual_block(rnaseq.shape[1],  name="rnaseq_enc"),
    residual_block(methylation.shape[1], name="meth_enc"),
    residual_block(mirna.shape[1],   name="mirna_enc"),
    residual_block(snv.shape[1],     name="snv_enc"),
]


ndims = 5        # number of latent factors

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

lr_gamma = (final_lr / init_lr) ** (1.0 / max(epochs, 1))

tot_num = rnaseq.shape[0] ## This is the total number of samples under analysis and is needed by DLVPM


regularizer_list = [regularizers.l1_l2(l1=0.001, l2=0.001),regularizers.l1_l2(l1=0.001, l2=0.001),regularizers.l1_l2(l1=0.001, l2=0.001),regularizers.l1_l2(l1=0.001, l2=0.001),regularizers.l1_l2(l1=0.001, l2=0.001)] ## These regularizers are applied to the final "projection" layer of the DLVPM model, used internally

DLVPM_Structural_instance = StructuralModel(Path, model_list, regularizer_list, tot_num, ndims, momentum=0.95,epsilon=0.001, orthogonalization='zca', train_DLV =True, order=True)

opt_list = [torch.optim.Adam(model.parameters(), lr=init_lr) for model in DLVPM_Structural_instance.model_list]
scheduler_list = [torch.optim.lr_scheduler.ExponentialLR(opt, gamma=lr_gamma) for opt in opt_list]
DLVPM_Structural_instance.compile(optimizer=opt_list)


DLVPM_Structural_instance.fit(X_arr, batch_size=batch_size, epochs=epochs,verbose=True, schedulers=scheduler_list)
mean_corr = DLVPM_Structural_instance.evaluate(X_arr)

print('The mean correlation between data-types connected by the path model is r=' + str(mean_corr["cross_metric"]))


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

print('The mean correlation between data-types connected by the path model is r=' + str(mean_corr_test["cross_metric"]))

test_DLVs = DLVPM_Structural_instance.predict(X_arr_test) ## Here, we obtain the full set of test_DLVs

## Associations between the first set of DLVs are:
print(np.corrcoef(test_DLVs[:,0,:].T))
## Associations between the second set of DLVs are:
print(np.corrcoef(test_DLVs[:,1,:].T))

corr_mat = DLVPM_Structural_instance.calculate_corrmat(test_DLVs) # This outputs correlation matrices between different data types included in the model

corrmean = [float(torch.mean(a.detach().cpu())) if torch.is_tensor(a) else float(np.mean(a)) for a in corr_mat]

print(corrmean)

corr_mat = DLVPM_Structural_instance.calculate_corrmat(test_DLVs) # This outputs correlation matrices between different data types included in the model

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
