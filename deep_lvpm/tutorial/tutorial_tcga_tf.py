
####### Tutorial 2 #########

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import layers, regularizers, optimizers
from importlib import resources

import deep_lvpm as DLVPM
from deep_lvpm.model import StructuralModel

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


