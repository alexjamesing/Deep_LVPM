####### MNIST Tutorial (PyTorch) #########
"""
Demonstrates DLVPM on MNIST: associates image features (CNN encoder)
with one-hot class labels using ZCA orthogonalisation.

Requires: torchvision
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms as T

from deep_lvpm.model import StructuralModel


def _load_mnist():
    transform = T.Compose([T.ToTensor()])
    train_ds = datasets.MNIST(
        root="/tmp/mnist", train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root="/tmp/mnist", train=False, download=True, transform=transform
    )

    def ds_to_arrays(ds):
        x = ds.data.float().unsqueeze(1) / 255.0  # (N, 1, 28, 28)
        y_cat = ds.targets
        y = torch.zeros(len(y_cat), 10)
        y[torch.arange(len(y_cat)), y_cat] = 1.0
        return x.numpy(), y.numpy(), y_cat.numpy()

    x_train, y_train, y_train_cat = ds_to_arrays(train_ds)
    x_test, y_test, y_test_cat = ds_to_arrays(test_ds)

    return x_train, y_train, y_train_cat, x_test, y_test, y_test_cat


x_train, y_train, y_train_cat, x_test, y_test, y_test_cat = _load_mnist()

print("x_train shape:", x_train.shape)
print(f"{x_train.shape[0]} train samples, {x_test.shape[0]} test samples")

data_train_list = [x_train, y_train]
data_test_list = [x_test, y_test]


# ------------------------------------------------------------------
# Measurement models
# ------------------------------------------------------------------


class CNNEncoder(nn.Module):
    """Small CNN to encode 28×28 grayscale images."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        if x.ndim == 2:
            # Flatten case: not expected here but just in case
            x = x.view(-1, 1, 28, 28)
        return self.net(x)


class IdentityEncoder(nn.Module):
    """Pass-through for the label view."""

    def forward(self, x):
        return x


cnn_model = CNNEncoder()
label_model = IdentityEncoder()
cnn_model.n_inputs = 1
label_model.n_inputs = 1

model_list = [cnn_model, label_model]

Path = np.array([[0, 1], [1, 0]], dtype="float32")
regularizer_list = [None, None]

ndims = 9
tot_num = x_train.shape[0]
batch_size = 256
epochs = 20

DLVPM_Model = StructuralModel(
    Path,
    model_list,
    regularizer_list,
    tot_num,
    ndims,
    orthogonalization="zca",
    train_DLV=False,
)

DLVPM_Model.build(data_train_list)
opt_list = [torch.optim.Adam(m.parameters(), lr=1e-5) for m in DLVPM_Model.model_list]
DLVPM_Model.compile(optimizer=opt_list)

# Match original validation_split=0.1
val_cut = int(len(x_train) * 0.9)
data_val_list = [x_train[val_cut:], y_train[val_cut:]]
data_train_list = [x_train[:val_cut], y_train[:val_cut]]

DLVPM_Model.fit(data_train_list, batch_size=batch_size, epochs=epochs, verbose=True,
                validation_data=data_val_list)

metrics = DLVPM_Model.evaluate(data_test_list)

DLVs = DLVPM_Model.predict(data_test_list)

Cmat1 = np.corrcoef(DLVs[:, 0, :].T)
print("Correlation matrix (DLV 1):")
print(Cmat1)

image_DLVs = (
    DLVPM_Model.model_list[0].predict([data_test_list[0]])
    if hasattr(DLVPM_Model.model_list[0], "predict")
    else None
)

# t-SNE visualisation
try:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    if image_DLVs is None:
        # Obtain image DLVs manually
        DLVPM_Model.eval()
        with torch.no_grad():
            DLVPM_Model.model_list[0].eval()
            t = torch.as_tensor(data_test_list[0], dtype=torch.float32).to(
                DLVPM_Model.device
            )
            image_DLVs = DLVPM_Model.model_list[0](t).cpu().numpy()

    rng = np.random.default_rng(42)
    idx = rng.choice(image_DLVs.shape[0], size=100, replace=False)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(image_DLVs[idx])

    plt.figure(figsize=(12, 8))
    y_sub = y_test[idx]
    for i in range(y_sub.shape[1]):
        pts = tsne_results[y_sub[:, i] == 1]
        plt.scatter(pts[:, 0], pts[:, 1], label=f"Category {i + 1}")
    plt.title("t-SNE projection of MNIST image DLVs")
    plt.legend()
    plt.show()
except ImportError:
    print("scikit-learn / matplotlib not available — skipping t-SNE plot.")
