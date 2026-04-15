"""
Demonstrates the siamese mode of DLVPM on CIFAR-10:
both views share the same CNN encoder (is_siamese=True).

Two stochastically augmented views of each image are produced inside
the encoder's forward() so each view receives independent random
augmentations on every batch, following a self-supervised contrastive
learning setup.

Requires: torchvision, scikit-learn
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torchvision import datasets
from tqdm import tqdm

from deep_lvpm.model import StructuralModel

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
train_ds = datasets.CIFAR10(root="/tmp/cifar10", train=True, download=True)
test_ds = datasets.CIFAR10(root="/tmp/cifar10", train=False, download=True)

x_train = np.array(train_ds.data, dtype="float32") / 255.0  # (50000, 32, 32, 3)
y_train_cat = np.array(train_ds.targets)

x_test = np.array(test_ds.data, dtype="float32") / 255.0  # (10000, 32, 32, 3)
y_test_cat = np.array(test_ds.targets)

# Convert HWC → CHW for PyTorch
x_train = x_train.transpose(0, 3, 1, 2)  # (50000, 3, 32, 32)
x_test = x_test.transpose(0, 3, 1, 2)  # (10000, 3, 32, 32)

print(f"Train: {x_train.shape},  Test: {x_test.shape}")


# ------------------------------------------------------------------
# Batch augmentation  (applied inside the encoder's forward())
# ------------------------------------------------------------------


def batch_augment(x: torch.Tensor) -> torch.Tensor:
    """Random crop + horizontal flip + grayscale on an (N, C, H, W) batch."""
    N, C, H, W = x.shape
    # Random horizontal flip (50 %)
    flip = torch.rand(N, device=x.device) < 0.5
    x = torch.where(flip.view(N, 1, 1, 1), x.flip(-1), x)
    # Random crop: reflect-pad by 4 then crop back to original size
    pad = 4
    x = F.pad(x, [pad, pad, pad, pad], mode="reflect")
    top = torch.randint(0, 2 * pad + 1, (1,)).item()
    left = torch.randint(0, 2 * pad + 1, (1,)).item()
    x = x[:, :, top : top + H, left : left + W]
    # Random grayscale (20 %)
    gray = x.mean(dim=1, keepdim=True).expand_as(x)
    gray_mask = torch.rand(N, device=x.device) < 0.2
    x = torch.where(gray_mask.view(N, 1, 1, 1), gray, x)
    return x


# ------------------------------------------------------------------
# Encoder architecture
# ------------------------------------------------------------------


class CIFAREncoder(nn.Module):
    """
    CNN encoder matching the keras3 siamese tutorial.

    Augmentation is applied stochastically inside forward() when in
    train mode so that both siamese views of the same batch receive
    independent random augmentations.

    Structured as backbone + projector so backbone features can be
    extracted for downstream evaluation without the projection head.
    """

    def __init__(self, ndims: int = 2048) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → (64, 16, 16)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → (128, 8, 8)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # → (256, 1, 1)
            nn.Flatten(),  # → (256,)
        )
        self.projector = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, ndims),
            nn.ReLU(),
            nn.BatchNorm1d(ndims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = batch_augment(x)
        return self.projector(self.backbone(x))


# ------------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------------

NDIMS = 2048
batch_size = 2048
epochs = 200

encoder = CIFAREncoder(ndims=NDIMS)
encoder.n_inputs = 1

# Both entries point to the same encoder object; StructuralModel handles
# weight-tying via is_siamese=True.
model_list = [encoder, encoder]

Path = np.array([[0, 1], [1, 0]], dtype="float32")
regularizer_list = [None, None]

model = StructuralModel(
    Path,
    model_list,
    regularizer_list,
    x_train.shape[0],
    NDIMS,
    orthogonalization="zca",
    train_DLV=True,
    is_siamese=True,
    diag_offset=1e-4,
    momentum=0.95,
    epsilon=1e-4,
)

# Pass x_train twice: both views share the same underlying images but
# receive independent stochastic augmentations inside encoder.forward().
X_train = [x_train, x_train]

model.build(X_train)
opt = torch.optim.Adam(model.model_list[0].parameters(), lr=1e-4)
model.compile(optimizer=opt)

model.fit(
    X_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=True,
)

# ------------------------------------------------------------------
# Downstream evaluation: linear SVM on backbone features
# ------------------------------------------------------------------
# model_list[0] is nn.Sequential(CIFAREncoder, ZCALayer); [0][0] is the encoder.
backbone = model.model_list[0][0].backbone

model.eval()
backbone.eval()

device = model.device


def extract_features(
    x_np: np.ndarray, model: nn.Module, batch: int = 1024, desc: str = "extract"
) -> np.ndarray:
    t = torch.as_tensor(x_np, dtype=torch.float32)
    if torch.device(device).type == "cuda":
        t = t.pin_memory()
    outs = []
    n = len(t)
    with torch.inference_mode():
        for start in tqdm(range(0, n, batch), desc=desc, unit="batch"):
            chunk = t[start : start + batch].to(device, non_blocking=True)
            outs.append(model(chunk))
    return torch.cat(outs).cpu().numpy()


train_feats = extract_features(x_train, backbone, desc="train features")
test_feats = extract_features(x_test, backbone, desc="test features")

print(f"Backbone features — train: {train_feats.shape},  test: {test_feats.shape}")

svm_subset = 1000  # set to None to use all training features
if svm_subset is not None and svm_subset < len(train_feats):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(train_feats), size=svm_subset, replace=False)
    train_feats_svm = train_feats[idx]
    y_train_svm = y_train_cat[idx]
    print(f"Using {svm_subset}/{len(train_feats)} training samples for SVM")
else:
    train_feats_svm = train_feats
    y_train_svm = y_train_cat

svm_clf = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "svm",
            LinearSVC(C=1.0, dual=False, max_iter=1000, random_state=42, verbose=0),
        ),
    ]
)
svm_clf.fit(train_feats_svm, y_train_svm)
predictions = svm_clf.predict(test_feats)
accuracy = accuracy_score(y_test_cat, predictions)

print(f"\nSVM accuracy on CIFAR-10 test set: {accuracy:.4f}")
print("Classification report:")
print(classification_report(y_test_cat, predictions, digits=4))
print("Confusion matrix:")
print(confusion_matrix(y_test_cat, predictions))
