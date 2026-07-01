

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from deep_lvpm import regularizers as dlvpm_regularizers
from deep_lvpm.model import StructuralModel

# ============================================================
# Configuration
# ============================================================

SEED = 1337
BATCH_SIZE = 2048
EPOCHS = 100
WEIGHT_DECAY = 0
NDIMS = 1024
LEARNING_RATE = 1e-4
MAX_GRADIENT_NORM = 1
REQUESTED_DEVICE = "auto"  # "auto", "cpu", "cuda", or "mps"
PRINT_AUGMENTATION_DIAGNOSTICS = True
CIFAR10_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Define core dataset metadata.
NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)


def choose_torch_device(requested_device="auto"):
    """Return the requested PyTorch device."""
    requested_device = str(requested_device).lower()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("REQUESTED_DEVICE='cuda', but CUDA is not available.")
    if requested_device == "mps" and not mps_available:
        raise RuntimeError("REQUESTED_DEVICE='mps', but Apple Silicon MPS is not available.")

    return torch.device(requested_device)


DEVICE = choose_torch_device(REQUESTED_DEVICE)
print(f"Using PyTorch device: {DEVICE}")

# Load CIFAR-10 and flatten label arrays.
train_dataset = datasets.CIFAR10(root=str(CIFAR10_DATA_DIR), train=True, download=True)
test_dataset = datasets.CIFAR10(root=str(CIFAR10_DATA_DIR), train=False, download=True)
x_train = train_dataset.data.astype("float32") / 255.0
y_train_cat = np.asarray(train_dataset.targets, dtype="int64")
x_test = test_dataset.data.astype("float32") / 255.0
y_test_cat = np.asarray(test_dataset.targets, dtype="int64")

# Prepare one-hot encodings for downstream evaluation.
y_train = np.eye(NUM_CLASSES, dtype="float32")[y_train_cat]
y_test = np.eye(NUM_CLASSES, dtype="float32")[y_test_cat]

# Fix seeds to keep runs reproducible.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Split training set into train/validation partitions.
VAL_FRACTION = 0.1
num_train = x_train.shape[0]
indices = np.arange(num_train)
rng = np.random.default_rng(SEED)
rng.shuffle(indices)
cutoff = int(num_train * (1 - VAL_FRACTION))
x_tr, x_val = x_train[indices[:cutoff]], x_train[indices[cutoff:]]

# Build stochastic augmentation pipeline used to form siamese views.


def _to_channel_first(batch: torch.Tensor) -> torch.Tensor:
    if batch.ndim == 4 and batch.shape[-1] == 3:
        batch = batch.permute(0, 3, 1, 2)
    return batch.contiguous()


def augment(batch: torch.Tensor, training: bool = True) -> torch.Tensor:
    """Apply independent CIFAR augmentations to a batch."""
    batch = _to_channel_first(batch.float())
    if not training:
        return batch

    cropped_images = []
    for image in batch:
        top = torch.randint(0, 32 - 24 + 1, (1,)).item()
        left = torch.randint(0, 32 - 24 + 1, (1,)).item()
        cropped_images.append(image[:, top:top + 24, left:left + 24])
    x = torch.stack(cropped_images, dim=0)
    x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)

    flip_mask = torch.rand(x.shape[0]) < 0.5
    if torch.any(flip_mask):
        x[flip_mask] = torch.flip(x[flip_mask], dims=(3,))

    gray_mask = torch.rand(x.shape[0], 1, 1, 1) < 0.2
    grayscale_weights = torch.as_tensor(
        [0.2989, 0.5870, 0.1140],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 3, 1, 1)
    grayscale = torch.sum(x * grayscale_weights, dim=1, keepdim=True).repeat(1, 3, 1, 1)
    x = torch.where(gray_mask.to(x.device), grayscale, x)
    return x


class SiameseViewsDataset(Dataset):
    """Return raw CIFAR images for batch-level siamese augmentation."""

    def __init__(self, x, training=True):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.training = bool(training)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index]


def collate_siamese_views(batch, training=True):
    batch = torch.stack(batch, dim=0)
    view_one = augment(batch, training=training)
    view_two = augment(batch, training=training)
    return ((view_one, view_two),)


def make_siamese_views_dataset(x, batch_size=256, shuffle=True, training=True):
    """Return a dataloader that yields pairs of augmented views."""
    dataset = SiameseViewsDataset(x, training=training)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=bool(training),
        collate_fn=lambda batch: collate_siamese_views(batch, training=training),
    )


def print_augmentation_diagnostics(data_loader):
    if not PRINT_AUGMENTATION_DIAGNOSTICS:
        return

    rng_state = torch.random.get_rng_state()
    try:
        first_batch = next(iter(data_loader))
        view_one, view_two = first_batch[0]

        flat_one = view_one.reshape(view_one.shape[0], -1)
        flat_two = view_two.reshape(view_two.shape[0], -1)
        flat_one = flat_one - flat_one.mean(dim=1, keepdim=True)
        flat_two = flat_two - flat_two.mean(dim=1, keepdim=True)
        numerator = torch.sum(flat_one * flat_two, dim=1)
        denominator = torch.sqrt(torch.sum(flat_one ** 2, dim=1) * torch.sum(flat_two ** 2, dim=1) + 1e-8)
        pixel_correlation = numerator / denominator

        mean_abs_difference = torch.mean(torch.abs(view_one - view_two)).item()
        mean_pixel_correlation = torch.mean(pixel_correlation).item()
        identical_pairs = (view_one == view_two).reshape(view_one.shape[0], -1).all(dim=1).sum().item()

        print(
            "Augmentation diagnostic: "
            f"mean_abs_difference={mean_abs_difference:.4f}, "
            f"mean_pixel_correlation={mean_pixel_correlation:.4f}, "
            f"identical_pairs={identical_pairs}/{view_one.shape[0]}"
        )
    finally:
        torch.random.set_rng_state(rng_state)


# Create datasets for siamese training.
train_ds = make_siamese_views_dataset(
    x_tr, batch_size=BATCH_SIZE, shuffle=False, training=True
)
val_ds = make_siamese_views_dataset(
    x_val, batch_size=BATCH_SIZE, shuffle=False, training=True
)
print_augmentation_diagnostics(train_ds)


class CIFARResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(output_channels)

        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs):
        shortcut = self.shortcut(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = x + shortcut
        x = F.relu(x)
        return x


class CIFARImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.stage1 = nn.Sequential(
            CIFARResidualBlock(128, 128),
            CIFARResidualBlock(128, 128),
        )
        self.stage2 = nn.Sequential(
            CIFARResidualBlock(128, 256, stride=2),
            CIFARResidualBlock(256, 256),
        )
        self.stage3 = nn.Sequential(
            CIFARResidualBlock(256, 512, stride=2),
            CIFARResidualBlock(512, 512),
        )
        self.stage4 = nn.Sequential(
            CIFARResidualBlock(512, 1024, stride=2),
            CIFARResidualBlock(1024, 1024),
        )
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense1 = nn.Linear(1024, 1024)
        self.n_inputs = 1
        self.apply(self._initialize_residual_model)

        for module in self.modules():
            if isinstance(module, CIFARResidualBlock):
                nn.init.zeros_(module.bn2.weight)

    def _initialize_residual_model(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def extract_average_pool_features(self, inputs):
        x = self.stem(inputs)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_average_pool(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, inputs):
        x = self.extract_average_pool_features(inputs)
        x = self.dense1(x)

        return x

    def regularization_loss(self):
        if WEIGHT_DECAY == 0:
            first_parameter = next(self.parameters())
            return torch.zeros((), dtype=first_parameter.dtype, device=first_parameter.device)
        first_parameter = next(self.parameters())
        penalty = torch.zeros((), dtype=first_parameter.dtype, device=first_parameter.device)
        for parameter in self.parameters():
            penalty = penalty + WEIGHT_DECAY * torch.sum(parameter ** 2)
        return penalty


class ClippedAdam(torch.optim.Adam):
    """Adam with Keras-style clipnorm applied inside optimizer.step()."""

    def __init__(self, params, clipnorm=None, **kwargs):
        super().__init__(params, **kwargs)
        self.clipnorm = clipnorm

    def step(self, closure=None):
        if self.clipnorm is not None:
            parameters_with_grad = []
            seen_parameter_ids = set()
            for param_group in self.param_groups:
                for parameter in param_group["params"]:
                    if parameter.grad is None or id(parameter) in seen_parameter_ids:
                        continue
                    parameters_with_grad.append(parameter)
                    seen_parameter_ids.add(id(parameter))
            if parameters_with_grad:
                nn.utils.clip_grad_norm_(parameters_with_grad, max_norm=float(self.clipnorm))

        return super().step(closure=closure)


CIFAR_image_model = CIFARImageModel()

# Build siamese DLVPM model with shared encoder replicas and ZCA
# orthogonalization across the two augmented image views.
model_list = [CIFAR_image_model, CIFAR_image_model]
adjacency = np.array([[0, 1], [1, 0]], dtype="float32")
regularizers = [dlvpm_regularizers.l2(WEIGHT_DECAY),dlvpm_regularizers.l2(WEIGHT_DECAY)]

dlvpm_model = StructuralModel(
    adjacency,
    model_list,
    regularizers,
    x_train.shape[0],
    NDIMS,
    orthogonalization="zca",
    train_DLV=True,
    is_siamese=True,
    diag_offset=1e-12,
    device=DEVICE,
)

# Compile with one shared optimiser, matching the shared Keras Siamese model.
shared_optimizer = ClippedAdam(
    dlvpm_model.model_list[0].parameters(),
    lr=LEARNING_RATE,
    eps=1e-7,
    clipnorm=MAX_GRADIENT_NORM,
)
optimizers = [shared_optimizer, shared_optimizer]
dlvpm_model.compile(optimizers)

# Train the siamese model and monitor validation performance.
dlvpm_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=True)


def predict_view_model(view_model: nn.Module, x, batch_size=32) -> np.ndarray:
    view_model.eval()
    x_tensor = _to_channel_first(torch.as_tensor(x, dtype=torch.float32))
    loader = DataLoader(x_tensor, batch_size=batch_size, shuffle=False)
    outputs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(dlvpm_model.device)
            outputs.append(view_model(batch).detach().cpu())
    return torch.cat(outputs, dim=0).numpy()


def predict_average_pool_features(view_model: nn.Module, x, batch_size=32) -> np.ndarray:
    view_model.eval()
    encoder = getattr(view_model, "encoder", view_model)

    if not hasattr(encoder, "extract_average_pool_features"):
        raise AttributeError("The image encoder does not expose extract_average_pool_features().")

    x_tensor = _to_channel_first(torch.as_tensor(x, dtype=torch.float32))
    loader = DataLoader(x_tensor, batch_size=batch_size, shuffle=False)
    outputs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(dlvpm_model.device)
            features = encoder.extract_average_pool_features(batch)
            outputs.append(features.detach().cpu())
    return torch.cat(outputs, dim=0).numpy()


def evaluate_linear_svm(feature_name, train_features, test_features):
    print(f"\nLinear SVM evaluation on {feature_name}")
    print(f"Train feature shape: {train_features.shape}")
    print(f"Test  feature shape: {test_features.shape}")

    svm_clf = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True)),
            ("svm", LinearSVC(C=1.0, max_iter=10000, random_state=42)),
        ]
    )
    svm_clf.fit(train_features, y_train_cat)
    predictions = svm_clf.predict(test_features)
    accuracy = accuracy_score(y_test_cat, predictions)

    print(f"SVM accuracy on CIFAR-10 test set: {accuracy:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test_cat, predictions, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test_cat, predictions))

    return {
        "accuracy": accuracy,
        "predictions": predictions,
    }


# Generate embeddings for downstream linear evaluation.
image_model = dlvpm_model.model_list[0]
train_dlvs = predict_view_model(image_model, x_train, batch_size=32)
test_dlvs = predict_view_model(image_model, x_test, batch_size=32)
train_average_pool_features = predict_average_pool_features(image_model, x_train, batch_size=32)
test_average_pool_features = predict_average_pool_features(image_model, x_test, batch_size=32)

test_ds = make_siamese_views_dataset(
    x_test, batch_size=BATCH_SIZE, shuffle=False, training=True
)

test_metrics = dlvpm_model.evaluate(test_ds)
print("Test DLVPM metrics:", test_metrics)

linear_evaluation_results = {}
linear_evaluation_results["final_dlvpm_factors"] = evaluate_linear_svm(
    "final DLVPM factors",
    train_dlvs,
    test_dlvs,
)
linear_evaluation_results["average_pool_features"] = evaluate_linear_svm(
    "last average pooling layer features",
    train_average_pool_features,
    test_average_pool_features,
)

print("\nLinear SVM accuracy summary:")
for result_name, result in linear_evaluation_results.items():
    print(f"{result_name}: {result['accuracy']:.4f}")

test_dlvs_pair = dlvpm_model.predict(test_ds)
corr_mat = dlvpm_model.calculate_corrmat(test_dlvs_pair)
corrmean = [np.mean(a.detach().cpu().numpy()) if torch.is_tensor(a) else np.mean(a) for a in corr_mat]
print("Mean latent correlation per factor:", corrmean)
