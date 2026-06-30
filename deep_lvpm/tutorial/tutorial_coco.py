#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tutorial: linking images and captions in MS COCO with DLVPM.

This tutorial is written as a procedural walkthrough rather than as a
standalone benchmark script. The aim is to show, step by step, how to:

1. load COCO images,
2. attach five human captions to every image,
3. treat the image and the captions as six linked data views,
4. train DLVPM to align those views in a shared latent space, and
5. evaluate the learned representation with an image-text retrieval task.

The code is intentionally top-down so that a new user can read one section,
run it, inspect the objects that are created, and then move to the next step.
"""

import os
import json
import random
import zipfile
import urllib.request
from collections import defaultdict

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:
    raise RuntimeError(
        "This tutorial uses native PyTorch. Install it with: "
        "python -m pip install -e ."
    ) from exc

try:
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
except Exception as exc:
    raise RuntimeError(
        "This tutorial requires torchvision. Install it with: python -m pip install torchvision"
    ) from exc

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    from transformers import AutoModel, AutoTokenizer
except Exception as exc:
    raise RuntimeError(
        "This tutorial uses Hugging Face Transformers with PyTorch. "
        "Install it with: python -m pip install 'transformers<5'"
    ) from exc

try:
    import fiftyone as fo
    import fiftyone.zoo as foz
except Exception as exc:
    raise RuntimeError(
        "This tutorial requires FiftyOne. Install it with: python -m pip install fiftyone"
    ) from exc

from deep_lvpm.model import StructuralModel
from deep_lvpm.multi_model import CLIP, VICReg, LeJEPA


# ============================================================
# Configuration
# ============================================================
# These are the main settings most users will want to edit.
#
# In this tutorial each sample contributes:
# - 1 image view
# - 5 caption views
#
# The five captions are treated as separate language views rather than being
# averaged together before training. This lets DLVPM learn which latent
# variables are shared between the image and each human description.
NUM_CAPTION_VIEWS = 5
IMG_SIZE = 224
MAX_TOKENS = 32
TEXT_MODEL_NAME = "distilbert-base-uncased"
TEXT_DROPOUT = 0.10

NDIMS = 512
BATCH_SIZE = 512
BENCHMARK_EPOCHS = 30

BENCHMARK_TRAIN_SAMPLES = 20000
BENCHMARK_VAL_SAMPLES = 5000
BENCHMARK_SAMPLES = 2048

NUM_WORKERS = 0
RUN_BASELINES = False
RETRIEVAL_KS = (1, 5, 10)
RANK_BOOTSTRAP_SAMPLES = 1000

LEARNING_RATE_START = 1e-5
LEARNING_RATE_END = 1e-4
LEARNING_RATE_WARMUP_EPOCHS = 5

TEST_FRACTION = 0.10
SEED = 51

# If you have already downloaded the COCO caption annotations, point this
# variable at the relevant directory. Otherwise the tutorial will try to find
# them locally and then download them if needed.
COCO_CAPTIONS_DIR = None

# These settings control how the FiftyOne COCO data loader behaves.
FO_TRAIN_SPLIT = "train"
FO_VAL_SPLIT = "validation"
FO_LABEL_TYPES = []
FO_CLASSES = None
FO_MAX_SAMPLES_TRAIN = None
FO_MAX_SAMPLES_VAL = None
FO_SHUFFLE = True

COCO_CAPTIONS_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_CAPTIONS_ARCHIVE = "annotations_trainval2017.zip"
COCO_CAPTIONS_CACHE_SUBDIR = "deep_lvpm/coco"

# The DLVPM path model is the key modelling assumption in this tutorial.
#
# We connect the image view to every caption view, but we do not directly
# connect captions to each other. This produces a star-shaped path model:
#
# image <-> caption 1
# image <-> caption 2
# image <-> caption 3
# image <-> caption 4
# image <-> caption 5
#
# This is a natural first example for DLVPM because the shared signal should
# represent image semantics that are consistently reflected in the captions.
Path = np.array(
    [
        [0, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ],
    dtype="float32",
)


# -----------------------------------------------------------------------------
# Runtime setup
# -----------------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    print("Using PyTorch MPS backend.")
elif torch.cuda.is_available():
    print("Using PyTorch CUDA backend.")
else:
    print("Using PyTorch CPU backend.")

print(
    "\nThis tutorial will build a six-view DLVPM model:\n"
    "- view 1: the image\n"
    "- views 2-6: five human captions for the same image\n"
)


# -----------------------------------------------------------------------------
# 1. Load MS COCO from FiftyOne Zoo
# -----------------------------------------------------------------------------
#
# The first thing we need is the image data itself. We use FiftyOne's COCO
# integration because it gives us a convenient way to load and inspect the
# official 2017 train and validation splits.
#
print("Step 1/6: loading MS COCO images with FiftyOne...")

train_view = foz.load_zoo_dataset(
    "coco-2017",
    split=FO_TRAIN_SPLIT,
    label_types=FO_LABEL_TYPES,
    classes=FO_CLASSES,
    max_samples=FO_MAX_SAMPLES_TRAIN,
    shuffle=FO_SHUFFLE,
    seed=SEED,
    dataset_name="dlvpm-coco2017-train",
    include_id=True,
)

val_view = foz.load_zoo_dataset(
    "coco-2017",
    split=FO_VAL_SPLIT,
    label_types=FO_LABEL_TYPES,
    classes=FO_CLASSES,
    max_samples=FO_MAX_SAMPLES_VAL,
    shuffle=FO_SHUFFLE,
    seed=SEED,
    dataset_name="dlvpm-coco2017-val",
    include_id=True,
)


# -----------------------------------------------------------------------------
# 2. Load COCO captions and link each image to five captions
# -----------------------------------------------------------------------------
#
# DLVPM works with multiple views of the same underlying sample. In COCO the
# natural shared sample is an image, and the additional views are the human
# captions that describe it.
#
# The next few helper functions do one simple job each:
# - find the official caption annotations,
# - load them into Python,
# - match each image to its five captions.
#
def resolve_coco_caption_annotations(split: str) -> str:
    """Return the local path to the official COCO caption annotations JSON."""
    split_name = "train" if split == FO_TRAIN_SPLIT else "val"
    annotation_name = f"captions_{split_name}2017.json"

    if COCO_CAPTIONS_DIR:
        direct_path = os.path.join(COCO_CAPTIONS_DIR, annotation_name)
        nested_path = os.path.join(COCO_CAPTIONS_DIR, "annotations", annotation_name)
        for candidate in (direct_path, nested_path):
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            "COCO_CAPTIONS_DIR was set, but the caption annotations were not found at "
            f"{direct_path} or {nested_path}."
        )

    candidate_paths: list[str] = []
    dataset_zoo_dir = getattr(getattr(fo, "config", None), "dataset_zoo_dir", None)
    if dataset_zoo_dir:
        candidate_paths.append(
            os.path.join(os.path.expanduser(dataset_zoo_dir), "coco-2017", "raw", annotation_name)
        )
    candidate_paths.append(
        os.path.join(os.path.expanduser("~/fiftyone"), "coco-2017", "raw", annotation_name)
    )

    for candidate in candidate_paths:
        if os.path.exists(candidate):
            print(f"Using local COCO caption annotations: {candidate}")
            return candidate

    try:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", COCO_CAPTIONS_CACHE_SUBDIR)
        os.makedirs(cache_dir, exist_ok=True)
        archive_path = os.path.join(cache_dir, COCO_CAPTIONS_ARCHIVE)
        if not os.path.exists(archive_path):
            print(f"Downloading COCO caption annotations to {archive_path}...")
            urllib.request.urlretrieve(COCO_CAPTIONS_URL, archive_path)
    except Exception as exc:
        searched = ", ".join(candidate_paths) if candidate_paths else "no local paths"
        raise RuntimeError(
            "Could not locate COCO caption annotations locally and automatic download "
            "failed. Set COCO_CAPTIONS_DIR to a directory containing "
            f"{annotation_name}, or place the file in one of: {searched}."
        ) from exc

    extract_dir = os.path.join(os.path.dirname(archive_path), "annotations_trainval2017")
    annotation_path = os.path.join(extract_dir, "annotations", annotation_name)

    if not os.path.exists(annotation_path):
        print(f"Extracting COCO caption annotations to {extract_dir}...")
        with zipfile.ZipFile(archive_path, "r") as zip_file:
            zip_file.extractall(extract_dir)

    return annotation_path


def load_coco_captions(annotation_path: str) -> dict[int, list[str]]:
    """Load official COCO captions keyed by image id."""
    with open(annotation_path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    captions_by_image_id: dict[int, list[str]] = defaultdict(list)
    annotations = sorted(
        payload.get("annotations", []),
        key=lambda annotation: (
            int(annotation.get("image_id", -1)),
            int(annotation.get("id", -1)),
        ),
    )
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        caption = str(annotation.get("caption", "")).strip()
        if caption:
            captions_by_image_id[image_id].append(caption)

    return dict(captions_by_image_id)


def extract_coco_image_id(sample: "fo.Sample") -> int | None:
    """Resolve a COCO image id from a FiftyOne sample."""
    try:
        coco_id = sample.get_field("coco_id")
    except Exception:
        coco_id = None

    if coco_id is not None:
        return int(coco_id)

    basename = os.path.splitext(os.path.basename(sample.filepath))[0]
    if basename.isdigit():
        return int(basename)

    return None


def coco_view_to_examples(
    dataset: "fo.Dataset",
    captions_by_image_id: dict[int, list[str]],
) -> tuple[list[str], list[list[str]]]:
    """Return image paths and five human captions per image."""
    image_paths: list[str] = []
    caption_sets: list[list[str]] = []
    skipped_missing_ids = 0
    skipped_missing_captions = 0
    skipped_short_caption_sets = 0
    truncated_caption_sets = 0

    for sample in dataset:
        image_id = extract_coco_image_id(sample)
        if image_id is None:
            skipped_missing_ids += 1
            continue

        captions = captions_by_image_id.get(image_id)
        if not captions:
            skipped_missing_captions += 1
            continue
        if len(captions) < NUM_CAPTION_VIEWS:
            skipped_short_caption_sets += 1
            continue
        if len(captions) > NUM_CAPTION_VIEWS:
            truncated_caption_sets += 1

        image_paths.append(sample.filepath)
        caption_sets.append(captions[:NUM_CAPTION_VIEWS])

    if skipped_missing_ids:
        print(f"Skipped {skipped_missing_ids} samples without a COCO image id.")
    if skipped_missing_captions:
        print(f"Skipped {skipped_missing_captions} samples without caption annotations.")
    if skipped_short_caption_sets:
        print(
            f"Skipped {skipped_short_caption_sets} samples with fewer than "
            f"{NUM_CAPTION_VIEWS} captions."
        )
    if truncated_caption_sets:
        print(
            f"Trimmed caption sets to the first {NUM_CAPTION_VIEWS} captions for "
            f"{truncated_caption_sets} samples."
        )

    return image_paths, caption_sets


train_caption_annotations = load_coco_captions(
    resolve_coco_caption_annotations(FO_TRAIN_SPLIT)
)
val_caption_annotations = load_coco_captions(
    resolve_coco_caption_annotations(FO_VAL_SPLIT)
)

train_paths_all, train_caption_sets_all = coco_view_to_examples(
    train_view,
    train_caption_annotations,
)
val_paths, val_caption_sets = coco_view_to_examples(
    val_view,
    val_caption_annotations,
)

if len(train_paths_all) == 0:
    raise RuntimeError("No training samples with five COCO captions were found.")
if len(val_paths) == 0:
    raise RuntimeError("No validation samples with five COCO captions were found.")
if len(train_paths_all) < 2:
    raise RuntimeError(
        "Need at least two training samples to create train/test splits."
    )

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(train_paths_all))
num_test = max(1, int(TEST_FRACTION * len(train_paths_all)))
num_test = min(num_test, len(train_paths_all) - 1)
test_idx = perm[:num_test]
train_idx = perm[num_test:]

train_paths = [train_paths_all[i] for i in train_idx]
train_caption_sets = [train_caption_sets_all[i] for i in train_idx]
test_paths = [train_paths_all[i] for i in test_idx]
test_caption_sets = [train_caption_sets_all[i] for i in test_idx]

print(f"Train samples: {len(train_paths)}")
print(f"Validation samples: {len(val_paths)}")
print(f"Test samples (held-out train subset): {len(test_paths)}")
print(f"Captions per image: {NUM_CAPTION_VIEWS}")
print(
    "\nAt this point, each training example consists of:\n"
    "- one image filepath\n"
    f"- {NUM_CAPTION_VIEWS} human captions\n"
)


# -----------------------------------------------------------------------------
# 3. Tokenize captions with a pretrained Hugging Face tokenizer
# -----------------------------------------------------------------------------
#
# The language views need to be converted into tensors before they can be
# passed into the text encoder. We use a pretrained DistilBERT tokenizer so
# that the tokenization scheme matches the text encoder used later.
#
RESAMPLE_BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, use_fast=True)


def tokenize_caption_sets(caption_sets: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize the five captions per image once up front."""
    flat_captions = np.asarray(caption_sets, dtype=object).reshape(-1).tolist()
    encoded = text_tokenizer(
        flat_captions,
        padding="max_length",
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="np",
    )
    token_ids = encoded["input_ids"].astype("int32").reshape(-1, NUM_CAPTION_VIEWS, MAX_TOKENS)
    attention_mask = encoded["attention_mask"].astype("int32").reshape(
        -1,
        NUM_CAPTION_VIEWS,
        MAX_TOKENS,
    )
    return token_ids, attention_mask


# -----------------------------------------------------------------------------
# 4. Torch-backed measurement models
# -----------------------------------------------------------------------------
#
# DLVPM itself learns the shared latent variables, but each data view still
# needs its own measurement model.
#
# In this tutorial we use:
# - an EfficientNet image encoder for the image view
# - a DistilBERT encoder for each caption view
#
class TextEncoderModule(nn.Module):
    """Caption encoder using a fully trainable pretrained DistilBERT backbone."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(TEXT_DROPOUT)
        self.projection = nn.Linear(self.backbone.config.hidden_size, NDIMS)
        self.relu = nn.ReLU()
        self.n_inputs = 2

    def _masked_mean_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        masked_hidden = last_hidden_state * mask
        summed = masked_hidden.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(self, inputs) -> torch.Tensor:
        input_ids, attention_mask = inputs
        outputs = self.backbone(
            input_ids=input_ids.long().contiguous(),
            attention_mask=attention_mask.long().contiguous(),
        )
        pooled = self._masked_mean_pool(
            outputs.last_hidden_state.contiguous(),
            attention_mask.long().contiguous(),
        )
        pooled = self.dropout(pooled)
        return self.relu(self.projection(pooled))


class ImageEncoderModule(nn.Module):
    """EfficientNetB0 image encoder matching the original tutorial architecture."""

    def __init__(self) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.image_base = efficientnet_b0(weights=weights)
        in_features = self.image_base.classifier[1].in_features
        self.image_base.classifier = nn.Sequential(nn.Linear(in_features, NDIMS), nn.ReLU())
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(1, 3, 1, 1),
        )
        self.n_inputs = 1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x_image = inputs.float()
        x_image = (x_image - self.image_mean) / self.image_std
        return self.image_base(x_image)


def build_image_encoder(NDIMS) -> nn.Module:
    del NDIMS
    return ImageEncoderModule()


def build_text_encoder(NDIMS) -> nn.Module:
    """DistilBERT text encoder replacing the original scratch transformer."""
    del NDIMS
    return TextEncoderModule(model_name=TEXT_MODEL_NAME)


def build_model_list(NDIMS) -> list[nn.Module]:
    """Build one image encoder and shared caption encoder views."""
    image_model = build_image_encoder(NDIMS)
    caption_model = build_text_encoder(NDIMS)
    return [image_model] + [caption_model for _ in range(NUM_CAPTION_VIEWS)]


# -----------------------------------------------------------------------------
# 5. Torch dataloaders
# -----------------------------------------------------------------------------
#
# The model expects a batch with six views. The dataloader therefore returns:
# - one tensor for images
# - five pairs of tensors for the five caption views
#
# This looks a little verbose, but it keeps the data flow explicit. A new user
# can print one batch here and see exactly what is being passed to the model.
#
class CocoRetrievalDataset(Dataset):
    """Load COCO images and five tokenized captions per sample."""

    def __init__(
        self,
        image_paths: list[str],
        caption_token_ids: np.ndarray,
        caption_attention_mask: np.ndarray,
        training: bool,
    ) -> None:
        self.image_paths = image_paths
        self.caption_token_ids = caption_token_ids
        self.caption_attention_mask = caption_attention_mask
        self.training = training

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, image_path: str) -> torch.Tensor:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize((IMG_SIZE, IMG_SIZE), resample=RESAMPLE_BICUBIC)
            if self.training and np.random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image_arr = np.asarray(image, dtype="float32").copy() / 255.0
            image_arr = np.transpose(image_arr, (2, 0, 1))
        return torch.from_numpy(image_arr)

    def __getitem__(self, index: int):
        image = self._load_image(self.image_paths[index])
        caption_token_views = self.caption_token_ids[index]
        caption_mask_views = self.caption_attention_mask[index]
        return (
            image,
            torch.from_numpy(caption_token_views[0].copy()).long(),
            torch.from_numpy(caption_mask_views[0].copy()).long(),
            torch.from_numpy(caption_token_views[1].copy()).long(),
            torch.from_numpy(caption_mask_views[1].copy()).long(),
            torch.from_numpy(caption_token_views[2].copy()).long(),
            torch.from_numpy(caption_mask_views[2].copy()).long(),
            torch.from_numpy(caption_token_views[3].copy()).long(),
            torch.from_numpy(caption_mask_views[3].copy()).long(),
            torch.from_numpy(caption_token_views[4].copy()).long(),
            torch.from_numpy(caption_mask_views[4].copy()).long(),
        )


def collate_multiview(batch):
    """Return batches as a single x-structure, matching the custom train_step API."""
    views = list(zip(*batch))
    collated_views = tuple(torch.stack(list(view), dim=0) for view in views)
    return (collated_views,)


def make_multiview_loader(
    image_paths: list[str],
    caption_sets: list[list[str]],
    training: bool,
) -> DataLoader:
    caption_token_ids, caption_attention_mask = tokenize_caption_sets(caption_sets)
    dataset = CocoRetrievalDataset(
        image_paths=image_paths,
        caption_token_ids=caption_token_ids,
        caption_attention_mask=caption_attention_mask,
        training=training,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=training,
        drop_last=training,
        num_workers=NUM_WORKERS,
        collate_fn=collate_multiview,
    )


# -----------------------------------------------------------------------------
# 6. Benchmark task: DLVPM vs CLIP vs VICReg on image-text retrieval
# -----------------------------------------------------------------------------
#
# After training, we evaluate the learned representation using a retrieval task.
# For each image we ask whether its matching caption set appears near the top of
# the ranked list, and vice versa. This is a simple way to test whether the
# latent space really links the image and language views.
#
def l2_normalize(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, eps)


def collect_image_text_embeddings(
    model: nn.Module,
    dataset: DataLoader,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect image embeddings and all five caption embeddings from a dataset."""
    image_batches: list[np.ndarray] = []
    text_batches: list[np.ndarray] = []
    total = 0

    for batch in dataset:
        views = model._batch_to_inputs(batch, device=model.device)
        views_nested = model.organize_inputs_by_model(views)
        model.eval()
        with torch.no_grad():
            image_embedding = model.model_list[0](views_nested[0])
        image_np = image_embedding.detach().cpu().numpy()

        text_view_embeddings: list[np.ndarray] = []
        for text_view_index in range(1, NUM_CAPTION_VIEWS + 1):
            with torch.no_grad():
                text_embedding = model.model_list[text_view_index](views_nested[text_view_index])
            text_np = text_embedding.detach().cpu().numpy()
            text_view_embeddings.append(text_np)
        text_batch = np.stack(text_view_embeddings, axis=1)

        image_batches.append(image_np)
        text_batches.append(text_batch)
        total += image_np.shape[0]

        if total >= max_samples:
            break

    if not image_batches:
        raise RuntimeError("No embeddings were collected for retrieval benchmark.")

    image_all = np.concatenate(image_batches, axis=0)[:max_samples]
    text_all = np.concatenate(text_batches, axis=0)[:max_samples]
    return image_all, text_all


def aggregate_caption_groups(text_embeddings: np.ndarray) -> np.ndarray:
    """Aggregate five caption embeddings into one normalized group embedding."""
    if text_embeddings.ndim != 3 or text_embeddings.shape[1] != NUM_CAPTION_VIEWS:
        raise ValueError(
            "Expected text embeddings with shape "
            f"(num_images, {NUM_CAPTION_VIEWS}, embedding_dim)."
        )

    num_images, _, embedding_dim = text_embeddings.shape
    flat_text = text_embeddings.reshape(num_images * NUM_CAPTION_VIEWS, embedding_dim)
    flat_text = l2_normalize(flat_text.astype("float32"))
    caption_embeddings = flat_text.reshape(num_images, NUM_CAPTION_VIEWS, embedding_dim)
    group_embeddings = np.mean(caption_embeddings, axis=1)
    return l2_normalize(group_embeddings)


def retrieval_metrics(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    ks: tuple[int, ...] = RETRIEVAL_KS,
) -> dict[str, float]:
    i2g_rank, g2i_rank = retrieval_rank_arrays(image_embeddings, text_embeddings)
    return retrieval_metrics_from_ranks(i2g_rank, g2i_rank, ks=ks)


def retrieval_rank_arrays(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-sample ranks for image <-> caption-set retrieval."""
    image_embeddings = l2_normalize(image_embeddings.astype("float32"))
    group_embeddings = aggregate_caption_groups(text_embeddings.astype("float32"))

    num_images = image_embeddings.shape[0]
    group_similarity = image_embeddings @ group_embeddings.T
    target_index = np.arange(num_images)

    i2g_order = np.argsort(-group_similarity, axis=1)
    i2g_rank = np.argmax(i2g_order == target_index[:, None], axis=1) + 1

    g2i_order = np.argsort(-group_similarity.T, axis=1)
    g2i_rank = np.argmax(g2i_order == target_index[:, None], axis=1) + 1

    return i2g_rank, g2i_rank


def retrieval_metrics_from_ranks(
    i2g_rank: np.ndarray,
    g2i_rank: np.ndarray,
    ks: tuple[int, ...] = RETRIEVAL_KS,
) -> dict[str, float]:
    """Compute summary metrics from precomputed retrieval ranks."""

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"i2g_top{k}"] = float(np.mean(i2g_rank <= k))
        metrics[f"g2i_top{k}"] = float(np.mean(g2i_rank <= k))

    metrics["i2g_median_rank"] = float(np.median(i2g_rank))
    metrics["g2i_median_rank"] = float(np.median(g2i_rank))
    return metrics


def print_retrieval_row(method_name: str, metrics: dict[str, float]) -> None:
    row = [method_name]
    for k in RETRIEVAL_KS:
        row.append(f"{metrics[f'i2g_top{k}']:.3f}")
        row.append(f"{metrics[f'g2i_top{k}']:.3f}")
    row.append(f"{metrics['i2g_median_rank']:.1f}")
    row.append(f"{metrics['g2i_median_rank']:.1f}")
    print(" | ".join(row))


def evaluate_retrieval_result(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    i2g_rank, g2i_rank = retrieval_rank_arrays(image_embeddings, text_embeddings)
    metrics = retrieval_metrics_from_ranks(i2g_rank, g2i_rank)
    rank_data = {
        "i2g_rank": i2g_rank,
        "g2i_rank": g2i_rank,
    }
    return metrics, rank_data


def bootstrap_combined_rank_ci(
    i2g_rank: np.ndarray,
    g2i_rank: np.ndarray,
    n_bootstrap: int = RANK_BOOTSTRAP_SAMPLES,
    seed: int = SEED,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    num_samples = len(i2g_rank)
    bootstrap_scores = np.empty(n_bootstrap, dtype="float64")

    for bootstrap_index in range(n_bootstrap):
        sample_index = rng.integers(0, num_samples, size=num_samples)
        bootstrap_scores[bootstrap_index] = 0.5 * (
            np.mean(i2g_rank[sample_index]) + np.mean(g2i_rank[sample_index])
        )

    point_estimate = 0.5 * (float(np.mean(i2g_rank)) + float(np.mean(g2i_rank)))
    lower = float(np.percentile(bootstrap_scores, 2.5))
    upper = float(np.percentile(bootstrap_scores, 97.5))
    return point_estimate, lower, upper


def plot_rank_benchmark(
    method_names: list[str],
    rank_results: dict[str, dict[str, np.ndarray]],
) -> None:
    summary_rows = []
    for method_name in method_names:
        rank_data = rank_results[method_name]
        point_estimate, lower, upper = bootstrap_combined_rank_ci(
            rank_data["i2g_rank"],
            rank_data["g2i_rank"],
        )
        summary_rows.append((method_name, point_estimate, lower, upper))

    summary_rows.sort(key=lambda row: row[1])
    sorted_methods = [row[0] for row in summary_rows]
    sorted_points = np.array([row[1] for row in summary_rows], dtype="float64")
    lower_errors = sorted_points - np.array([row[2] for row in summary_rows], dtype="float64")
    upper_errors = np.array([row[3] for row in summary_rows], dtype="float64") - sorted_points

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_positions = np.arange(len(sorted_methods))
    ax.barh(
        y_positions,
        sorted_points,
        xerr=np.vstack([lower_errors, upper_errors]),
        color="#8fbcd4",
        edgecolor="black",
        capsize=4,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_methods)
    ax.invert_yaxis()
    ax.set_xlabel("Bootstrapped Mean Retrieval Rank (lower is better)")
    ax.set_title("COCO Retrieval Benchmark by Method")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    plt.show()


print("\nStep 6/6: training DLVPM and evaluating image-text retrieval...")

benchmark_train_n = min(BENCHMARK_TRAIN_SAMPLES, len(train_paths))
benchmark_val_n = min(BENCHMARK_VAL_SAMPLES, len(val_paths))
benchmark_test_n = min(BENCHMARK_SAMPLES, len(test_paths))

benchmark_train_paths = train_paths[:benchmark_train_n]
benchmark_train_caption_sets = train_caption_sets[:benchmark_train_n]
benchmark_val_paths = val_paths[:benchmark_val_n]
benchmark_val_caption_sets = val_caption_sets[:benchmark_val_n]
benchmark_test_paths = test_paths[:benchmark_test_n]
benchmark_test_caption_sets = test_caption_sets[:benchmark_test_n]

benchmark_train_ds = make_multiview_loader(
    benchmark_train_paths,
    benchmark_train_caption_sets,
    training=True,
)
benchmark_val_ds = make_multiview_loader(
    benchmark_val_paths,
    benchmark_val_caption_sets,
    training=False,
)
benchmark_test_ds = make_multiview_loader(
    benchmark_test_paths,
    benchmark_test_caption_sets,
    training=False,
)

print(
    "Benchmark dataset sizes - "
    f"train: {benchmark_train_n}, val: {benchmark_val_n}, test: {benchmark_test_n} "
    f"(captions per image: {NUM_CAPTION_VIEWS})"
)
if RUN_BASELINES:
    print(f"Training each method for {BENCHMARK_EPOCHS} epochs with identical encoders.")
else:
    print(f"Training DLVPM only for {BENCHMARK_EPOCHS} epochs.")

benchmark_results: dict[str, dict[str, float]] = {}
benchmark_rank_results: dict[str, dict[str, np.ndarray]] = {}
method_order = ["DLVPM"]
steps_per_epoch = max(1, benchmark_train_n // BATCH_SIZE)
warmup_steps = max(1, steps_per_epoch * LEARNING_RATE_WARMUP_EPOCHS)


def make_optimizer_list(model: nn.Module) -> list[torch.optim.Optimizer]:
    return [torch.optim.Adam(view_model.parameters(), lr=LEARNING_RATE_START) for view_model in model.model_list]


def make_scheduler_list(optimizers: list[torch.optim.Optimizer]) -> list[torch.optim.lr_scheduler.LambdaLR]:
    def schedule(epoch: int) -> float:
        if epoch < LEARNING_RATE_WARMUP_EPOCHS:
            return 1.0
        return LEARNING_RATE_END / LEARNING_RATE_START

    return [torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule) for optimizer in optimizers]

# We now construct the actual DLVPM model.
#
# Notice how the key ingredients are exactly the same as in the other tutorials:
# - `Path` defines which views should be associated
# - `model_list` defines the measurement model for each view
# - `regularizer_list` controls the final projection regularization
# - `tot_num` is the total number of training samples
# - `ndims` is the number of latent variables to learn
#
# Here we also use:
# - `order=True` to order the latent variables by shared structure
# - `order_association_cutoff=0.99` to discard the final ordered dimensions
#   once the cumulative shared association mass has effectively saturated
print("Training DLVPM benchmark model...")
dlvpm_benchmark_models = build_model_list(NDIMS)
dlvpm_benchmark = StructuralModel(
    Path=Path,
    model_list=dlvpm_benchmark_models,
    regularizer_list=[None for _ in dlvpm_benchmark_models],
    tot_num=benchmark_train_n,
    ndims=NDIMS,
    orthogonalization="zca",
    diag_offset=1e-6,
    train_DLV=True,
    momentum=0.95,
    order=True,
    order_association_cutoff=0.99,
)
dlvpm_optimizers = make_optimizer_list(dlvpm_benchmark)
dlvpm_schedulers = make_scheduler_list(dlvpm_optimizers)
dlvpm_benchmark.compile(dlvpm_optimizers)
dlvpm_benchmark.fit(
    benchmark_train_ds,
    validation_data=benchmark_val_ds,
    epochs=BENCHMARK_EPOCHS,
    verbose=True,
    schedulers=dlvpm_schedulers,
)
dlvpm_img, dlvpm_txt = collect_image_text_embeddings(
    dlvpm_benchmark,
    benchmark_test_ds,
    max_samples=benchmark_test_n,
)
benchmark_results["DLVPM"], benchmark_rank_results["DLVPM"] = evaluate_retrieval_result(
    dlvpm_img,
    dlvpm_txt,
)

# Intermediate correlation plotting is disabled so the benchmark can run through cleanly.

if RUN_BASELINES:
    # CLIP baseline
    print("Training CLIP baseline...")
    clip_model_list = build_model_list(NDIMS)
    clip_model = CLIP(
        model_list=clip_model_list,
        regularizer_list=[None for _ in clip_model_list],
        ndims=NDIMS,
        is_siamese=False,
    )
    clip_optimizers = make_optimizer_list(clip_model)
    clip_model.compile(clip_optimizers)
    clip_model.fit(
        benchmark_train_ds,
        validation_data=benchmark_val_ds,
        epochs=BENCHMARK_EPOCHS,
        verbose=True,
    )
    clip_img, clip_txt = collect_image_text_embeddings(
        clip_model,
        benchmark_test_ds,
        max_samples=benchmark_test_n,
    )
    benchmark_results["CLIP"], benchmark_rank_results["CLIP"] = evaluate_retrieval_result(
        clip_img,
        clip_txt,
    )
    method_order.append("CLIP")

    # VICReg baseline
    print("Training VICReg baseline...")
    vic_model_list = build_model_list(NDIMS)
    vic_model = VICReg(
        model_list=vic_model_list,
        regularizer_list=[None for _ in vic_model_list],
        ndims=NDIMS,
        is_siamese=False,
    )
    vic_optimizers = make_optimizer_list(vic_model)
    vic_model.compile(vic_optimizers)
    vic_model.fit(
        benchmark_train_ds,
        validation_data=benchmark_val_ds,
        epochs=BENCHMARK_EPOCHS,
        verbose=True,
    )
    vic_img, vic_txt = collect_image_text_embeddings(
        vic_model,
        benchmark_test_ds,
        max_samples=benchmark_test_n,
    )
    benchmark_results["VICReg"], benchmark_rank_results["VICReg"] = evaluate_retrieval_result(
        vic_img,
        vic_txt,
    )
    method_order.append("VICReg")

    print("Training LeJEPA baseline...")
    lejepa_model_list = build_model_list(NDIMS)
    lejepa_model = LeJEPA(
        model_list=lejepa_model_list,
        regularizer_list=[None for _ in lejepa_model_list],
        ndims=NDIMS,
        is_siamese=False,
    )
    lejepa_optimizers = make_optimizer_list(lejepa_model)
    lejepa_model.compile(lejepa_optimizers)
    lejepa_model.fit(
        benchmark_train_ds,
        validation_data=benchmark_val_ds,
        epochs=BENCHMARK_EPOCHS,
        verbose=True,
    )
    lejepa_img, lejepa_txt = collect_image_text_embeddings(
        lejepa_model,
        benchmark_test_ds,
        max_samples=benchmark_test_n,
    )
    benchmark_results["LeJEPA"], benchmark_rank_results["LeJEPA"] = evaluate_retrieval_result(
        lejepa_img,
        lejepa_txt,
    )
    method_order.append("LeJEPA")

print(
    "\nFinal retrieval summary:\n"
    "- higher Top-K values are better\n"
    "- lower median rank values are better\n"
)
header = ["Method"]
for k in RETRIEVAL_KS:
    header.extend([f"i2g_top{k}", f"g2i_top{k}"])
header.extend(["i2g_med_rank", "g2i_med_rank"])
print(" | ".join(header))
print("-" * 100)
for method in method_order:
    print_retrieval_row(method, benchmark_results[method])

plot_rank_benchmark(method_order, benchmark_rank_results)
