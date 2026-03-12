#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import zipfile
from collections import defaultdict

# Keras 3 defaults to the JAX backend unless selected explicitly.
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import layers

try:
    import fiftyone as fo
    import fiftyone.zoo as foz
except Exception as exc:
    raise RuntimeError(
        "This tutorial requires FiftyOne. Install it with: pip install fiftyone"
    ) from exc

from deep_lvpm.model import StructuralModel
from deep_lvpm.multi_model import CLIP, VICReg


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
NUM_CAPTION_VIEWS = 5
IMG_SIZE = 224
MAX_TOKENS = 32
VOCAB_SIZE = 30000
EMBED_DIM = 256
TRANSFORMER_HEADS = 4
TRANSFORMER_FF_DIM = 512
TRANSFORMER_DROPOUT = 0.1

NDIMS = 512
BATCH_SIZE = 256
LEARNING_RATE = 1e-5
BENCHMARK_EPOCHS = 10
BENCHMARK_TRAIN_SAMPLES = 20000
BENCHMARK_VAL_SAMPLES = 5000
BENCHMARK_SAMPLES = 2048
RETRIEVAL_KS = (1, 5, 10)

N_VIEWS = NUM_CAPTION_VIEWS + 1
Path = tf.ones((N_VIEWS, N_VIEWS), dtype=tf.float32) - tf.eye(N_VIEWS, dtype=tf.float32)

TEST_FRACTION = 0.10  # COCO test split has no captions, so we hold out part of train
SEED = 51
AUTOTUNE = tf.data.AUTOTUNE

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
COCO_CAPTIONS_DIR = os.environ.get("COCO_CAPTIONS_DIR")


# -----------------------------------------------------------------------------
# Runtime setup
# -----------------------------------------------------------------------------
keras.utils.set_random_seed(SEED)
tf.config.run_functions_eagerly(False)

for device in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(device, True)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# 1. Load MS COCO from FiftyOne Zoo
# -----------------------------------------------------------------------------
print("Loading MS COCO train and validation splits with FiftyOne...")

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
        archive_path = keras.utils.get_file(
            fname=COCO_CAPTIONS_ARCHIVE,
            origin=COCO_CAPTIONS_URL,
            cache_subdir=COCO_CAPTIONS_CACHE_SUBDIR,
        )
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


# -----------------------------------------------------------------------------
# 3. Vectorize captions (shared tokenizer for all text views)
# -----------------------------------------------------------------------------
text_vectorizer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_TOKENS,
    standardize="lower_and_strip_punctuation",
)

train_captions_flat = np.asarray(train_caption_sets, dtype=str).reshape(-1)
caption_ds = tf.data.Dataset.from_tensor_slices(train_captions_flat)
caption_ds = caption_ds.shuffle(len(train_captions_flat), seed=SEED)
caption_ds = caption_ds.batch(1024).prefetch(AUTOTUNE)
text_vectorizer.adapt(caption_ds)


# -----------------------------------------------------------------------------
# 4. Define measurement models
# -----------------------------------------------------------------------------
def build_image_encoder() -> keras.Model:
    """EfficientNetB0 image encoder."""
    image_inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    image_base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    image_base.trainable = True

    x_image = keras.applications.efficientnet.preprocess_input(image_inputs)
    x_image = image_base(x_image, training=False)
    image_outputs = layers.Dense(128, activation="relu", name="image_projection")(x_image)
    return keras.Model(image_inputs, image_outputs, name="coco_efficientnetb0")


def transformer_encoder_block(
    inputs: tf.Tensor,
    num_heads: int,
    ff_dim: int,
    dropout_rate: float,
) -> tf.Tensor:
    attn_input = layers.LayerNormalization(epsilon=1e-6)(inputs)
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=EMBED_DIM // num_heads,
        dropout=dropout_rate,
    )(attn_input, attn_input)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    x = layers.Add()([inputs, attn_output])

    ff_input = layers.LayerNormalization(epsilon=1e-6)(x)
    ff_output = layers.Dense(ff_dim, activation="relu")(ff_input)
    ff_output = layers.Dropout(dropout_rate)(ff_output)
    ff_output = layers.Dense(EMBED_DIM)(ff_output)
    return layers.Add()([x, ff_output])


def build_text_encoder() -> keras.Model:
    """Transformer text encoder."""
    text_inputs = keras.Input(shape=(MAX_TOKENS,), dtype="int32", name="caption_tokens")
    token_embeddings = layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBED_DIM,
        mask_zero=True,
        name="token_embedding",
    )(text_inputs)

    position_indices = tf.range(start=0, limit=MAX_TOKENS, delta=1)
    position_embeddings = layers.Embedding(
        input_dim=MAX_TOKENS,
        output_dim=EMBED_DIM,
        name="position_embedding",
    )(position_indices)

    x_text = token_embeddings + position_embeddings
    x_text = transformer_encoder_block(
        x_text,
        num_heads=TRANSFORMER_HEADS,
        ff_dim=TRANSFORMER_FF_DIM,
        dropout_rate=TRANSFORMER_DROPOUT,
    )
    x_text = layers.GlobalAveragePooling1D()(x_text)
    text_outputs = layers.Dense(128, activation="relu", name="text_projection")(x_text)
    return keras.Model(text_inputs, text_outputs, name="coco_caption_transformer")


def build_model_list() -> list[keras.Model]:
    """Build one image encoder and shared caption encoder views."""
    image_model = build_image_encoder()
    caption_model = build_text_encoder()
    return [image_model] + [caption_model for _ in range(NUM_CAPTION_VIEWS)]


# -----------------------------------------------------------------------------
# 5. Create tf.data pipelines
# -----------------------------------------------------------------------------
def make_multiview_dataset(
    image_paths: list[str],
    caption_sets: list[list[str]],
    training: bool,
) -> tf.data.Dataset:
    caption_array = np.asarray(caption_sets, dtype=str)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, caption_array))

    if training:
        dataset = dataset.shuffle(len(image_paths), seed=SEED, reshuffle_each_iteration=True)

    def map_example(image_path: tf.Tensor, captions: tf.Tensor):
        image_bytes = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = tf.cast(image, tf.float32)

        caption_tokens = tf.cast(text_vectorizer(captions), tf.int32)

        views = [image]
        for i in range(NUM_CAPTION_VIEWS):
            views.append(caption_tokens[i])

        # A 1-tuple ensures Keras treats this as a single "x" structure.
        return (tuple(views),)

    dataset = dataset.map(map_example, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=training)
    return dataset.prefetch(AUTOTUNE)


# -----------------------------------------------------------------------------
# 6. Benchmark task: DLVPM vs CLIP vs VICReg on image-text retrieval
# -----------------------------------------------------------------------------
def l2_normalize(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, eps)


def collect_image_text_embeddings(
    model: keras.Model,
    dataset: tf.data.Dataset,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect image embeddings and all five caption embeddings from a dataset."""
    image_batches: list[np.ndarray] = []
    text_batches: list[np.ndarray] = []
    total = 0

    for batch in dataset:
        views = batch[0]
        image_tensor = views[0]
        image_embedding = model.model_list[0](image_tensor, training=False)
        image_np = np.asarray(keras.ops.convert_to_numpy(image_embedding))

        text_view_embeddings: list[np.ndarray] = []
        for text_view_index in range(1, NUM_CAPTION_VIEWS + 1):
            text_embedding = model.model_list[text_view_index](
                views[text_view_index],
                training=False,
            )
            text_np = np.asarray(keras.ops.convert_to_numpy(text_embedding))
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


def retrieval_metrics(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    ks: tuple[int, ...] = RETRIEVAL_KS,
) -> dict[str, float]:
    """Compute caption-averaged Recall@K and mean rank on 5-caption COCO retrieval."""
    image_embeddings = l2_normalize(image_embeddings.astype("float32"))
    text_embeddings = l2_normalize(text_embeddings.astype("float32"))

    if text_embeddings.ndim != 3 or text_embeddings.shape[1] != NUM_CAPTION_VIEWS:
        raise ValueError(
            "Expected text embeddings with shape "
            f"(num_images, {NUM_CAPTION_VIEWS}, embedding_dim)."
        )

    num_images = image_embeddings.shape[0]
    flat_text_embeddings = text_embeddings.reshape(num_images * NUM_CAPTION_VIEWS, -1)
    caption_owner = np.repeat(np.arange(num_images), NUM_CAPTION_VIEWS)

    i2t_similarity = image_embeddings @ flat_text_embeddings.T
    i2t_order = np.argsort(-i2t_similarity, axis=1)
    i2t_positive_mask = caption_owner[i2t_order] == np.arange(num_images)[:, None]
    i2t_positive_rank = np.where(i2t_positive_mask)[1].reshape(num_images, NUM_CAPTION_VIEWS) + 1

    t2i_similarity = flat_text_embeddings @ image_embeddings.T
    t2i_order = np.argsort(-t2i_similarity, axis=1)
    t2i_rank = np.argmax(t2i_order == caption_owner[:, None], axis=1) + 1
    t2i_rank = t2i_rank.reshape(num_images, NUM_CAPTION_VIEWS)

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"i2t_R@{k}"] = float(np.mean(np.mean(i2t_positive_rank <= k, axis=1)))
        metrics[f"t2i_R@{k}"] = float(np.mean(np.mean(t2i_rank <= k, axis=1)))

    metrics["i2t_median_rank"] = float(np.median(np.mean(i2t_positive_rank, axis=1)))
    metrics["t2i_median_rank"] = float(np.median(np.mean(t2i_rank, axis=1)))
    return metrics


def print_retrieval_row(method_name: str, metrics: dict[str, float]) -> None:
    row = [method_name]
    for k in RETRIEVAL_KS:
        row.append(f"{metrics[f'i2t_R@{k}']:.3f}")
        row.append(f"{metrics[f't2i_R@{k}']:.3f}")
    row.append(f"{metrics['i2t_median_rank']:.1f}")
    row.append(f"{metrics['t2i_median_rank']:.1f}")
    print(" | ".join(row))


print("\nRunning retrieval benchmark on held-out COCO image/caption groups...")

benchmark_train_n = min(BENCHMARK_TRAIN_SAMPLES, len(train_paths))
benchmark_val_n = min(BENCHMARK_VAL_SAMPLES, len(val_paths))
benchmark_test_n = min(BENCHMARK_SAMPLES, len(test_paths))

benchmark_train_paths = train_paths[:benchmark_train_n]
benchmark_train_caption_sets = train_caption_sets[:benchmark_train_n]
benchmark_val_paths = val_paths[:benchmark_val_n]
benchmark_val_caption_sets = val_caption_sets[:benchmark_val_n]
benchmark_test_paths = test_paths[:benchmark_test_n]
benchmark_test_caption_sets = test_caption_sets[:benchmark_test_n]

benchmark_train_ds = make_multiview_dataset(
    benchmark_train_paths,
    benchmark_train_caption_sets,
    training=True,
)
benchmark_val_ds = make_multiview_dataset(
    benchmark_val_paths,
    benchmark_val_caption_sets,
    training=False,
)
benchmark_test_ds = make_multiview_dataset(
    benchmark_test_paths,
    benchmark_test_caption_sets,
    training=False,
)

print(
    "Benchmark dataset sizes - "
    f"train: {benchmark_train_n}, val: {benchmark_val_n}, test: {benchmark_test_n} "
    f"(captions per image: {NUM_CAPTION_VIEWS})"
)
print(f"Training each method for {BENCHMARK_EPOCHS} epochs with identical encoders.")

benchmark_results: dict[str, dict[str, float]] = {}

# DLVPM baseline
print("Training DLVPM benchmark model...")
dlvpm_benchmark_models = build_model_list()
dlvpm_benchmark = StructuralModel(
    Path=Path,
    model_list=dlvpm_benchmark_models,
    regularizer_list=[None for _ in dlvpm_benchmark_models],
    tot_num=benchmark_train_n,
    ndims=NDIMS,
    orthogonalization="zca",
    diag_offset=1e-6,
    train_DLV=True,
)
dlvpm_optimizers = [
    keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    for _ in dlvpm_benchmark_models
]
dlvpm_benchmark.compile(dlvpm_optimizers)
dlvpm_benchmark.fit(
    benchmark_train_ds,
    validation_data=benchmark_val_ds,
    epochs=BENCHMARK_EPOCHS,
    verbose=True,
)
dlvpm_img, dlvpm_txt = collect_image_text_embeddings(
    dlvpm_benchmark,
    benchmark_test_ds,
    max_samples=benchmark_test_n,
)
benchmark_results["DLVPM"] = retrieval_metrics(dlvpm_img, dlvpm_txt)

# CLIP baseline
print("Training CLIP baseline...")
clip_model_list = build_model_list()
clip_model = CLIP(
    model_list=clip_model_list,
    regularizer_list=[None for _ in clip_model_list],
    ndims=NDIMS,
    is_siamese=False,
)
clip_optimizers = [
    keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    for _ in clip_model_list
]
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
benchmark_results["CLIP"] = retrieval_metrics(clip_img, clip_txt)

# VICReg baseline
print("Training VICReg baseline...")
vic_model_list = build_model_list()
vic_model = VICReg(
    Path=Path,
    model_list=vic_model_list,
    regularizer_list=[None for _ in vic_model_list],
    ndims=NDIMS,
    is_siamese=False,
)
vic_optimizers = [
    keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    for _ in vic_model_list
]
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
benchmark_results["VICReg"] = retrieval_metrics(vic_img, vic_txt)

print(
    "\nRetrieval benchmark (higher Recall@K is better, lower median rank is better):"
)
header = ["Method"]
for k in RETRIEVAL_KS:
    header.extend([f"i2t_R@{k}", f"t2i_R@{k}"])
header.extend(["i2t_med_rank", "t2i_med_rank"])
print(" | ".join(header))
print("-" * 100)
for method in ("DLVPM", "CLIP", "VICReg"):
    print_retrieval_row(method, benchmark_results[method])
