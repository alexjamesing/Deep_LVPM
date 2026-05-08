#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MS COCO image-caption retrieval benchmark on the Keras TensorFlow backend.
"""

import os
import json
import random
import zipfile
from collections import defaultdict

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("USE_TF", "1")
os.environ.setdefault("USE_TORCH", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import tensorflow as tf
except Exception as exc:
    raise RuntimeError(
        "This tutorial uses the Keras TensorFlow backend. Install it with: "
        "python -m pip install -e '.[tf-apple]'"
    ) from exc

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers, ops
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from deep_lvpm.plot import plot_correlation_chord_row

try:
    import fiftyone as fo
    import fiftyone.zoo as foz
except Exception as exc:
    raise RuntimeError(
        "This tutorial requires FiftyOne. Install it with: python -m pip install fiftyone"
    ) from exc

from deep_lvpm.model import SecondOrderStructuralModel
from deep_lvpm.model import StructuralModel
from deep_lvpm.multi_model import CLIP, VICReg, LeJEPA


if keras.backend.backend() != "tensorflow":
    raise RuntimeError(
        "Keras did not start with the TensorFlow backend. Re-run this script with "
        "KERAS_BACKEND=tensorflow."
    )


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


NUM_CAPTION_VIEWS = 5
IMG_SIZE = 224
MAX_TOKENS = 32
VOCAB_SIZE = 30000
EMBED_DIM = 256
TRANSFORMER_HEADS = 4
TRANSFORMER_FF_DIM = 512
NUM_TRANSFORMER_BLOCKS = 2
TEXT_DROPOUT = 0.10
NDIMS = env_int("DLVPM_COCO_NDIMS", 128)
BATCH_SIZE = env_int("DLVPM_COCO_BATCH_SIZE", 512)
ORTHOG_WEIGHT = 1e-2
LEARNING_RATE_START = 1e-3
LEARNING_RATE_END = 1e-4
LEARNING_RATE_WARMUP_EPOCHS = 0
BENCHMARK_EPOCHS = env_int("DLVPM_COCO_EPOCHS", 10)
BENCHMARK_TRAIN_SAMPLES = env_int("DLVPM_COCO_TRAIN_SAMPLES", 20000)
BENCHMARK_VAL_SAMPLES = env_int("DLVPM_COCO_VAL_SAMPLES", 5000)
BENCHMARK_SAMPLES = env_int("DLVPM_COCO_TEST_SAMPLES", 2048)
RUN_BASELINES = False
RUN_CIFAR = True
RETRIEVAL_KS = (1, 5, 10)
RANK_BOOTSTRAP_SAMPLES = env_int("DLVPM_COCO_RANK_BOOTSTRAPS", 1000)
CIFAR_BATCH_SIZE = env_int("DLVPM_COCO_CIFAR_BATCH_SIZE", 256)

N_VIEWS = NUM_CAPTION_VIEWS + 1

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

TEST_FRACTION = 0.10
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


random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

for device in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(device, True)
    except Exception:
        pass

if tf.config.list_physical_devices("GPU"):
    print("Using TensorFlow GPU backend.")
else:
    print("Using TensorFlow CPU backend.")


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


def resolve_coco_caption_annotations(split: str) -> str:
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


train_caption_annotations = load_coco_captions(resolve_coco_caption_annotations(FO_TRAIN_SPLIT))
val_caption_annotations = load_coco_captions(resolve_coco_caption_annotations(FO_VAL_SPLIT))

train_paths_all, train_caption_sets_all = coco_view_to_examples(train_view, train_caption_annotations)
val_paths, val_caption_sets = coco_view_to_examples(val_view, val_caption_annotations)

if len(train_paths_all) == 0:
    raise RuntimeError("No training samples with five COCO captions were found.")
if len(val_paths) == 0:
    raise RuntimeError("No validation samples with five COCO captions were found.")
if len(train_paths_all) < 2:
    raise RuntimeError("Need at least two training samples to create train/test splits.")

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


text_vectorizer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="count",
    pad_to_max_tokens=True,
    standardize="lower_and_strip_punctuation",
)

train_captions_flat = np.asarray(train_caption_sets, dtype=str).reshape(-1)
caption_ds = tf.data.Dataset.from_tensor_slices(train_captions_flat)
caption_ds = caption_ds.shuffle(len(train_captions_flat), seed=SEED)
caption_ds = caption_ds.batch(1024).prefetch(AUTOTUNE)
text_vectorizer.adapt(caption_ds)

def build_image_encoder(ndims: int) -> keras.Model:
    image_inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x_image = layers.Rescaling(1.0 / 255.0, name="image_rescaling")(image_inputs)
    x_image = layers.Conv2D(128, 3, padding="same", activation="relu", name="conv1")(x_image)
    x_image = layers.MaxPooling2D(pool_size=2, name="pool1")(x_image)
    x_image = layers.Conv2D(256, 3, padding="same", activation="relu", name="conv2")(x_image)
    x_image = layers.MaxPooling2D(pool_size=2, name="pool2")(x_image)
    x_image = layers.Conv2D(512, 3, padding="same", activation="relu", name="conv3")(x_image)
    x_image = layers.MaxPooling2D(pool_size=2, name="pool3")(x_image)
    x_image = layers.Conv2D(1024, 3, padding="same", activation="relu", name="conv4")(x_image)
    image_features = layers.GlobalAveragePooling2D(name="image_pool")(x_image)
    image_outputs = layers.Dense(256, activation="relu", name="image_projection")(image_features)
    return keras.Model(image_inputs, image_outputs, name="coco_image_cnn_tf")


def build_text_encoder(ndims: int) -> keras.Model:
    text_inputs = keras.Input(shape=(VOCAB_SIZE,), dtype="float32", name="caption_bow")
    x_text = layers.LayerNormalization(epsilon=1e-6)(text_inputs)
    x_text = layers.Dense(EMBED_DIM, activation="relu", name="bow_hidden")(x_text)
    x_text = layers.Dropout(TEXT_DROPOUT)(x_text)
    text_outputs = layers.Dense(ndims, activation="relu", name="text_projection")(x_text)
    return keras.Model(text_inputs, text_outputs, name="coco_caption_tf")


def build_model_list(ndims: int) -> list[keras.Model]:
    image_model = build_image_encoder(ndims)
    caption_model = build_text_encoder(ndims)
    return [image_model] + [caption_model for _ in range(NUM_CAPTION_VIEWS)]


def make_multiview_dataset(
    image_paths: list[str],
    caption_sets: list[list[str]],
    training: bool,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            np.asarray(image_paths, dtype=str),
            np.asarray(caption_sets, dtype=str),
        )
    )

    if training:
        dataset = dataset.shuffle(len(image_paths), seed=SEED, reshuffle_each_iteration=True)

    def map_example(image_path, captions):
        image_bytes = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = tf.cast(image, tf.float32)
        if training:
            image = tf.image.random_flip_left_right(image)

        caption_tokens = tf.cast(text_vectorizer(captions), tf.float32)

        views = (
            image,
            caption_tokens[0],
            caption_tokens[1],
            caption_tokens[2],
            caption_tokens[3],
            caption_tokens[4],
        )
        return (views,)

    dataset = dataset.map(map_example, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=training)
    return dataset.prefetch(AUTOTUNE)


def l2_normalize(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, eps)


def collect_image_text_embeddings(
    model: keras.Model,
    dataset: tf.data.Dataset,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    image_batches: list[np.ndarray] = []
    text_batches: list[np.ndarray] = []
    total = 0

    for batch in dataset:
        views = batch[0]
        views_nested = model.organize_inputs_by_model(views)
        image_embedding = model.model_list[0](views_nested[0], training=False)
        image_np = np.asarray(keras.ops.convert_to_numpy(image_embedding))

        text_view_embeddings: list[np.ndarray] = []
        for text_view_index in range(1, NUM_CAPTION_VIEWS + 1):
            text_embedding = model.model_list[text_view_index](
                views_nested[text_view_index],
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


def make_cifar_image_dataset(
    images: np.ndarray,
    batch_size: int = CIFAR_BATCH_SIZE,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(images)

    def preprocess(image):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        return image

    dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset.prefetch(AUTOTUNE)


def collect_cifar_embeddings(
    image_model: keras.Model,
    images: np.ndarray,
    batch_size: int = CIFAR_BATCH_SIZE,
) -> np.ndarray:
    image_ds = make_cifar_image_dataset(images, batch_size=batch_size)
    embeddings = image_model.predict(image_ds, verbose=1)
    return np.asarray(embeddings)


def aggregate_caption_groups(text_embeddings: np.ndarray) -> np.ndarray:
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
if RUN_BASELINES:
    print(f"Training each method for {BENCHMARK_EPOCHS} epochs with identical encoders.")
else:
    print(f"Training DLVPM only for {BENCHMARK_EPOCHS} epochs.")

benchmark_results: dict[str, dict[str, float]] = {}
benchmark_rank_results: dict[str, dict[str, np.ndarray]] = {}
method_order = ["DLVPM"]
steps_per_epoch = max(1, benchmark_train_n // BATCH_SIZE)
warmup_steps = max(1, steps_per_epoch * LEARNING_RATE_WARMUP_EPOCHS)
lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[warmup_steps],
    values=[LEARNING_RATE_START, LEARNING_RATE_END],
)

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
    order_type="callback",
    # order_association_cutoff=0.99,
)
dlvpm_optimizers = [
    keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    for _ in dlvpm_benchmark_models
]



dlvpm_benchmark.compile(dlvpm_optimizers)
dlvpm_benchmark.fit(
    benchmark_train_ds,
    validation_data=benchmark_val_ds,
    epochs=BENCHMARK_EPOCHS,
    verbose=True,
)


FINE_TUNE_EPOCHS = 10


# Re-compile so Keras respects the new trainable flags.
# Reuse the existing optimizers if possible.
dlvpm_optimizers = [
    keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0)
    for _ in dlvpm_benchmark_models
]
dlvpm_benchmark.compile(dlvpm_optimizers)

dlvpm_benchmark.order=False

dlvpm_benchmark.attention_mse=False

dlvpm_association_cutoff=False

dlvpm_benchmark.fit(
    benchmark_train_ds,
    validation_data=benchmark_val_ds,
    epochs=FINE_TUNE_EPOCHS,
    verbose=True,
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

dlvpm_test_dlvs = dlvpm_benchmark.predict(benchmark_test_ds)
dlvpm_corr_mat = dlvpm_benchmark.calculate_corrmat(dlvpm_test_dlvs)
plot_correlation_chord_row(
    dlvpm_corr_mat,
    ["Image", "Caption 1", "Caption 2", "Caption 3", "Caption 4", "Caption 5"],
    min_corr=0,
    node_cmap_name="Pastel1",
    figure_title="COCO DLVPM Cross-View Correlations",
    show_edge_labels=True,
    dpi=300,
    show=True,
)

if RUN_BASELINES:
    print("Training CLIP baseline...")
    clip_model_list = build_model_list(NDIMS)
    clip_model = CLIP(
        model_list=clip_model_list,
        regularizer_list=[None for _ in clip_model_list],
        ndims=NDIMS,
        is_siamese=False,
    )
    clip_optimizers = [
        keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
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
    benchmark_results["CLIP"], benchmark_rank_results["CLIP"] = evaluate_retrieval_result(
        clip_img,
        clip_txt,
    )
    method_order.append("CLIP")

    print("Training VICReg baseline...")
    vic_model_list = build_model_list(NDIMS)
    vic_model = VICReg(
        model_list=vic_model_list,
        regularizer_list=[None for _ in vic_model_list],
        ndims=NDIMS,
        is_siamese=False,
    )
    vic_optimizers = [
        keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
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
    lejepa_optimizers = [
        keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
        for _ in lejepa_model_list
    ]
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
    "\nGroup-level retrieval benchmark (higher Top-K accuracy is better, lower median rank is better):"
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


if RUN_CIFAR:
    print("\nRunning downstream CIFAR-10 evaluation on the trained COCO image encoder...")

    (x_train_cifar, y_train_cat), (x_test_cifar, y_test_cat) = keras.datasets.cifar10.load_data()
    y_train_cat = y_train_cat.squeeze()
    y_test_cat = y_test_cat.squeeze()

    image_model = dlvpm_benchmark.model_list[0]
    train_dlvs = collect_cifar_embeddings(
        image_model,
        x_train_cifar,
        batch_size=CIFAR_BATCH_SIZE,
    )
    test_dlvs = collect_cifar_embeddings(
        image_model,
        x_test_cifar,
        batch_size=CIFAR_BATCH_SIZE,
    )

    print(f"Train DLVs shape: {train_dlvs.shape}")
    print(f"Test  DLVs shape: {test_dlvs.shape}")

    svm_clf = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True)),
            ("svm", LinearSVC(C=1.0, max_iter=10000, random_state=42)),
        ]
    )
    svm_clf.fit(train_dlvs, y_train_cat)
    predictions = svm_clf.predict(test_dlvs)
    accuracy = accuracy_score(y_test_cat, predictions)

    print(f"\nSVM accuracy on CIFAR-10 test set: {accuracy:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test_cat, predictions, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test_cat, predictions))
