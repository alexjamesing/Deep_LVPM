MS COCO Tutorial (PyTorch Backend)
==================================

This tutorial shows how to use DLVPM to link images and captions in the MS COCO
dataset. The central idea is simple: each image is treated as one data view, and
each of its five human-written captions is treated as an additional language
view. DLVPM is then used to learn latent variables that are shared between the
image and caption modalities.

The full code lives in :mod:`deep_lvpm.tutorial.tutorial_coco_pytorch`, but the
walkthrough below is written so that you can read and run it in sections.

Prerequisites
-------------

This tutorial is heavier than the MNIST and TCGA examples. It requires:

- the PyTorch backend for Keras 3
- `FiftyOne <https://voxel51.com/fiftyone/>`_ to access COCO
- `transformers <https://huggingface.co/docs/transformers/index>`_ for the caption encoder
- internet access or local copies of the COCO image and caption files

In practice, a GPU or Apple Silicon machine is strongly recommended.

1. Set up the runtime and core configuration
--------------------------------------------

We begin by forcing the Keras torch backend and defining the main tutorial
settings in one place. The most important user-editable choices are:

- ``NUM_CAPTION_VIEWS``: how many captions per image to keep
- ``NDIMS``: the number of DLVs to learn
- ``BATCH_SIZE`` and ``BENCHMARK_EPOCHS``: the main training schedule

.. code-block:: python

   import os
   import json
   import random
   import zipfile
   from collections import defaultdict

   os.environ.setdefault("KERAS_BACKEND", "torch")
   os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
   os.environ.setdefault("USE_TF", "0")
   os.environ.setdefault("USE_TORCH", "1")
   os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

   import numpy as np
   import matplotlib.pyplot as plt
   from PIL import Image
   import torch
   import torch.nn as nn
   from torch.utils.data import DataLoader, Dataset
   import keras
   from keras import layers
   from transformers import AutoModel, AutoTokenizer
   import fiftyone as fo
   import fiftyone.zoo as foz

   from deep_lvpm.model import StructuralModel
   from deep_lvpm.multi_model import CLIP, VICReg, LeJEPA

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
   COCO_CAPTIONS_DIR = None

2. Define the COCO path model
-----------------------------

DLVPM needs a path matrix describing which views should be associated. In this
tutorial we use a star-shaped path model:

- the image is connected to each caption
- the captions are not directly connected to one another

That gives a six-view path matrix with one image node and five caption nodes.

.. code-block:: python

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

This is a useful first multimodal example because the shared structure should
capture image semantics that are consistently reflected in natural language.

3. Load the COCO image splits
-----------------------------

We next load the 2017 training and validation image splits using FiftyOne. The
images are loaded first; the caption annotations are attached in the next step.

.. code-block:: python

   FO_TRAIN_SPLIT = "train"
   FO_VAL_SPLIT = "validation"
   FO_LABEL_TYPES = []
   FO_CLASSES = None
   FO_MAX_SAMPLES_TRAIN = None
   FO_MAX_SAMPLES_VAL = None
   FO_SHUFFLE = True

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

4. Load caption annotations and build six-view samples
------------------------------------------------------

The next stage links each image to its five official COCO captions. The helper
functions in the tutorial do three things:

- locate the official ``captions_train2017.json`` and ``captions_val2017.json`` files
- load the caption annotations into dictionaries keyed by image id
- extract one image filepath and five captions per sample

.. code-block:: python

   COCO_CAPTIONS_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
   COCO_CAPTIONS_ARCHIVE = "annotations_trainval2017.zip"
   COCO_CAPTIONS_CACHE_SUBDIR = "deep_lvpm/coco"

   def resolve_coco_caption_annotations(split: str) -> str:
       ...

   def load_coco_captions(annotation_path: str) -> dict[int, list[str]]:
       ...

   def extract_coco_image_id(sample: "fo.Sample") -> int | None:
       ...

   def coco_view_to_examples(
       dataset: "fo.Dataset",
       captions_by_image_id: dict[int, list[str]],
   ) -> tuple[list[str], list[list[str]]]:
       ...

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

At the end of this step, each training example contains:

- one image path
- five human captions

5. Tokenize the caption views
-----------------------------

The caption views are processed with a DistilBERT encoder, so we tokenize the
text using the matching Hugging Face tokenizer.

.. code-block:: python

   RESAMPLE_BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
   text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, use_fast=True)

   def tokenize_caption_sets(caption_sets: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
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

6. Define measurement models for the image and caption views
------------------------------------------------------------

DLVPM requires one measurement model per data view. In this tutorial:

- the image view uses ``EfficientNetB0``
- each caption view uses a DistilBERT encoder followed by a Dense projection

Because the caption backbone is a native PyTorch module, the tutorial uses a
small wrapper layer so it can be called inside a Keras model running on the
torch backend.

.. code-block:: python

   class TorchModuleLayer(keras.layers.Layer):
       """Wrap a PyTorch module for execution inside a Keras torch-backend model."""
       ...

   class TextEncoderModule(nn.Module):
       """Caption encoder using a fully trainable pretrained DistilBERT backbone."""

       def __init__(self, model_name: str) -> None:
           super().__init__()
           self.backbone = AutoModel.from_pretrained(model_name)
           self.dropout = nn.Dropout(TEXT_DROPOUT)

       def forward(self, inputs) -> torch.Tensor:
           input_ids, attention_mask = inputs
           outputs = self.backbone(
               input_ids=input_ids.long().contiguous(),
               attention_mask=attention_mask.long().contiguous(),
           )
           ...

   def build_image_encoder(NDIMS) -> keras.Model:
       image_inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
       image_base = keras.applications.EfficientNetB0(
           include_top=False,
           weights="imagenet",
           pooling="avg",
           input_shape=(IMG_SIZE, IMG_SIZE, 3),
       )
       image_base.trainable = True

       x_image = keras.applications.efficientnet.preprocess_input(image_inputs)
       image_features = image_base(x_image, training=False)
       image_outputs = layers.Dense(NDIMS, activation="relu", name="image_projection")(image_features)
       return keras.Model(image_inputs, image_outputs, name="coco_efficientnetb0")

   def build_text_encoder(NDIMS) -> keras.Model:
       token_ids = keras.Input(shape=(MAX_TOKENS,), dtype="int32", name="caption_token_ids")
       attention_mask = keras.Input(shape=(MAX_TOKENS,), dtype="int32", name="caption_attention_mask")
       text_features = TorchModuleLayer(
           TextEncoderModule(model_name=TEXT_MODEL_NAME),
           input_dtype=("int32", "int32"),
           name="caption_backbone",
       )([token_ids, attention_mask])
       text_outputs = layers.Dense(NDIMS, activation="relu", name="text_projection")(text_features)
       return keras.Model([token_ids, attention_mask], text_outputs, name="coco_caption_torch")

   def build_model_list(NDIMS) -> list[keras.Model]:
       image_model = build_image_encoder(NDIMS)
       caption_model = build_text_encoder(NDIMS)
       return [image_model] + [caption_model for _ in range(NUM_CAPTION_VIEWS)]

7. Build multiview dataloaders
------------------------------

The dataloader returns one image tensor plus five tokenized caption views per
sample. This is the exact six-view structure that ``StructuralModel`` expects.

.. code-block:: python

   class CocoRetrievalDataset(Dataset):
       """Load COCO images and five tokenized captions per sample."""
       ...

   def collate_multiview(batch):
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

8. Build and train the DLVPM model
----------------------------------

Once the measurement models and loaders are ready, we can build the DLVPM
structural model in the same way as the other tutorials. The most important
arguments are:

- ``Path``: the six-view image-caption path matrix
- ``model_list``: one image encoder plus five caption encoders
- ``tot_num``: the total number of training samples
- ``ndims``: the number of DLVs
- ``order=True``: order latent variables by shared structure
- ``order_association_cutoff=0.99``: discard final ordered dimensions once the
  cumulative association mass has saturated

.. code-block:: python

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

   steps_per_epoch = max(1, benchmark_train_n // BATCH_SIZE)
   warmup_steps = max(1, steps_per_epoch * LEARNING_RATE_WARMUP_EPOCHS)
   lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
       boundaries=[warmup_steps],
       values=[LEARNING_RATE_START, LEARNING_RATE_END],
   )

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

9. Evaluate image-text retrieval
--------------------------------

After training, we test whether matching images and caption sets are close in
the learned latent space. The tutorial does this by:

- collecting image embeddings and caption embeddings
- averaging the five captions for each image into one caption-group embedding
- ranking image-to-caption and caption-to-image matches
- reporting top-k accuracy and median rank

.. code-block:: python

   def collect_image_text_embeddings(
       model: keras.Model,
       dataset: DataLoader,
       max_samples: int,
   ) -> tuple[np.ndarray, np.ndarray]:
       ...

   def aggregate_caption_groups(text_embeddings: np.ndarray) -> np.ndarray:
       ...

   def retrieval_rank_arrays(
       image_embeddings: np.ndarray,
       text_embeddings: np.ndarray,
   ) -> tuple[np.ndarray, np.ndarray]:
       ...

   def retrieval_metrics_from_ranks(
       i2g_rank: np.ndarray,
       g2i_rank: np.ndarray,
       ks: tuple[int, ...] = RETRIEVAL_KS,
   ) -> dict[str, float]:
       ...

   dlvpm_img, dlvpm_txt = collect_image_text_embeddings(
       dlvpm_benchmark,
       benchmark_test_ds,
       max_samples=benchmark_test_n,
   )

   benchmark_results, rank_results = evaluate_retrieval_result(
       dlvpm_img,
       dlvpm_txt,
   )

Typical summary metrics include:

- ``i2g_top1`` and ``g2i_top1``: top-1 retrieval accuracy
- ``i2g_top5`` and ``g2i_top5``: top-5 retrieval accuracy
- ``i2g_median_rank`` and ``g2i_median_rank``: median rank of the correct match

10. Optional multimodal baselines
---------------------------------

If ``RUN_BASELINES=True``, the tutorial also trains CLIP, VICReg, and LeJEPA
using the same image and caption encoders. This gives a direct comparison
between DLVPM and several other multimodal representation-learning methods on
the same held-out retrieval task.

.. code-block:: python

   if RUN_BASELINES:
       clip_model = CLIP(...)
       vic_model = VICReg(...)
       lejepa_model = LeJEPA(...)

This is useful if your goal is benchmarking rather than simply learning to use
DLVPM.

Summary
-------

This tutorial illustrates a full multimodal DLVPM workflow:

- define a structural path model over six views
- build separate image and text measurement models
- train DLVPM to align the views in a shared latent space
- evaluate the latent representation with a retrieval benchmark

If you want a simpler starting point, begin with the :doc:`mnist` or
:doc:`tcga_torch` tutorials and then return to this COCO example once you are
comfortable with the general DLVPM pattern.
