MS COCO Tutorial
================

This tutorial links images and captions in MS COCO.  Each image is one data
view, and each of its five human-written captions is a separate language view.
DLVPM learns latent variables shared between the image and caption modalities,
then the learned representation is evaluated with an image-text retrieval task.

The full runnable script is :mod:`deep_lvpm.tutorial.tutorial_coco`.

Run it with:

.. code-block:: bash

   python -m deep_lvpm.tutorial.tutorial_coco

Prerequisites
-------------

This is the heaviest tutorial.  It uses ``torchvision`` image models,
Hugging Face ``transformers``, ``fiftyone`` for COCO access, and the standard
scientific Python stack.  A GPU or Apple Silicon machine is strongly
recommended.

1. Configure the benchmark
--------------------------

The tutorial keeps the important settings near the top of the script.  The
main modelling choice is to treat each sample as six views: one image and five
caption views.

.. code-block:: python

   import os
   import json
   import random
   import zipfile
   import urllib.request
   from collections import defaultdict

   os.environ.setdefault("USE_TF", "0")
   os.environ.setdefault("USE_TORCH", "1")
   os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

   import numpy as np
   import matplotlib.pyplot as plt
   from PIL import Image
   import torch
   import torch.nn as nn
   from torch.utils.data import DataLoader, Dataset
   from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
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
   LEARNING_RATE_START = 1e-5
   LEARNING_RATE_END = 1e-4
   LEARNING_RATE_WARMUP_EPOCHS = 5
   TEST_FRACTION = 0.10
   SEED = 51

2. Define the image-caption path model
--------------------------------------

The path matrix is star-shaped.  The image view is connected to every caption
view.  Captions are not directly connected to one another, because the shared
structure should pass through the image.

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

3. Load COCO images and captions
--------------------------------

The image splits come from the FiftyOne COCO zoo integration.  The caption JSON
files are resolved locally when available and downloaded from the official COCO
annotation archive when needed.

.. code-block:: python

   train_view = foz.load_zoo_dataset(
       "coco-2017",
       split="train",
       label_types=[],
       shuffle=True,
       seed=SEED,
       dataset_name="dlvpm-coco2017-train",
       include_id=True,
   )

   val_view = foz.load_zoo_dataset(
       "coco-2017",
       split="validation",
       label_types=[],
       shuffle=True,
       seed=SEED,
       dataset_name="dlvpm-coco2017-val",
       include_id=True,
   )

The helper functions in the script do three jobs: find the annotation JSON,
load captions into a dictionary keyed by image id, and turn each image into one
filepath plus five captions.

.. code-block:: python

   def resolve_coco_caption_annotations(split: str) -> str:
       # Locate captions_train2017.json or captions_val2017.json.
       # If the file is not present locally, download and extract the official
       # COCO annotations archive.
       ...

   def load_coco_captions(annotation_path: str) -> dict[int, list[str]]:
       with open(annotation_path, "r", encoding="utf-8") as f:
           payload = json.load(f)

       captions_by_image_id = defaultdict(list)
       for annotation in payload["annotations"]:
           captions_by_image_id[int(annotation["image_id"])].append(annotation["caption"])
       return captions_by_image_id

   def coco_view_to_examples(view, captions_by_image_id):
       image_paths = []
       caption_sets = []
       for sample in view:
           image_id = extract_coco_image_id(sample)
           captions = captions_by_image_id.get(image_id, [])
           if len(captions) < NUM_CAPTION_VIEWS:
               continue
           image_paths.append(sample.filepath)
           caption_sets.append(captions[:NUM_CAPTION_VIEWS])
       return image_paths, caption_sets

4. Tokenize caption views
-------------------------

The caption encoder is DistilBERT, so each caption view is tokenized into
``input_ids`` and ``attention_mask`` arrays.  Each sample has five captions,
and each caption is treated as its own DLVPM view.

.. code-block:: python

   tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

   def tokenize_caption_sets(caption_sets: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
       flat_captions = [caption for caption_set in caption_sets for caption in caption_set]
       tokenized = tokenizer(
           flat_captions,
           padding="max_length",
           truncation=True,
           max_length=MAX_TOKENS,
           return_tensors="np",
       )
       input_ids = tokenized["input_ids"].reshape(len(caption_sets), NUM_CAPTION_VIEWS, MAX_TOKENS)
       attention_mask = tokenized["attention_mask"].reshape(len(caption_sets), NUM_CAPTION_VIEWS, MAX_TOKENS)
       return input_ids.astype("int64"), attention_mask.astype("int64")

5. Define image and text measurement models
-------------------------------------------

The image measurement model uses EfficientNet-B0 and replaces the classifier
with a projection to ``NDIMS``.  The caption measurement model uses DistilBERT
and projects the pooled text representation to the same width.

.. code-block:: python

   class TextEncoderModule(nn.Module):
       def __init__(self, model_name: str) -> None:
           super().__init__()
           self.text_model = AutoModel.from_pretrained(model_name)
           hidden_size = int(self.text_model.config.hidden_size)
           self.dropout = nn.Dropout(TEXT_DROPOUT)
           self.projection = nn.Linear(hidden_size, NDIMS)
           self.n_inputs = 2

       def forward(self, inputs):
           input_ids, attention_mask = inputs
           outputs = self.text_model(
               input_ids=input_ids.long(),
               attention_mask=attention_mask.long(),
           )
           cls_embedding = outputs.last_hidden_state[:, 0, :]
           cls_embedding = self.dropout(cls_embedding)
           return self.projection(cls_embedding)

   class ImageEncoderModule(nn.Module):
       def __init__(self) -> None:
           super().__init__()
           weights = EfficientNet_B0_Weights.DEFAULT
           self.transforms = weights.transforms()
           self.backbone = efficientnet_b0(weights=weights)
           in_features = int(self.backbone.classifier[1].in_features)
           self.backbone.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, NDIMS))
           self.n_inputs = 1

       def forward(self, images):
           return self.backbone(images)

   def build_model_list(NDIMS) -> list[nn.Module]:
       image_model = ImageEncoderModule()
       text_models = [TextEncoderModule(TEXT_MODEL_NAME) for _ in range(NUM_CAPTION_VIEWS)]
       return [image_model] + text_models

6. Build multiview dataloaders
------------------------------

Each batch contains one image tensor plus five pairs of caption tensors.  The
``n_inputs`` attribute on the text encoder tells ``StructuralModel`` that each
caption view consumes two tensors.

.. code-block:: python

   class CocoRetrievalDataset(Dataset):
       def __init__(self, image_paths, caption_sets, training=True):
           self.image_paths = list(image_paths)
           self.caption_sets = list(caption_sets)
           self.training = bool(training)
           self.input_ids, self.attention_mask = tokenize_caption_sets(self.caption_sets)

       def __len__(self):
           return len(self.image_paths)

       def __getitem__(self, index):
           image = Image.open(self.image_paths[index]).convert("RGB")
           image = image.resize((IMG_SIZE, IMG_SIZE), RESAMPLE_BICUBIC)
           image = np.asarray(image).astype("float32") / 255.0
           image = np.transpose(image, (2, 0, 1))

           caption_inputs = []
           for caption_index in range(NUM_CAPTION_VIEWS):
               caption_inputs.append(self.input_ids[index, caption_index])
               caption_inputs.append(self.attention_mask[index, caption_index])
           return tuple([image] + caption_inputs)

   def collate_multiview(batch):
       columns = list(zip(*batch))
       return tuple(torch.as_tensor(np.stack(column)) for column in columns)

   def make_multiview_loader(image_paths, caption_sets, training=True):
       dataset = CocoRetrievalDataset(image_paths, caption_sets, training=training)
       return DataLoader(
           dataset,
           batch_size=BATCH_SIZE,
           shuffle=bool(training),
           num_workers=NUM_WORKERS,
           collate_fn=collate_multiview,
           drop_last=bool(training),
       )

7. Train DLVPM
--------------

Before training, the script builds smaller benchmark splits from the full COCO
views.  This keeps the tutorial practical while preserving a held-out test
subset for retrieval evaluation.

.. code-block:: python

   benchmark_train_n = min(BENCHMARK_TRAIN_SAMPLES, len(train_paths))
   benchmark_val_n = min(BENCHMARK_VAL_SAMPLES, len(val_paths))
   benchmark_test_n = min(BENCHMARK_SAMPLES, len(test_paths))

   benchmark_train_ds = make_multiview_loader(
       train_paths[:benchmark_train_n],
       train_caption_sets[:benchmark_train_n],
       training=True,
   )
   benchmark_val_ds = make_multiview_loader(
       val_paths[:benchmark_val_n],
       val_caption_sets[:benchmark_val_n],
       training=False,
   )
   benchmark_test_ds = make_multiview_loader(
       test_paths[:benchmark_test_n],
       test_caption_sets[:benchmark_test_n],
       training=False,
   )

The benchmark model uses ZCA orthogonalization and ordered DLVs.  The ordered
dimension cutoff keeps the smallest number of ordered dimensions whose
cumulative association mass reaches the cutoff.

.. code-block:: python

   def make_optimizer_list(model: nn.Module) -> list[torch.optim.Optimizer]:
       return [
           torch.optim.Adam(view_model.parameters(), lr=LEARNING_RATE_START)
           for view_model in model.model_list
       ]

   def make_scheduler_list(optimizers: list[torch.optim.Optimizer]) -> list[torch.optim.lr_scheduler.LambdaLR]:
       def schedule(epoch: int) -> float:
           if epoch < LEARNING_RATE_WARMUP_EPOCHS:
               return 1.0
           return LEARNING_RATE_END / LEARNING_RATE_START

       return [torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule) for optimizer in optimizers]

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

8. Evaluate image-text retrieval
--------------------------------

After training, the script collects image embeddings and aggregated caption
embeddings, normalizes them, and computes bidirectional retrieval ranks.  The
reported metrics include recall at ``k`` and median rank.

.. code-block:: python

   def l2_normalize(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
       denom = np.linalg.norm(matrix, axis=1, keepdims=True)
       return matrix / np.maximum(denom, eps)

   def aggregate_caption_groups(text_embeddings: np.ndarray) -> np.ndarray:
       return text_embeddings.reshape(-1, NUM_CAPTION_VIEWS, text_embeddings.shape[-1]).mean(axis=1)

   def retrieval_metrics_from_ranks(ranks: np.ndarray) -> dict[str, float]:
       metrics = {f"recall@{k}": float(np.mean(ranks <= k)) for k in RETRIEVAL_KS}
       metrics["median_rank"] = float(np.median(ranks))
       metrics["mean_rank"] = float(np.mean(ranks))
       return metrics

   dlvpm_img, dlvpm_txt = collect_image_text_embeddings(
       dlvpm_benchmark,
       benchmark_test_ds,
       max_samples=benchmark_test_n,
   )
   benchmark_results["DLVPM"], benchmark_rank_results["DLVPM"] = evaluate_retrieval_result(
       dlvpm_img,
       dlvpm_txt,
   )

9. Optional baselines
---------------------

If ``RUN_BASELINES`` is set to ``True``, the same encoders and data are used to
train CLIP, VICReg, and LeJEPA baselines from :mod:`deep_lvpm.multi_model`.
This is intended for benchmarking rather than for the minimum DLVPM workflow.

.. code-block:: python

   if RUN_BASELINES:
       for method_name, method_class in [
           ("CLIP", CLIP),
           ("VICReg", VICReg),
           ("LeJEPA", LeJEPA),
       ]:
           baseline_models = build_model_list(NDIMS)
           baseline_model = method_class(
               baseline_models,
               [None for _ in baseline_models],
               NDIMS,
           )
           baseline_optimizers = make_optimizer_list(baseline_model)
           baseline_model.compile(baseline_optimizers)
           baseline_model.fit(
               benchmark_train_ds,
               validation_data=benchmark_val_ds,
               epochs=BENCHMARK_EPOCHS,
               verbose=True,
           )

Summary
-------

The COCO tutorial is a full multimodal example: it builds six aligned data
views, uses pretrained image and text encoders as PyTorch measurement models,
trains DLVPM with an image-caption path matrix, and evaluates whether the
learned latent space supports retrieval.
