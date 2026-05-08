MS COCO Image-Text Tutorial
===========================

This tutorial demonstrates how to train a DLVPM model on MS COCO using one image view and multiple text views.
The image encoder is an EfficientNetB0 convolutional network and the text encoder is a Transformer block.
Each image is paired with its five official COCO captions, and retrieval metrics
are averaged across the five captions for each image.

Prerequisites
-------------

Install :mod:`deep_lvpm` with TensorFlow. The COCO tutorial dependencies are
installed by default.
Set the backend before running the script.

.. code-block:: bash

   export KERAS_BACKEND=tensorflow
   pip install -e ".[tf-cpu]"

1. Imports and configuration
----------------------------

The tutorial script lives at :mod:`deep_lvpm.tutorial.tutorial_coco_tf`.
It defines the dataset/model hyperparameters up front for easy editing in an IDE.

.. code-block:: python

   import os
   os.environ["KERAS_BACKEND"] = "tensorflow"

   import numpy as np
   import tensorflow as tf
   import keras
   from keras import layers
   import fiftyone as fo
   import fiftyone.zoo as foz

   from deep_lvpm.model import StructuralModel

   NUM_CAPTION_VIEWS = 5
   IMG_SIZE = 224
   MAX_TOKENS = 32
   VOCAB_SIZE = 30000
   EMBED_DIM = 256
   TRANSFORMER_HEADS = 4
   TRANSFORMER_FF_DIM = 512

2. Load COCO and build five-caption groups
------------------------------------------

COCO images are loaded from the FiftyOne zoo. The script then loads the official
caption annotation JSON files and links each image to its five captions. Since
COCO's public ``test`` split does not include captions, the tutorial creates a
held-out test subset from the training split.

.. code-block:: python

   train_view = foz.load_zoo_dataset(
       "coco-2017",
       split="train",
       label_types=[],
       shuffle=True,
       seed=51,
       dataset_name="dlvpm-coco2017-train",
       include_id=True,
   )

   val_view = foz.load_zoo_dataset(
       "coco-2017",
       split="validation",
       label_types=[],
       shuffle=True,
       seed=51,
       dataset_name="dlvpm-coco2017-val",
       include_id=True,
   )

   train_caption_annotations = load_coco_captions(
       resolve_coco_caption_annotations("train")
   )
   val_caption_annotations = load_coco_captions(
       resolve_coco_caption_annotations("validation")
   )

   # Each example is now:
   #   image path -> [caption_1, caption_2, caption_3, caption_4, caption_5]

3. Define EfficientNetB0 and Transformer measurement models
------------------------------------------------------------

The image view uses EfficientNetB0 (ImageNet weights) with a small projection head.
The text view uses token and positional embeddings followed by a Transformer encoder block.

.. code-block:: python

   image_inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
   image_base = keras.applications.EfficientNetB0(
       include_top=False,
       weights="imagenet",
       pooling="avg",
       input_shape=(IMG_SIZE, IMG_SIZE, 3),
   )
   x_image = keras.applications.efficientnet.preprocess_input(image_inputs)
   x_image = image_base(x_image, training=False)
   image_outputs = layers.Dense(128, activation="relu", name="image_projection")(x_image)
   image_model = keras.Model(image_inputs, image_outputs)

   text_inputs = keras.Input(shape=(MAX_TOKENS,), dtype="int32", name="caption_tokens")
   token_embeddings = layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)(text_inputs)
   position_indices = tf.range(start=0, limit=MAX_TOKENS, delta=1)
   position_embeddings = layers.Embedding(MAX_TOKENS, EMBED_DIM)(position_indices)
   x_text = token_embeddings + position_embeddings

   # Transformer encoder block (MHA + FFN with residual connections)
   ...

4. Build multi-view datasets for DLVPM
--------------------------------------

For each sample, the pipeline emits one image plus ``NUM_CAPTION_VIEWS`` tokenised text views.
This structure is returned as ``(tuple(views),)`` so Keras treats it as one multi-input ``x``.

.. code-block:: python

   def make_multiview_dataset(image_paths, caption_sets, training):
       dataset = tf.data.Dataset.from_tensor_slices((image_paths, caption_sets))

       def map_example(image_path, captions):
           image = tf.io.decode_jpeg(tf.io.read_file(image_path), channels=3)
           image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
           image = tf.cast(image, tf.float32)

           caption_tokens = tf.cast(text_vectorizer(captions), tf.int32)

           views = [image] + [caption_tokens[i] for i in range(NUM_CAPTION_VIEWS)]
           return (tuple(views),)

       return dataset.map(map_example).batch(128).prefetch(tf.data.AUTOTUNE)

5. Build and compile the StructuralModel
----------------------------------------

The model list contains one image encoder and repeated references to the same text encoder
(shared caption weights across text views).

.. code-block:: python

   model_list = [image_model] + [caption_model for _ in range(NUM_CAPTION_VIEWS)]
   regularizer_list = [None for _ in model_list]

   n_views = len(model_list)
   Path = tf.ones((n_views, n_views), dtype=tf.float32) - tf.eye(n_views, dtype=tf.float32)

   dlvpm_model = StructuralModel(
       Path=Path,
       model_list=model_list,
       regularizer_list=regularizer_list,
       tot_num=len(train_paths),
       ndims=256,
       orthogonalization="zca",
       diag_offset=1e-6,
       train_DLV=True,
   )

   optimizer_list = [keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0)
                     for _ in model_list]
   dlvpm_model.compile(optimizer_list)

6. Benchmark against CLIP, VICReg, and LeJEPA
---------------------------------------------

The script trains DLVPM,
CLIP, VICReg, and LeJEPA with the same COCO pipeline and compares cross-modal retrieval quality.

Benchmark task definition
^^^^^^^^^^^^^^^^^^^^^^^^^

For each image in a held-out set, retrieve its matching five-caption group from
a pool of caption groups (image-to-group retrieval, abbreviated ``i2g``), and
also retrieve the matching image given the aggregated caption group
(``g2i``). The script aggregates the five caption embeddings for each image
into one caption-group embedding before ranking, so the benchmark measures
whether a method links the image and caption set as a whole.

Metrics:

1. ``Top-K`` group accuracy (``i2g_topK`` and ``g2i_topK``) reports whether the
   correct caption group or image is retrieved within the top ``K`` ranked
   candidates. ``i2g_top1 = 0.45`` means 45% of images retrieve the correct
   caption group as their first-ranked match.
2. Median rank is computed directly on group-level ranks. Lower values indicate
   that matched image/caption groups appear nearer the top of the ranked list.

Higher Top-K accuracy and lower median rank both indicate better cross-modal
retrieval performance. Reporting both numbers provides a quick way to compare
strict matching quality and overall ordering quality.

Benchmark configuration (in script)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   BENCHMARK_EPOCHS = 5
   BENCHMARK_TRAIN_SAMPLES = 20000
   BENCHMARK_VAL_SAMPLES = 5000
   BENCHMARK_SAMPLES = 2048
   RETRIEVAL_KS = (1, 5, 10)

These knobs determine how many COCO samples are used for training/validation/testing
the benchmark models and which Top-K cutoffs are reported.

Implementation in the tutorial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The script performs the benchmark by:

1. Building fixed benchmark train/validation/test subsets.
2. Training DLVPM, CLIP, VICReg, and LeJEPA for the same number of epochs.
3. Collecting image embeddings plus all five caption embeddings from each trained model.
4. L2-normalising embeddings.
5. Aggregating each five-caption set into one normalized caption-group embedding.
6. Computing cosine similarity between images and caption groups.
7. Reporting ``i2g_topK``, ``g2i_topK``, and median ranks in a comparison table.

.. code-block:: python

   benchmark_train_n = min(BENCHMARK_TRAIN_SAMPLES, len(train_paths))
   benchmark_val_n = min(BENCHMARK_VAL_SAMPLES, len(val_paths))
   benchmark_test_n = min(BENCHMARK_SAMPLES, len(test_paths))

   benchmark_train_ds = make_multiview_dataset(
       train_paths[:benchmark_train_n], train_caption_sets[:benchmark_train_n], training=True
   )
   benchmark_val_ds = make_multiview_dataset(
       val_paths[:benchmark_val_n], val_caption_sets[:benchmark_val_n], training=False
   )
   benchmark_test_ds = make_multiview_dataset(
       test_paths[:benchmark_test_n], test_caption_sets[:benchmark_test_n], training=False
   )

   benchmark_results = {}

   # Train each method with identical settings.
   dlvpm_benchmark = StructuralModel(...)
   ...
   benchmark_results["DLVPM"] = retrieval_metrics(dlvpm_img, dlvpm_txt)

   clip_model = CLIP(...)
   ...
   benchmark_results["CLIP"] = retrieval_metrics(clip_img, clip_txt)

   vic_model = VICReg(...)
   ...
   benchmark_results["VICReg"] = retrieval_metrics(vic_img, vic_txt)

   lejepa_model = LeJEPA(...)
   ...
   benchmark_results["LeJEPA"] = retrieval_metrics(lejepa_img, lejepa_txt)

   # Compute and print retrieval metrics table
   header = ["Method", "i2g_top1", "g2i_top1", ...]
   ...

.. code-block:: python

   def retrieval_metrics(image_embeddings, text_embeddings, ks=(1, 5, 10)):
       image_embeddings = l2_normalize(image_embeddings.astype("float32"))
       group_embeddings = aggregate_caption_groups(text_embeddings.astype("float32"))
       group_similarity = image_embeddings @ group_embeddings.T
       target_index = np.arange(image_embeddings.shape[0])

       i2g_order = np.argsort(-group_similarity, axis=1)
       i2g_rank = np.argmax(i2g_order == target_index[:, None], axis=1) + 1

       g2i_order = np.argsort(-group_similarity.T, axis=1)
       g2i_rank = np.argmax(g2i_order == target_index[:, None], axis=1) + 1

       metrics = {}
       for k in ks:
           metrics[f"i2g_top{k}"] = float(np.mean(i2g_rank <= k))
           metrics[f"g2i_top{k}"] = float(np.mean(g2i_rank <= k))
       metrics["i2g_median_rank"] = float(np.median(i2g_rank))
       metrics["g2i_median_rank"] = float(np.median(g2i_rank))
       return metrics

The benchmark section is implemented directly in
:mod:`deep_lvpm.tutorial.tutorial_coco_tf`.
