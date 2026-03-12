MS COCO Image-Text Tutorial
===========================

This tutorial demonstrates how to train a DLVPM model on MS COCO using one image view and multiple text views.
The image encoder is an EfficientNetB0 convolutional network and the text encoder is a Transformer block.
Each image is paired with its five official COCO captions, and retrieval metrics
are averaged across the five captions for each image.

Prerequisites
-------------

Install :mod:`deep_lvpm` with TensorFlow and the COCO tutorial extra dependencies.
Set the backend before running the script.

.. code-block:: bash

   export KERAS_BACKEND=tensorflow
   pip install -e ".[tf-cpu,tutorial-coco]"

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

6. Benchmark against CLIP and VICReg
------------------------------------

The script trains DLVPM,
CLIP, and VICReg with the same COCO pipeline and compares cross-modal retrieval quality.

Benchmark task definition
^^^^^^^^^^^^^^^^^^^^^^^^^

For each image in a held-out set, retrieve its matching captions from a pool of
all captions (image-to-text retrieval, abbreviated ``i2t``), and also retrieve
the matching image given each caption (text-to-image retrieval, ``t2i``). The
script averages retrieval accuracy across the five captions attached to each
image, so the benchmark measures true COCO caption retrieval rather than
single-caption matching.

Metrics:

1. ``Recall@K`` (``i2t_R@K`` and ``t2i_R@K``) reports caption-averaged retrieval
   success. ``i2t_R@1 = 0.45`` means that, on average, 45% of the five ground-truth
   captions per image are ranked first. ``t2i_R@10`` reports how often each caption
   retrieves its paired image in the top 10, averaged over the five captions linked
   to every image.
2. Median rank is computed from the mean rank across the five captions associated
   with each image. Lower values indicate that matched image/caption groups appear
   nearer the top of the ranked list.

Higher Recall@K and lower median rank both indicate better cross-modal retrieval
performance. Reporting both numbers provides a quick way to compare precision at
strict cutoffs (Recall@K) as well as the overall ordering quality (median rank).

Benchmark configuration (in script)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   BENCHMARK_EPOCHS = 5
   BENCHMARK_TRAIN_SAMPLES = 20000
   BENCHMARK_VAL_SAMPLES = 5000
   BENCHMARK_SAMPLES = 2048
   RETRIEVAL_KS = (1, 5, 10)

These knobs determine how many COCO samples are used for training/validation/testing
the benchmark models and which Recall@K cutoffs are reported.

Implementation in the tutorial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The script performs the benchmark by:

1. Building fixed benchmark train/validation/test subsets.
2. Training DLVPM, CLIP, and VICReg for the same number of epochs.
3. Collecting image embeddings plus all five caption embeddings from each trained model.
4. L2-normalising embeddings.
5. Computing cosine similarity between all image-text pairs.
6. Reporting ``i2t_R@K``, ``t2i_R@K``, and median ranks in a comparison table.

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

   # Compute and print retrieval metrics table
   header = ["Method", "i2t_R@1", "t2i_R@1", ...]
   ...

.. code-block:: python

   def retrieval_metrics(image_embeddings, text_embeddings, ks=(1, 5, 10)):
       image_embeddings = l2_normalize(image_embeddings.astype("float32"))
       text_embeddings = l2_normalize(text_embeddings.astype("float32"))
       flat_text_embeddings = text_embeddings.reshape(
           image_embeddings.shape[0] * NUM_CAPTION_VIEWS, -1
       )
       caption_owner = np.repeat(np.arange(image_embeddings.shape[0]), NUM_CAPTION_VIEWS)

       i2t_order = np.argsort(-(image_embeddings @ flat_text_embeddings.T), axis=1)
       i2t_positive_mask = caption_owner[i2t_order] == np.arange(image_embeddings.shape[0])[:, None]
       i2t_rank = np.where(i2t_positive_mask)[1].reshape(-1, NUM_CAPTION_VIEWS) + 1

       t2i_order = np.argsort(-(flat_text_embeddings @ image_embeddings.T), axis=1)
       t2i_rank = np.argmax(t2i_order == caption_owner[:, None], axis=1).reshape(-1, NUM_CAPTION_VIEWS) + 1

       metrics = {}
       for k in ks:
           metrics[f"i2t_R@{k}"] = float(np.mean(np.mean(i2t_rank <= k, axis=1)))
           metrics[f"t2i_R@{k}"] = float(np.mean(np.mean(t2i_rank <= k, axis=1)))
       metrics["i2t_median_rank"] = float(np.median(np.mean(i2t_rank, axis=1)))
       metrics["t2i_median_rank"] = float(np.median(np.mean(t2i_rank, axis=1)))
       return metrics

The benchmark section is implemented directly in
:mod:`deep_lvpm.tutorial.tutorial_coco_tf`.
