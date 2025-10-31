Introduction
============

Deep Latent Variable Path Modelling (DLVPM) is a flexible framework for linking disparate data types by learning sets of orthogonal deep latent variables (DLVs).  It combines deep neural networks (measurement models) with a user‑specified **structural path model** to capture and optimise associations between data views.

DLVPM models are constructed with the high‑level Keras API.  For each data view you define a Keras model (e.g., a convolutional network for images or a fully connected network for omics), and DLVPM learns a shared latent representation by maximising correlations between the network outputs.  The structural path matrix specifies which latent factors are connected across views.

This documentation explains how to install the toolbox, demonstrates three end-to-end tutorials (MNIST digits, a TCGA lung cancer multi‑omics dataset, and a Siamese CIFAR‑10 contrastive workflow), and describes the API for the core classes and custom layers.  Users unfamiliar with Keras or TensorFlow may wish to consult the `TensorFlow Keras guide <https://www.tensorflow.org/guide/keras>`_ for background; PyTorch users can consult the `PyTorch overview <https://pytorch.org/get-started/locally/>`_ before switching the Keras backend.

New in the Keras 3 release:

* **Multi-backend execution** – install the TensorFlow or PyTorch extras that match your hardware and set ``KERAS_BACKEND`` accordingly.
* **Expanded metrics** – :meth:`StructuralModel.evaluate` now reports total loss, cross-view correlation, mean squared error, and a redundancy score that captures within-view correlations.
* **Updated tutorials** – all walkthroughs have been refreshed for Keras 3 and the new Siamese example highlights the ``is_siamese`` mode for contrastive learning.

This work has now been published here: https://www.nature.com/articles/s42256-025-01052-4.
