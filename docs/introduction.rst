Introduction
============

Deep Latent Variable Path Modelling (DLVPM) is a flexible framework for linking disparate data types by learning sets of orthogonal deep latent variables (DLVs).  It combines deep neural networks (measurement models) with a user‑specified **structural path model** to capture and optimise associations between data views.

DLVPM models are constructed with the high‑level Keras API.  The toolbox now targets **Keras 3** and runs backend‑agnostically on either TensorFlow or PyTorch (select one by installing the matching pip extra).  For each data view you define a Keras model, and DLVPM learns a shared latent representation by maximising correlations between the network outputs.  The structural path matrix specifies which latent factors are connected across views.

This documentation explains how to install the toolbox, demonstrates example applications (MNIST digits, a TCGA lung cancer multi‑omics dataset, and a Siamese CIFAR‑10 tutorial), and describes the API for the core classes and custom layers.  Users new to Keras 3 can consult the `TensorFlow Keras guide <https://www.tensorflow.org/guide/keras>`_ or refer to the Keras 3 backend documentation for backend‑specific tips.

This work has now been published here: https://www.nature.com/articles/s42256-025-01052-4.
