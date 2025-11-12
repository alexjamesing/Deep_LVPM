Introduction
============

Deep Latent Variable Path Modelling (DLVPM) is a flexible framework for linking disparate data types by learning sets of orthogonal deep latent variables (DLVs).  It combines deep neural networks (measurement models) with a user‑specified **structural path model** to capture and optimise associations between data views.

DLVPM models are constructed with the high‑level Keras API.  For each data view you define a Keras model (e.g., a convolutional network for images or a fully connected network for omics), and DLVPM learns a shared latent representation by maximising correlations between the network outputs.  The structural path matrix specifies which latent factors are connected across views.

This documentation explains how to install the toolbox, demonstrates two example applications (MNIST digits and a TCGA lung cancer multi‑omics dataset), and describes the API for the core classes and custom layers.  Users unfamiliar with Keras or TensorFlow may wish to consult the `TensorFlow Keras guide <https://www.tensorflow.org/guide/keras>`_ for background.

This work has now been published here: https://www.nature.com/articles/s42256-025-01052-4.