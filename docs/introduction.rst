Introduction
============

Deep Latent Variable Path Modelling (DLVPM) is a method for path/structural
equation modelling using deep neural networks.  The aim of the method is to
connect different data types together via sets of orthogonal deep latent
variables (DLVs).  The method can also be used in a Siamese network
configuration to learn representations of a single data type.

This version is implemented in native PyTorch.  For each data view you define a
``torch.nn.Module`` measurement model, and DLVPM learns a shared latent
representation by maximising correlations between the network outputs.  The
structural path matrix specifies which latent factors are connected across
views.

The toolbox retains the familiar high-level methods ``compile``, ``fit``,
``evaluate``, and ``predict`` as Deep LVPM convenience methods.  Internally,
these methods use standard PyTorch training commands: ``model.train()``,
``optimizer.zero_grad()``, a forward pass, loss calculation, ``loss.backward()``,
``optimizer.step()``, ``model.eval()``, and ``torch.no_grad()``.

The documentation explains how to install the toolbox, demonstrates example
applications (MNIST digits, TCGA lung cancer multi-omics, TCGA survival,
Siamese CIFAR-10, and COCO image-text learning), and describes the API for the
core classes and custom layers.
