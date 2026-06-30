MNIST Tutorial
==============

This tutorial demonstrates the smallest complete DLVPM workflow.  The two data
views are MNIST images and one-hot digit labels.  The image view is encoded by a
small convolutional PyTorch model, the label view is passed through unchanged,
and ``StructuralModel`` learns deep latent variables (DLVs) that are correlated
between the two views.

The full runnable script is :mod:`deep_lvpm.tutorial.tutorial_mnist`.

Run it with:

.. code-block:: bash

   python -m deep_lvpm.tutorial.tutorial_mnist

Prerequisites
-------------

Install :mod:`deep_lvpm` as described in :doc:`/installation`.  This example
uses ``torch``, ``torchvision``, ``numpy``, ``scikit-learn``, and
``matplotlib``.  It can run on CPU, although the first run will download MNIST
through ``torchvision``.

1. Import packages and load MNIST
---------------------------------

The current branch is native PyTorch, so the tutorial imports ``torch`` and
``torch.nn`` directly.  MNIST comes from ``torchvision.datasets.MNIST``.  Images
are converted to NumPy arrays because the DLVPM convenience methods accept
NumPy arrays and convert them to tensors internally.

.. code-block:: python

   import numpy as np
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from torchvision import datasets
   from sklearn.manifold import TSNE
   import matplotlib.pyplot as plt

   from deep_lvpm.model import StructuralModel

   num_classes = 10
   input_shape = (1, 28, 28)

   train_dataset = datasets.MNIST(root="data", train=True, download=True)
   test_dataset = datasets.MNIST(root="data", train=False, download=True)
   x_train = train_dataset.data.numpy()
   y_train_cat = np.asarray(train_dataset.targets, dtype="int64")
   x_test = test_dataset.data.numpy()
   y_test_cat = np.asarray(test_dataset.targets, dtype="int64")

The image tensor layout is channel-first, matching PyTorch convolution layers:
``(n_samples, 1, 28, 28)``.  Labels are converted to one-hot rows so they can be
used as a second view.

.. code-block:: python

   x_train = x_train.astype("float32") / 255
   x_test = x_test.astype("float32") / 255
   x_train = np.expand_dims(x_train, 1)
   x_test = np.expand_dims(x_test, 1)

   print("x_train shape:", x_train.shape)
   print(x_train.shape[0], "train samples")
   print(x_test.shape[0], "test samples")

   y_train = np.eye(num_classes, dtype="float32")[y_train_cat]
   y_test = np.eye(num_classes, dtype="float32")[y_test_cat]

   data_train_list = [x_train, y_train]
   data_test_list = [x_test, y_test]

2. Define measurement models
----------------------------

DLVPM needs one measurement model per data view.  Measurement models are plain
``torch.nn.Module`` instances.  ``StructuralModel`` appends the final DLVPM
projection layer to each module when it is constructed.

The image model is a small convolutional encoder.  It returns feature vectors,
not class probabilities, because DLVPM is learning latent variables rather than
training a classifier.

.. code-block:: python

   class MNISTImageModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
           self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
           self.flatten = nn.Flatten()
           self.dense = nn.Linear(64 * 5 * 5, 512)
           self.dropout = nn.Dropout(0.1)
           self.n_inputs = 1

       def forward(self, inputs):
           x = F.relu(self.conv1(inputs))
           x = F.max_pool2d(x, kernel_size=(2, 2))
           x = F.relu(self.conv2(x))
           x = F.max_pool2d(x, kernel_size=(2, 2))
           x = self.flatten(x)
           x = F.relu(self.dense(x))
           x = self.dropout(x)
           return x

The label model is intentionally simple.  The one-hot labels are already a
compact view, so the model just returns its input.

.. code-block:: python

   class MNISTLabelModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.n_inputs = 1

       def forward(self, inputs):
           return inputs

   MNIST_image_model = MNISTImageModel()
   MNIST_label_model = MNISTLabelModel()
   model_list = [MNIST_image_model, MNIST_label_model]

3. Define the path matrix
-------------------------

The path matrix tells DLVPM which views should be connected.  This two-view
example uses a symmetric matrix: images are linked to labels, and labels are
linked back to images.

.. code-block:: python

   Path = np.array(
       [
           [0, 1],
           [1, 0],
       ],
       dtype="float32",
   )

4. Build and compile the model
------------------------------

The model learns nine DLVs per view.  ``tot_num`` is the number of training
samples and is used internally by the projection layers.  ``orthogonalization``
is set to ``"zca"`` so the learned DLVs are decorrelated within each view.

.. code-block:: python

   regularizer_list = [None, None]

   ndims = 9
   tot_num = x_train.shape[0]
   batch_size = 256
   epochs = 20

   DLVPM_Model = StructuralModel(
       Path,
       model_list,
       regularizer_list,
       tot_num,
       ndims,
       orthogonalization="zca",
       train_DLV=False,
   )

Optimizers are ordinary PyTorch optimizers.  Create them from
``DLVPM_Model.model_list`` rather than the original encoders, because
``StructuralModel`` has wrapped each encoder with its DLVPM projection layer.

.. code-block:: python

   optimizer_list = [
       torch.optim.Adam(model.parameters(), lr=1e-5)
       for model in DLVPM_Model.model_list
   ]

   DLVPM_Model.compile(optimizer=optimizer_list)

5. Train and evaluate
---------------------

The high-level training call mirrors the old toolbox style, but the
implementation is PyTorch internally.  The returned history contains
``total_loss``, ``cross_metric``, ``mse_loss``, and ``redundancy`` values.

.. code-block:: python

   DLVPM_Model.fit(
       data_train_list,
       batch_size=batch_size,
       epochs=epochs,
       verbose=True,
       validation_split=0.1,
   )

   metrics = DLVPM_Model.evaluate(data_test_list)
   print(metrics)

6. Inspect the latent space
---------------------------

``predict`` returns a NumPy array with shape ``(n_samples, ndims, n_views)``.
The final dimension indexes the image and label views.  The example checks the
cross-view correlation for the first DLV and then visualises the image DLVs
with t-SNE.

.. code-block:: python

   DLVs = DLVPM_Model.predict(data_test_list)
   Cmat1 = np.corrcoef(DLVs[:, 0, :].T)
   print(Cmat1)

   image_DLVs = (
       DLVPM_Model.model_list[0](
           torch.as_tensor(data_test_list[0], dtype=torch.float32)
       )
       .detach()
       .cpu()
       .numpy()
   )

   random_indices = np.random.choice(image_DLVs.shape[0], size=100, replace=False)
   image_DLVs_plot = image_DLVs[random_indices, :]
   y_test_plot = y_test[random_indices, :]

   tsne = TSNE(n_components=2, random_state=42)
   tsne_results = tsne.fit_transform(image_DLVs_plot)

   plt.figure(figsize=(12, 8))
   for i in range(y_test_plot.shape[1]):
       points = tsne_results[y_test_plot[:, i] == 1]
       plt.scatter(points[:, 0], points[:, 1], label=f"Category {i + 1}")

   plt.title("t-SNE projection of the dataset")
   plt.legend()

Summary
-------

This tutorial shows the basic DLVPM pattern: prepare aligned data views, define
one ``torch.nn.Module`` per view, connect the views with a path matrix, compile
with PyTorch optimizers, train with ``fit``, and inspect the learned DLVs with
``predict``.
