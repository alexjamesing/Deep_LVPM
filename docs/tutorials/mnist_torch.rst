MNIST Tutorial (PyTorch backend)
================================

This companion tutorial mirrors ``deep_lvpm/tutorial/tutorial_mnist_torch.py``.  We keep the measurement models in pure :mod:`torch.nn` modules, wrap them so they integrate with Keras 3 running on the torch backend, and then train the same two-view StructuralModel used in the TensorFlow walkthrough.

Prerequisites
-------------

* Install :mod:`deep_lvpm` with the PyTorch extras (``pip install deep-lvpm[torch]``).
* Set ``KERAS_BACKEND=torch`` or export the environment variable before running the script.
* Ensure PyTorch is installed with either CPU or CUDA support depending on your hardware.

Step 1 – Load and preprocess MNIST
----------------------------------

The data preparation is identical to the TensorFlow version so both tutorials can be compared like-for-like.

.. code-block:: python

   import os
   import numpy as np
   import torch
   import torch.nn as nn

   import keras
   from keras.optimizers import Adam
   from sklearn.manifold import TSNE

   from deep_lvpm.models.StructuralModel import StructuralModel

   os.environ.setdefault("KERAS_BACKEND", "torch")

   num_classes = 10

   print("Loading MNIST data...")
   (x_train, y_train_cat), (x_test, y_test_cat) = keras.datasets.mnist.load_data()

   print("Preprocessing images and labels...")
   x_train = x_train.astype("float32") / 255.0
   x_test = x_test.astype("float32") / 255.0
   x_train = np.expand_dims(x_train, axis=-1)
   x_test = np.expand_dims(x_test, axis=-1)

   y_train = keras.utils.to_categorical(y_train_cat, num_classes)
   y_test = keras.utils.to_categorical(y_test_cat, num_classes)

   data_train = [x_train, y_train]
   data_test = [x_test, y_test]

Step 2 – Author pure PyTorch measurement modules
------------------------------------------------

We recreate the image CNN and label identity encoder using :class:`torch.nn.Sequential`.  ``TorchMeasurementModel`` (defined in the script) handles device transfer and appends the FactorLayer without using any Keras layers inside the measurement modules themselves.

.. code-block:: python

   from deep_lvpm.layers.FactorLayer import FactorLayer

   class TorchMeasurementModel(keras.Model):
       """Wrap a torch.nn.Module so it can plug into StructuralModel."""

       def __init__(self, torch_module, tot_num, ndims, momentum, epsilon, name):
           super().__init__(name=name)
           self.torch_module = torch_module
           self.factor_layer = FactorLayer(
               kernel_regularizer=None,
               tot_num=tot_num,
               ndims=ndims,
               momentum=momentum,
               epsilon=epsilon,
           )
           self._current_device = None

       def _prepare_tensor(self, inputs):
           tensor = torch.as_tensor(inputs, dtype=torch.float32)
           if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
               tensor = tensor.permute(0, 3, 1, 2).contiguous()
           return tensor

       def call(self, inputs, training=False):
           tensor = self._prepare_tensor(inputs)
           device = tensor.device
           if self._current_device != device:
               self.torch_module.to(device)
               self._current_device = device
           self.torch_module.train(training)
           features = self.torch_module(tensor)
           if features.ndim > 2:
               features = torch.flatten(features, start_dim=1)
           return self.factor_layer(features, training=training)

   ndims = 9
   momentum = 0.95
   epsilon = 1e-4
   tot_num = x_train.shape[0]

   image_torch_module = nn.Sequential(
       nn.Conv2d(1, 32, kernel_size=3),
       nn.ReLU(inplace=True),
       nn.MaxPool2d(kernel_size=2),
       nn.Conv2d(32, 64, kernel_size=3),
       nn.ReLU(inplace=True),
       nn.MaxPool2d(kernel_size=2),
       nn.Flatten(),
       nn.Linear(64 * 5 * 5, 128),
       nn.ReLU(inplace=True),
       nn.Dropout(p=0.5),
   )

   label_torch_module = nn.Sequential(nn.Identity())

   image_model = TorchMeasurementModel(
       torch_module=image_torch_module,
       tot_num=tot_num,
       ndims=ndims,
       momentum=momentum,
       epsilon=epsilon,
       name="mnist_image_encoder",
   )

   label_model = TorchMeasurementModel(
       torch_module=label_torch_module,
       tot_num=tot_num,
       ndims=ndims,
       momentum=momentum,
       epsilon=epsilon,
       name="mnist_label_encoder",
   )

Step 3 – Assemble and train the StructuralModel
-----------------------------------------------

Because the Torch measurement models already include FactorLayer, pass ``run_from_config=True`` so the StructuralModel refrains from adding another one.  Optimisation mirrors the TensorFlow script with one Adam per view.

.. code-block:: python

   adjacency = np.array([[0, 1], [1, 0]], dtype="float32")

   structural_model = StructuralModel(
       Path=adjacency,
       model_list=[image_model, label_model],
       regularizer_list=[None, None],
       tot_num=tot_num,
       ndims=ndims,
       orthogonalization="Moore-Penrose",
       momentum=momentum,
       epsilon=epsilon,
       train_DLV=False,
       run_from_config=True,
   )

   optimisers = [Adam(learning_rate=1e-4), Adam(learning_rate=1e-4)]
   structural_model.compile(optimizer=optimisers)

   history = structural_model.fit(
       data_train,
       batch_size=256,
       epochs=20,
       verbose=True,
       validation_split=0.1,
   )

Step 4 – Evaluate metrics and latent variables
----------------------------------------------

The evaluate/predict pattern is identical to the TensorFlow backend, giving you a straightforward way to compare outputs.

.. code-block:: python

   metrics = structural_model.evaluate(data_test, verbose=False)
   metrics = {name: float(value) for name, value in metrics.items()}
   print("Test metrics:", metrics)

   latent = structural_model.predict(data_test, verbose=False)
   image_latent = structural_model.model_list[0].predict(data_test[0], verbose=False)

   tsne = TSNE(n_components=2, random_state=42)
   rng = np.random.default_rng(42)
   sample_indices = rng.choice(
       image_latent.shape[0],
       size=min(200, image_latent.shape[0]),
       replace=False,
   )
   tsne_projection = tsne.fit_transform(image_latent[sample_indices])

   print("Latent tensor shape:", latent.shape)
   print("t-SNE projection shape:", tsne_projection.shape)
   print("Training history keys:", list(history.history))

Next steps
----------

* Toggle ``torch.backends.cudnn.benchmark`` or pin models to specific devices when running on GPU.
* Experiment with larger PyTorch encoders or additional regularisation (e.g., dropout placement) and observe how the StructuralModel metrics respond.
* Compare metrics against the TensorFlow backend tutorial to ensure parity across backends for your use case.
