Siamese CIFAR-10 Tutorial
=========================

This tutorial shows how DLVPM can learn a representation of one data type by
creating two independently augmented views of the same CIFAR-10 image.  The two
views pass through a shared residual PyTorch encoder, and
``StructuralModel(is_siamese=True)`` learns DLVs that align the augmented views.
The learned representation is then evaluated with a linear SVM.

The full runnable script is :mod:`deep_lvpm.tutorial.tutorial_siamese`.

Run it with:

.. code-block:: bash

   python -m deep_lvpm.tutorial.tutorial_siamese

Prerequisites
-------------

This example is heavier than the MNIST tutorial.  It uses CIFAR-10, stochastic
image augmentation, a residual convolutional encoder, and a large default batch
size.  A CUDA or Apple Silicon device is useful, but the script can be forced to
CPU by changing ``REQUESTED_DEVICE``.

1. Configure the run
--------------------

The script keeps the main user-editable settings near the top.  The device
selector accepts ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"`` and raises a
clear error if the requested accelerator is unavailable.

.. code-block:: python

   import random
   from pathlib import Path

   import numpy as np
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.svm import LinearSVC
   from torch.utils.data import DataLoader, Dataset
   from torchvision import datasets

   from deep_lvpm import regularizers as dlvpm_regularizers
   from deep_lvpm.model import StructuralModel

   SEED = 1337
   BATCH_SIZE = 2048
   EPOCHS = 10
   WEIGHT_DECAY = 0
   NDIMS = 512
   LEARNING_RATE = 1e-4
   MAX_GRADIENT_NORM = 1
   REQUESTED_DEVICE = "auto"
   PRINT_AUGMENTATION_DIAGNOSTICS = True
   CIFAR10_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

   NUM_CLASSES = 10
   INPUT_SHAPE = (32, 32, 3)

.. code-block:: python

   def choose_torch_device(requested_device="auto"):
       requested_device = str(requested_device).lower()
       mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

       if requested_device == "auto":
           if torch.cuda.is_available():
               return torch.device("cuda")
           if mps_available:
               return torch.device("mps")
           return torch.device("cpu")

       if requested_device == "cuda" and not torch.cuda.is_available():
           raise RuntimeError("REQUESTED_DEVICE='cuda', but CUDA is not available.")
       if requested_device == "mps" and not mps_available:
           raise RuntimeError("REQUESTED_DEVICE='mps', but Apple Silicon MPS is not available.")

       return torch.device(requested_device)

   DEVICE = choose_torch_device(REQUESTED_DEVICE)
   print(f"Using PyTorch device: {DEVICE}")

2. Load CIFAR-10 and make a validation split
--------------------------------------------

CIFAR-10 is loaded through ``torchvision``.  The script keeps both integer
labels and one-hot labels because the integer labels are convenient for the
linear SVM evaluation at the end.

.. code-block:: python

   train_dataset = datasets.CIFAR10(root=str(CIFAR10_DATA_DIR), train=True, download=True)
   test_dataset = datasets.CIFAR10(root=str(CIFAR10_DATA_DIR), train=False, download=True)
   x_train = train_dataset.data.astype("float32") / 255.0
   y_train_cat = np.asarray(train_dataset.targets, dtype="int64")
   x_test = test_dataset.data.astype("float32") / 255.0
   y_test_cat = np.asarray(test_dataset.targets, dtype="int64")

   y_train = np.eye(NUM_CLASSES, dtype="float32")[y_train_cat]
   y_test = np.eye(NUM_CLASSES, dtype="float32")[y_test_cat]

   random.seed(SEED)
   np.random.seed(SEED)
   torch.manual_seed(SEED)

   VAL_FRACTION = 0.1
   num_train = x_train.shape[0]
   indices = np.arange(num_train)
   rng = np.random.default_rng(SEED)
   rng.shuffle(indices)
   cutoff = int(num_train * (1 - VAL_FRACTION))
   x_tr, x_val = x_train[indices[:cutoff]], x_train[indices[cutoff:]]

3. Create two augmented views per image
---------------------------------------

The Siamese setup is self-supervised: each batch contains two different
augmentations of the same underlying image.  The augmentation function converts
images to channel-first tensors, then applies random crops, resizing, flips, and
occasional grayscale conversion.

.. code-block:: python

   def _to_channel_first(batch: torch.Tensor) -> torch.Tensor:
       if batch.ndim == 4 and batch.shape[-1] == 3:
           batch = batch.permute(0, 3, 1, 2)
       return batch.contiguous()

   def augment(batch: torch.Tensor, training: bool = True) -> torch.Tensor:
       batch = _to_channel_first(batch.float())
       if not training:
           return batch

       cropped_images = []
       for image in batch:
           top = torch.randint(0, 32 - 24 + 1, (1,)).item()
           left = torch.randint(0, 32 - 24 + 1, (1,)).item()
           cropped_images.append(image[:, top:top + 24, left:left + 24])
       x = torch.stack(cropped_images, dim=0)
       x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)

       flip_mask = torch.rand(x.shape[0]) < 0.5
       if torch.any(flip_mask):
           x[flip_mask] = torch.flip(x[flip_mask], dims=(3,))

       gray_mask = torch.rand(x.shape[0], 1, 1, 1) < 0.2
       grayscale_weights = torch.as_tensor(
           [0.2989, 0.5870, 0.1140],
           dtype=x.dtype,
           device=x.device,
       ).view(1, 3, 1, 1)
       grayscale = torch.sum(x * grayscale_weights, dim=1, keepdim=True).repeat(1, 3, 1, 1)
       x = torch.where(gray_mask.to(x.device), grayscale, x)
       return x

The dataset and collate function return ``((view_one, view_two),)`` so DLVPM
sees the two augmented images as two inputs belonging to the same Siamese
measurement model.

.. code-block:: python

   class SiameseViewsDataset(Dataset):
       def __init__(self, x, training=True):
           self.x = torch.as_tensor(x, dtype=torch.float32)
           self.training = bool(training)

       def __len__(self):
           return self.x.shape[0]

       def __getitem__(self, index):
           return self.x[index]

   def collate_siamese_views(batch, training=True):
       batch = torch.stack(batch, dim=0)
       view_one = augment(batch, training=training)
       view_two = augment(batch, training=training)
       return ((view_one, view_two),)

   def make_siamese_views_dataset(x, batch_size=256, shuffle=True, training=True):
       dataset = SiameseViewsDataset(x, training=training)
       return DataLoader(
           dataset,
           batch_size=int(batch_size),
           shuffle=bool(shuffle),
           drop_last=bool(training),
           collate_fn=lambda batch: collate_siamese_views(batch, training=training),
       )

   train_ds = make_siamese_views_dataset(x_tr, batch_size=BATCH_SIZE, shuffle=False, training=True)
   val_ds = make_siamese_views_dataset(x_val, batch_size=BATCH_SIZE, shuffle=False, training=True)

4. Define the shared CIFAR encoder
----------------------------------

The image encoder is a small residual convolutional network.  It exposes
``extract_average_pool_features`` so the tutorial can later compare the final
DLVPM factors with features from the convolutional base before the dense
projection.

.. code-block:: python

   class CIFARResidualBlock(nn.Module):
       def __init__(self, input_channels, output_channels, stride=1):
           super().__init__()
           self.conv1 = nn.Conv2d(
               input_channels,
               output_channels,
               kernel_size=3,
               stride=stride,
               padding=1,
               bias=False,
           )
           self.bn1 = nn.BatchNorm2d(output_channels)
           self.conv2 = nn.Conv2d(
               output_channels,
               output_channels,
               kernel_size=3,
               padding=1,
               bias=False,
           )
           self.bn2 = nn.BatchNorm2d(output_channels)

           if stride != 1 or input_channels != output_channels:
               self.shortcut = nn.Sequential(
                   nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                   nn.BatchNorm2d(output_channels),
               )
           else:
               self.shortcut = nn.Identity()

       def forward(self, inputs):
           x = F.relu(self.bn1(self.conv1(inputs)))
           x = self.bn2(self.conv2(x))
           x = x + self.shortcut(inputs)
           return F.relu(x)

The complete ``CIFARImageModel`` stacks those residual blocks, applies global
average pooling, and then maps the pooled features to the DLVPM input width.

.. code-block:: python

   class CIFARImageModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.stem = nn.Sequential(
               nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
               nn.BatchNorm2d(64),
               nn.ReLU(inplace=True),
           )
           self.layer1 = CIFARResidualBlock(64, 64)
           self.layer2 = CIFARResidualBlock(64, 128, stride=2)
           self.layer3 = CIFARResidualBlock(128, 256, stride=2)
           self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
           self.dense1 = nn.Linear(256, NDIMS)
           self.n_inputs = 1

       def extract_average_pool_features(self, inputs):
           x = self.stem(inputs)
           x = self.layer1(x)
           x = self.layer2(x)
           x = self.layer3(x)
           x = self.avg_pool(x)
           return torch.flatten(x, start_dim=1)

       def forward(self, inputs):
           x = self.extract_average_pool_features(inputs)
           x = self.dense1(x)
           return x

5. Build and train the Siamese StructuralModel
----------------------------------------------

``is_siamese=True`` tells ``StructuralModel`` to wrap the first encoder once
and reuse it for both views.  The path matrix connects view 1 to view 2 and
view 2 to view 1, because both views are augmented versions of the same image.

.. code-block:: python

   CIFAR_image_model = CIFARImageModel()

   model_list = [CIFAR_image_model, CIFAR_image_model]
   adjacency = np.array([[0, 1], [1, 0]], dtype="float32")
   regularizers = [
       dlvpm_regularizers.l2(WEIGHT_DECAY),
       dlvpm_regularizers.l2(WEIGHT_DECAY),
   ]

   dlvpm_model = StructuralModel(
       adjacency,
       model_list,
       regularizers,
       x_train.shape[0],
       NDIMS,
       orthogonalization="zca",
       train_DLV=True,
       is_siamese=True,
       diag_offset=1e-12,
       device=DEVICE,
   )

The script uses a small subclass of Adam to apply Keras-style gradient
``clipnorm`` inside ``optimizer.step``.  The same optimizer object is supplied
for both views because the weights are shared.

.. code-block:: python

   class ClippedAdam(torch.optim.Adam):
       def __init__(self, params, clipnorm=None, **kwargs):
           super().__init__(params, **kwargs)
           self.clipnorm = clipnorm

       def step(self, closure=None):
           if self.clipnorm is not None:
               parameters_with_grad = []
               seen_parameter_ids = set()
               for param_group in self.param_groups:
                   for parameter in param_group["params"]:
                       if parameter.grad is None or id(parameter) in seen_parameter_ids:
                           continue
                       parameters_with_grad.append(parameter)
                       seen_parameter_ids.add(id(parameter))
               if parameters_with_grad:
                   nn.utils.clip_grad_norm_(parameters_with_grad, max_norm=float(self.clipnorm))

           return super().step(closure=closure)

   shared_optimizer = ClippedAdam(
       dlvpm_model.model_list[0].parameters(),
       lr=LEARNING_RATE,
       eps=1e-7,
       clipnorm=MAX_GRADIENT_NORM,
   )
   optimizers = [shared_optimizer, shared_optimizer]
   dlvpm_model.compile(optimizers)

   dlvpm_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=True)

6. Export features for linear evaluation
----------------------------------------

After training, the tutorial exports two feature sets: final DLVPM factors and
average-pooled convolutional features.  This tests whether the DLVPM projection
and the convolutional base are useful for downstream classification.

.. code-block:: python

   def predict_view_model(view_model: nn.Module, x, batch_size=32) -> np.ndarray:
       view_model.eval()
       x_tensor = _to_channel_first(torch.as_tensor(x, dtype=torch.float32))
       loader = DataLoader(x_tensor, batch_size=batch_size, shuffle=False)
       outputs = []
       with torch.no_grad():
           for batch in loader:
               batch = batch.to(dlvpm_model.device)
               outputs.append(view_model(batch).detach().cpu())
       return torch.cat(outputs, dim=0).numpy()

   def predict_average_pool_features(view_model: nn.Module, x, batch_size=32) -> np.ndarray:
       view_model.eval()
       encoder = getattr(view_model, "encoder", view_model)

       if not hasattr(encoder, "extract_average_pool_features"):
           raise AttributeError("The image encoder does not expose extract_average_pool_features().")

       x_tensor = _to_channel_first(torch.as_tensor(x, dtype=torch.float32))
       loader = DataLoader(x_tensor, batch_size=batch_size, shuffle=False)
       outputs = []
       with torch.no_grad():
           for batch in loader:
               batch = batch.to(dlvpm_model.device)
               features = encoder.extract_average_pool_features(batch)
               outputs.append(features.detach().cpu())
       return torch.cat(outputs, dim=0).numpy()

7. Run a linear SVM probe
-------------------------

The linear probe standardizes each feature set and fits a ``LinearSVC`` on the
CIFAR-10 training labels.  This is not part of DLVPM training; it is a
downstream check of the learned representation.

.. code-block:: python

   def evaluate_linear_svm(feature_name, train_features, test_features):
       print(f"\nLinear SVM evaluation on {feature_name}")
       print(f"Train feature shape: {train_features.shape}")
       print(f"Test  feature shape: {test_features.shape}")

       svm_clf = Pipeline(
           [
               ("scaler", StandardScaler(with_mean=True)),
               ("svm", LinearSVC(C=1.0, max_iter=10000, random_state=42)),
           ]
       )
       svm_clf.fit(train_features, y_train_cat)
       predictions = svm_clf.predict(test_features)
       accuracy = accuracy_score(y_test_cat, predictions)

       print(f"SVM accuracy on CIFAR-10 test set: {accuracy:.4f}\n")
       print(classification_report(y_test_cat, predictions, digits=4))
       print(confusion_matrix(y_test_cat, predictions))

       return {"accuracy": accuracy, "predictions": predictions}

   image_model = dlvpm_model.model_list[0]
   train_dlvs = predict_view_model(image_model, x_train, batch_size=32)
   test_dlvs = predict_view_model(image_model, x_test, batch_size=32)
   train_average_pool_features = predict_average_pool_features(image_model, x_train, batch_size=32)
   test_average_pool_features = predict_average_pool_features(image_model, x_test, batch_size=32)

   test_ds = make_siamese_views_dataset(x_test, batch_size=BATCH_SIZE, shuffle=False, training=True)
   test_metrics = dlvpm_model.evaluate(test_ds)
   print("Test DLVPM metrics:", test_metrics)

   linear_evaluation_results = {}
   linear_evaluation_results["final_dlvpm_factors"] = evaluate_linear_svm(
       "final DLVPM factors",
       train_dlvs,
       test_dlvs,
   )
   linear_evaluation_results["average_pool_features"] = evaluate_linear_svm(
       "last average pooling layer features",
       train_average_pool_features,
       test_average_pool_features,
   )

Summary
-------

The Siamese tutorial uses DLVPM as a self-supervised representation learner:
build two augmented views of each image, share one PyTorch encoder across both
views, train the shared encoder through ``StructuralModel``, then evaluate the
learned representation with a simple linear classifier.
