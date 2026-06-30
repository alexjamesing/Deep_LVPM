Installation
============

Deep LVPM targets **native PyTorch**.  The toolbox keeps the familiar
``fit`` / ``evaluate`` / ``predict`` convenience methods, but the implementation
uses PyTorch modules, optimizers, tensors, and checkpoints directly.

We strongly suggest creating a clean conda environment or virtual environment
before installing.  The commands below match the instructions in
:file:`README.md`.

Conda environment
-----------------

To create a conda environment and install the package from PyPI:

.. code-block:: bash

   conda create -n dlvpm-torch python=3.11 -y
   conda activate dlvpm-torch

   pip install deep-lvpm

For NVIDIA CUDA, install the CUDA-enabled PyTorch wheel for your platform first,
then install Deep LVPM:

.. code-block:: bash

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   pip install deep-lvpm

GitHub install
--------------

To install the current default GitHub branch directly:

.. code-block:: bash

   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm"

Editable local install
----------------------

.. code-block:: bash

   git clone https://github.com/alexjamesing/Deep_LVPM.git
   cd Deep_LVPM
   pip install -e ".[tutorials,dev]"

Virtualenv
----------

.. code-block:: bash

   python3 -m venv dlvpm-torch
   source dlvpm-torch/bin/activate           # Windows: dlvpm-torch\Scripts\activate
   pip install deep-lvpm

Verifying the install
---------------------

.. code-block:: bash

   python -c "import torch, deep_lvpm; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"

Additional notes
----------------

* Apple Silicon uses standard PyTorch wheels with MPS support where available.
* CUDA-enabled PyTorch wheels should be installed from the PyTorch index for
  your platform and driver.
* PyTorch checkpoints use ``state_dict`` / ``torch.save``.
* Optional dependency groups include ``tutorials``, ``coco``, ``survival``,
  ``docs``, and ``dev``.  For example: ``pip install "deep-lvpm[survival]"``.
