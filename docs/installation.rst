Installation
============

Deep LVPM now targets **Keras 3** (multi-backend).  You install *one* backend (TensorFlow or PyTorch) via a pip
extra, then force the backend if necessary with ``KERAS_BACKEND``.  The current release has been exercised on
Python 3.10–3.12; Python 3.12 is recommended for fresh environments.

We strongly suggest creating a clean conda environment (or venv/micromamba etc) before installing.  The commands below match
the instructions in :file:`README.md`.

Choose a backend
----------------

DLVPM provides extras so you only pull the runtime you need.  Pick exactly one of the following:

* **TensorFlow**: ``tf-cpu`` (portable CPU build), ``tf-gpu`` (Linux + NVIDIA CUDA wheel), ``tf-apple`` (Apple Silicon).
* **PyTorch**: ``torch-cpu`` (CPU builds), ``torch-apple`` (Apple Silicon), ``torch-gpu`` (install CUDA PyTorch first, then use the empty extra to avoid CPU wheels).

.. note::

   DLVPM is now compatible with **both TensorFlow and PyTorch backends** through Keras 3.  Set ``KERAS_BACKEND=tensorflow``
   or ``KERAS_BACKEND=torch`` if you have multiple runtimes installed and want to force a specific backend.

Conda environment
-----------------

To create a conda environment and install the package from the ``keras3`` branch:

.. code-block:: bash

   conda create -n dlvpm-k3 python=3.12 -y
   conda activate dlvpm-k3

   # TensorFlow backends -------------------------------------------------
   # CPU (Linux/Windows/Intel Mac)
   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git@keras3#egg=deep-lvpm[tf-cpu]"

   # NVIDIA GPU (Linux, bundled CUDA wheel)
   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git@keras3#egg=deep-lvpm[tf-gpu]"

   # Apple Silicon (M-series)
   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git@keras3#egg=deep-lvpm[tf-apple]"

   # PyTorch backends ----------------------------------------------------
   # CPU (Linux/Windows/macOS Intel)
   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git@keras3#egg=deep-lvpm[torch-cpu]"

   # Apple Silicon (M-series)
   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git@keras3#egg=deep-lvpm[torch-apple]"

   # NVIDIA GPU (CUDA) – install CUDA-enabled PyTorch first, then:
   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git@keras3#egg=deep-lvpm[torch-gpu]"

Virtualenv
----------

To use Python's ``venv`` module instead of conda:

.. code-block:: bash

   python3 -m venv dlvpm-k3
   source dlvpm-k3/bin/activate           # Windows: dlvpm-k3\Scripts\activate

   # Use the same pip install commands shown in the conda section above,
   # selecting exactly one backend extra (tf-* or torch-*).

Verifying the backend
---------------------

After installing, confirm which backend Keras selected:

.. code-block:: bash

   python -c "import keras, os; print('KERAS_BACKEND=', os.getenv('KERAS_BACKEND')); print('Detected backend:', keras.backend.backend())"

Set the backend explicitly if you have both runtimes available:

.. code-block:: bash

   export KERAS_BACKEND=tensorflow   # or: torch

Additional notes
----------------

* ``tf-gpu`` is Linux-only and installs ``tensorflow[and-cuda]`` (no extra CUDA toolkit needed).
* ``torch-gpu`` intentionally does **not** install PyTorch; install the CUDA wheel from ``pytorch.org`` first, then use the extra.
* Tutorials in :mod:`deep_lvpm.tutorial` default to TensorFlow but can run on PyTorch by setting ``KERAS_BACKEND=torch`` before launching.
