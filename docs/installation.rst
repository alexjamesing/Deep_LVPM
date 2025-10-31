Installation
============

The Keras 3 release of Deep LVPM supports multiple backends via installation **extras**.  
Pick the extra that matches your preferred backend:

* ``[tf-cpu]`` – TensorFlow CPU wheels (Linux, Windows, Intel macOS).
* ``[tf-gpu]`` – TensorFlow with bundled CUDA/cuDNN wheels (Linux with NVIDIA GPU).
* ``[tf-apple]`` – TensorFlow + Metal plugins for Apple Silicon (macOS arm64, Python 3.10–3.11).
* ``[torch-cpu]`` – PyTorch CPU wheels (Linux, Windows, Intel macOS, Apple Silicon).
* ``[torch-apple]`` – PyTorch MPS wheels (Apple Silicon).
* ``[torch-gpu]`` – Empty extra intended for CUDA-enabled PyTorch installs (preinstall from `pytorch.org <https://pytorch.org>`_ first).

You can switch backends at runtime by exporting the ``KERAS_BACKEND`` environment variable (``tensorflow`` or ``torch``).  Unless otherwise noted, the tutorials default to TensorFlow.

Conda environment
-----------------

.. code-block:: bash

   # create a new conda environment (Python 3.11 works across backends)
   conda create -n dlvpm-k3 python=3.11 -y
   conda activate dlvpm-k3

   # TensorFlow CPU
   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[tf-cpu]"

   # TensorFlow + CUDA (Linux, NVIDIA GPU)
   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[tf-gpu]"

   # TensorFlow on Apple Silicon (macOS arm64)
   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[tf-apple]"

   # PyTorch CPU
   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[torch-cpu]"

   # PyTorch Apple Silicon (uses MPS acceleration)
   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[torch-apple]"

   # PyTorch CUDA (install the CUDA wheel first, then add Deep LVPM)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[torch-gpu]"

Virtualenv
----------

.. code-block:: bash

   python3 -m venv dlvpm-k3
   source dlvpm-k3/bin/activate
   # Windows: dlvpm-k3\Scripts\activate

   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[tf-cpu]"
   # or replace [tf-cpu] with another extra from the list above

Verifying the backend
---------------------

After installation, confirm which backend Keras detected:

.. code-block:: bash

   python -c "import keras, os; print('KERAS_BACKEND=', os.getenv('KERAS_BACKEND')); print('Detected:', keras.backend.backend())"

If you need to switch:

.. code-block:: bash

   export KERAS_BACKEND=tensorflow  # or: torch

Notes
-----

* ``[tf-gpu]`` installs ``tensorflow[and-cuda]==2.20.0``.  The wheel bundles the CUDA runtime and is only available on Linux; it will fall back to CPU if no compatible GPU is present.
* ``[tf-apple]`` installs ``tensorflow-macos==2.16.2`` and ``tensorflow-metal==1.2.0``.  Apple currently publishes wheels for Python 3.10–3.11—use one of those versions when targeting Apple Silicon.
* ``[torch-gpu]`` intentionally has no dependencies to avoid pulling CPU wheels from PyPI.  Always install the CUDA builds from the PyTorch index first, then add the Deep LVPM extra.
