Installation
============

The Deep LVPM toolbox is distributed as a Python package that depends on Keras and TensorFlow.  
The most recent release has been tested with **TensorFlow 2.16.2** and **Python 3.12**.

We recommend creating a fresh virtual environment using either conda or Python's built-in ``venv`` module 
before installing the package. Deep LVPM now supports installation **extras** so you can choose the 
appropriate TensorFlow build for your hardware.

.. note::

   Intel-based Macs should use the ``[cpu]`` extra.  
   The ``[gpu]`` extra is only useful if you have an NVIDIA GPU with supported CUDA drivers.  
   The ``[apple]`` extra is only for Apple Silicon M-series chips.

Conda environment
-----------------

To create a conda environment and install the package from GitHub:

.. code-block:: bash

   # create a new conda environment with Python 3.12
   conda create -n myenv python=3.12
   conda activate myenv

   # CPU-only install (Linux/Windows/Intel Mac)
   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[cpu]"

   # GPU install (Linux/Windows with NVIDIA GPU)
   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[gpu]"

   # Apple Silicon install (M-series only)
   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[apple]"

Virtualenv
----------

To use Python's ``venv`` module instead of conda:

.. code-block:: bash

   # create a new virtual environment (requires Python ≥3.12)
   python3 -m venv myenv

   # activate the environment on macOS/Linux
   source myenv/bin/activate

   # activate the environment on Windows
   # myenv\Scripts\activate

   # CPU-only install (Linux/Windows/Intel Mac)
   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[cpu]"

   # GPU install (Linux/Windows with NVIDIA GPU)
   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[gpu]"

   # Apple Silicon install (M-series only)
   pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[apple]"

.. warning::

   The ``[gpu]`` extra installs ``tensorflow[and-cuda]==2.16.2`` which includes the full CUDA and cuDNN runtime.
   This is a large download and will fall back to CPU if no GPU is detected.  
   The ``[apple]`` extra installs ``tensorflow-macos`` and ``tensorflow-metal`` for GPU acceleration 
   on Apple Silicon.

   The current version of DLVPM is built on **Keras 2** (with a TensorFlow backend).  We are actively updating the package to **Keras 3**, which will enable compatibility with multiple backends (TensorFlow, PyTorch and JAX).  Users should ensure that they install the correct version of TensorFlow as specified above to avoid compatibility issues.