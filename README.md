
![Alt text](dlvpm_logo_final.png)

# Deep LVPM

Deep Latent Variable Path Modelling (DLVPM) is a method for path/structural equation modelling utilising deep neural networks. The aim of the method is to connect different data types together via sets of orthogonal deep latent variables (DLVs). This work has now been published here: https://www.nature.com/articles/s42256-025-01052-4

# Installation

This package was most recently tested on TensorFlow 2.16.2, which is compatible with Python 3.12.  
We recommend creating a fresh environment using `conda` or `venv` and then installing the package  
**with the appropriate extra** for your hardware:

- **CPU-only** (works on Linux, Windows, and Intel-based Macs): `deep-lvpm[cpu]`
- **GPU (Linux/Windows with NVIDIA)**: `deep-lvpm[gpu]`
- **Apple Silicon (M-series chips: M1/M2/M3/M4)**: `deep-lvpm[apple]`

> **Note:** Intel-based Macs should use the `cpu` extra (or `gpu` if they have an NVIDIA GPU with CUDA support, which is rare on Macs).

conda:

~~~

## conda

```bash
# Create and activate environment
conda create -n myenv python=3.12
conda activate myenv

# CPU install (Linux/Windows/Intel Mac)
pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[cpu]"

# GPU install (Linux/Windows with NVIDIA GPU)
pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[gpu]"

# Apple Silicon install (M-series only)
pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[apple]"

~~~

venv:

~~~

# Create and activate environment
python3 -m venv myenv   # requires Python 3.12+
source myenv/bin/activate  # mac/linux
myenv\Scripts\activate     # windows

# CPU install (Linux/Windows/Intel Mac)
pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[cpu]"

# GPU install (Linux/Windows with NVIDIA GPU)
pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[gpu]"

# Apple Silicon install (M-series only)
pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[apple]"

~~~

BRIEF NOTE: The DLVPM method is currently based on keras2, which uses the tensorflow backend. We are currently updating the package to keras3 which allows the use of tensorflow, pytorch and jax.

