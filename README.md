
<p align="center">
  <img src="dlvpm_logo_final.png" alt="Deep LVPM logo" width="35%">
</p>

Deep Latent Variable Path Modelling (DLVPM) is a method for path/structural equation modelling utilising deep neural networks. The aim of the method is to connect different data types together via sets of orthogonal deep latent variables (DLVs). Full documentation for this package can be found here: https://deep-lvpm.readthedocs.io/en/latest/. This work has now been published here: https://www.nature.com/articles/s42256-025-01052-4. 

Backend-agnostic with Keras 3: DLVPM is implemented on Keras 3 and runs backend-agnostically on either TensorFlow or PyTorch; select your preferred backend by installing the corresponding extra (e.g., tf-cpu, tf-gpu, tf-apple, torch-cpu, torch-apple, or preinstall CUDA PyTorch for torch-gpu) and, if needed, set KERAS_BACKEND=tensorflow or KERAS_BACKEND=torch. The high-level Keras API (model.fit, model.evaluate, etc.) is unchanged across backends.

If you find this project valuable, please consider giving it a star on GitHub.  Your support helps others discover the project and motivates continued development!

![Chord animation](chord_animation.gif)

The gif above shows a plot of model training in progress on a three factor DLVPM model linking different omics and Imaging data types from Lung Cancer patients. The dataset this model is trained on is included in this package.

# Installing deep-lvpm (keras 3 / multi-backend) — keras3 branch

> **TL;DR:** This branch uses **Keras 3** (multi-backend). You must install **one backend** (TensorFlow **or** PyTorch) via the provided extras. Then install from the **`keras3`** branch URL.

---

## Choose a backend

### TensorFlow (default)
- CPU: `tf-cpu`
- NVIDIA GPU on **Linux** (bundled CUDA wheels): `tf-gpu`
- Apple Silicon (M-series): `tf-apple`

### PyTorch
- CPU (Linux/Windows/macOS Intel): `torch-cpu`
- Apple Silicon (M-series): `torch-apple`
- NVIDIA GPU: `torch-gpu` *(empty extra; preinstall CUDA wheels from pytorch.org first)*


### Python
- Python `>= 3.11` 
---

## Conda (recommended)

Create and activate an environment:

```bash
conda create -n dlvpm-k3 python=3.11 -y
conda activate dlvpm-k3
```

### TensorFlow backends

**CPU (Linux/Windows/Intel Mac):**
```bash
pip install "git+https://github.com/alexjamesing/Deep_LVPM.git@keras3#egg=deep-lvpm[tf-cpu]"
```

**NVIDIA GPU (Linux, bundled CUDA):**
```bash
pip install "git+https://github.com/alexjamesing/Deep_LVPM.git@keras3#egg=deep-lvpm[tf-gpu]"
```

**Apple Silicon (M-series):**
```bash
pip install "git+https://github.com/alexjamesing/Deep_LVPM.git@keras3#egg=deep-lvpm[tf-apple]"
```

### PyTorch backends

**CPU (Linux/Windows/macOS Intel):**
```bash
pip install "git+https://github.com/alexjamesing/Deep_LVPM.git@keras3#egg=deep-lvpm[torch-cpu]"
```

**Apple Silicon (M-series):**
```bash
pip install "git+https://github.com/alexjamesing/Deep_LVPM.git@keras3#egg=deep-lvpm[torch-apple]"
```

**NVIDIA GPU (CUDA):**
```bash
# 1) Install CUDA-enabled PyTorch from https://pytorch.org (per your CUDA/driver), e.g.:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# 2) Then install deep-lvpm without pulling CPU wheels:
pip install "git+https://github.com/alexjamesing/Deep_LVPM.git@keras3#egg=deep-lvpm[torch-gpu]"
```

---

## venv

Create and activate a virtual environment:

```bash
python3 -m venv dlvpm-k3
source dlvpm-k3/bin/activate   # Windows: dlvpm-k3\Scripts\activate
```
Then choose **one** of the same install lines as above (e.g., `...[tf-cpu]`, `...[tf-gpu]`, etc.).

---

## Verifying the backend

```bash
python -c "import keras, os; print('KERAS_BACKEND=', os.getenv('KERAS_BACKEND')); print('Detected backend:', keras.backend.backend())"
```

If needed, set the backend explicitly and re-run:
```bash
export KERAS_BACKEND=tensorflow   # or: torch
```


## Notes

- `tf-gpu` is **Linux-only** and uses `tensorflow[and-cuda]` (no separate CUDA toolkit install required).
- `torch-gpu` extra is intentionally empty to avoid pulling CPU wheels from PyPI; always install CUDA-enabled PyTorch first.
- If multiple backends are installed, Keras will pick one; use `KERAS_BACKEND` to force your choice.
- The MS COCO tutorial dependencies are now installed by default; for example:
  `pip install "git+https://github.com/alexjamesing/Deep_LVPM.git@keras3#egg=deep-lvpm[tf-cpu]"`

## Tutorials & Metrics

Four turnkey tutorials ship with the toolbox and default to the TensorFlow backend. Launch them with:

- `python -m deep_lvpm.tutorial.tutorial_mnist_tf` – associate MNIST images with labels and visualise the latent space.
- `python -m deep_lvpm.tutorial.tutorial_tcga_tf` – integrate five TCGA lung cancer modalities using residual encoders.
- `python -m deep_lvpm.tutorial.tutorial_siamese_tf` – train a Siamese encoder on CIFAR-10 and probe the embeddings linearly.
- `python -m deep_lvpm.tutorial.tutorial_coco_tf` – train an image-text model on MS COCO with EfficientNetB0 + Transformer and benchmark true five-caption retrieval for DLVPM/CLIP/VICReg.

All tutorials report the expanded `StructuralModel.evaluate` metrics (`total_loss`, `cross_metric`, `mse_loss`, and `redundancy`) introduced in this release so you can monitor both cross-view alignment and within-view redundancy.
