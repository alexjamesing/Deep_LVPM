
<p align="center">
  <img src="https://raw.githubusercontent.com/alexjamesing/Deep_LVPM/master/dlvpm_logo_final.png" alt="Deep LVPM logo" width="35%">
</p>

[![Tests](https://github.com/alexjamesing/Deep_LVPM/actions/workflows/install-matrix.yaml/badge.svg)](https://github.com/alexjamesing/Deep_LVPM/actions/workflows/install-matrix.yaml)
[![Documentation](https://readthedocs.org/projects/deep-lvpm/badge/?version=master)](https://deep-lvpm.readthedocs.io/en/master/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs42256--025--01052--4-blue)](https://doi.org/10.1038/s42256-025-01052-4)

# Deep LVPM

Deep LVPM is a PyTorch toolbox for multimodal representation learning and Deep Latent Variable Path Modelling.

Deep Latent Variable Path Modelling (DLVPM) is a method for path/structural equation modelling using deep neural networks. It connects different data types through sets of orthogonal deep latent variables (DLVs), and can also be used in a Siamese configuration to learn representations of a single data type. Full documentation is available at [deep-lvpm.readthedocs.io](https://deep-lvpm.readthedocs.io/en/master/), and the method is described in the [Nature Machine Intelligence paper](https://doi.org/10.1038/s42256-025-01052-4).

DLVPM is implemented directly in PyTorch. The high-level toolbox API (`model.fit`, `model.evaluate`, `model.predict`, etc.) is retained for convenience, but these methods use ordinary PyTorch commands internally (`model.train()`, forward passes, loss calculation, `loss.backward()`, `optimizer.step()`, `model.eval()`, and `torch.no_grad()`).

## Branches And Legacy Versions

The current default branch is the native PyTorch version of the toolbox. Earlier versions remain available for reproducibility and comparison:

- [`keras3`](https://github.com/alexjamesing/Deep_LVPM/tree/keras3): Keras 3 version of the toolbox, compatible with both TensorFlow and PyTorch backends.
- [`keras2`](https://github.com/alexjamesing/Deep_LVPM/tree/keras2): publication-era version associated with the original DLVPM paper, [Integrating multimodal cancer data using deep latent variable path modelling](https://doi.org/10.1038/s42256-025-01052-4).

## Multimodal Methods

| Method | Purpose |
| --- | --- |
| DLVPM | Deep latent variable path models for multimodal structural modelling |
| CLIP / SimCLR | Contrastive image-text, multimodal, and Siamese representation learning |
| VICReg | Variance-invariance-covariance regularized multimodal learning |
| LeJEPA | Latent joint embedding predictive architecture for multimodal views |
| DGCCA | Deep Generalised Canonical Correlation Analysis |

This package also contains implementations of Deep Generalised Canonical Correlation Analysis and multimodal adaptations of VICReg, LeJEPA, and CLIP/SimCLR. Each method can be used to connect multimodal datasets or to learn representations of a single data type.

![Chord animation](https://raw.githubusercontent.com/alexjamesing/Deep_LVPM/master/chord_animation.gif)

The animation above shows model training in progress on a three-factor DLVPM model linking omics and imaging data types from lung cancer patients. The dataset used for this example is included in the package.

## Installation

Deep LVPM supports Python `3.11` and `3.12`.

Create and activate a clean environment:

```bash
conda create -n dlvpm-torch python=3.11 -y
conda activate dlvpm-torch
```

Install the package from PyPI:

```bash
pip install deep-lvpm
```

To install directly from GitHub:

```bash
pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm"
```

For NVIDIA CUDA, install the CUDA-enabled PyTorch wheel for your platform first, then install Deep LVPM:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install deep-lvpm
```

For an editable local install:

```bash
git clone https://github.com/alexjamesing/Deep_LVPM.git
cd Deep_LVPM
pip install -e ".[tutorials,dev]"
```

Useful extras are:

- `deep-lvpm[tutorials]` for the standard tutorials.
- `deep-lvpm[coco]` for the MS COCO image-text tutorial.
- `deep-lvpm[survival]` for the TCGA survival analysis dependencies.
- `deep-lvpm[docs]` for building the documentation.
- `deep-lvpm[dev]` for tests, package builds, and metadata checks.

Verify the install:

```bash
python -c "import torch, deep_lvpm; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"
```

Apple Silicon uses standard PyTorch wheels with MPS support where available. CUDA-enabled PyTorch wheels should be installed from the PyTorch index for your platform and driver.

## Tutorials And Metrics

Turnkey tutorials ship with the toolbox and use native PyTorch. Launch them with:

- `python -m deep_lvpm.tutorial.tutorial_mnist` - associate MNIST images with labels and visualise the latent space.
- `python -m deep_lvpm.tutorial.tutorial_tcga` - integrate five TCGA lung cancer modalities using residual encoders.
- `python -m deep_lvpm.tutorial.tutorial_siamese` - train a residual PyTorch Siamese encoder on CIFAR-10 and compare linear probes on final DLVPM factors and average-pooled convolutional features.
- `python -m deep_lvpm.tutorial.tutorial_coco` - train an image-text model on MS COCO and benchmark true five-caption retrieval for DLVPM/CLIP/VICReg.
- `python -m deep_lvpm.tutorial.tutorial_tcga_survival` - run the TCGA pan-cancer survival example with PyTorch encoders, integrated gradients, and PyTorch checkpoints.

All tutorials report the expanded `StructuralModel.evaluate` metrics (`total_loss`, `cross_metric`, `mse_loss`, and `redundancy`) so you can monitor both cross-view alignment and within-view redundancy.

## Citation

If you use Deep LVPM, please cite:

Ing A, Andrades A, Cosenza MR, Korbel JO. Integrating multimodal cancer data using deep latent variable path modelling. Nature Machine Intelligence 7, 1053-1075 (2025). https://doi.org/10.1038/s42256-025-01052-4
